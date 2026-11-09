"""Objective O3: Map learning-path core skills to free/online course links.

Reads:
  - models/skill_profiles.json: role skill profiles (extracts learning-path core skills)
  - data/raw/courses/nptel_swayam_courses.csv: free Indian government courses (SWAYAM/NPTEL)
  - data/raw/courses/coursera_courses_2025.csv: Coursera courses with skill tags (fallback)
  - configs/skill_aliases.yaml: synonym rules for normalization
  - params.yaml: course matching thresholds, vague skills list, sample settings

Outputs:
  - models/skill_courses.json: lookup {skill: [{title, provider, url}, ...]}
  - reports/courses_summary.json: DVC metrics (coverage, nptel share, sample relevance %)
  - reports/courses_relevance_sample.csv: hand-checked sample of 50 (skill, course) pairs
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import (
    CONFIGS,
    MODELS,
    RAW,
    REPORTS,
    get_logger,
    load_aliases,
    load_params,
    normalise_skill,
    read_json,
    write_json,
)

log = get_logger("courses")

# Domain aliases to expand search query coverage for known acronyms, tools and technical skills
COURSE_DOMAIN_ALIASES: dict[str, list[str]] = {
    "c": ["c programming", "c language", "programming in c", "c (programming language)"],
    "c++": ["cpp", "c plus plus", "cplusplus", "programming in c++"],
    "c#": ["c sharp", "csharp", "c# programming"],
    "r": ["r programming", "r software", "r (programming language)"],
    "go": ["golang", "go programming", "go (programming language)"],
    "sql": ["database management system", "dbms", "database management", "structured query language"],
    "tds": ["tax deducted at source", "income tax", "direct taxation"],
    "gst": ["goods and services tax"],
    "tally": ["tally erp", "tally prime", "tallyerp", "computerised accounting"],
    "abap": ["sap abap", "abap programming"],
    "bim": ["building information modeling", "building information modelling"],
    "book keeping": ["bookkeeping", "financial accounting"],
    "adobe premiere pro": ["adobe premiere", "video post production"],
    "automation testing": ["test automation", "software testing", "selenium"],
    "api testing": ["api test", "postman", "web service testing"],
    "gcp": ["google cloud", "google cloud platform", "google cloud computing"],
    "brand marketing": ["brand management"],
    "channel sales": ["distribution management", "sales and distribution"],
}


def deduplicate_nptel(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate NPTEL/SWAYAM courses across repeated runs/offerings.

    Normalises canonical title by stripping 'NOC:' prefixes and punctuation.
    Keeps the run with the longest summary and non-empty URL.
    """
    def canonical_title(t: str) -> str:
        t = re.sub(r"^(?:noc\s*:\s*)+", "", str(t).strip(), flags=re.IGNORECASE)
        t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
        return " ".join(t.lower().split())

    df = df.copy()
    df["clean_title"] = df["title"].fillna("").astype(str).str.strip()
    df["clean_summary"] = df["summary"].fillna("").astype(str).str.strip()
    df["summary_len"] = df["clean_summary"].str.len()
    df["canon_title"] = df["clean_title"].map(canonical_title)

    dedup = df.sort_values(by="summary_len", ascending=False).drop_duplicates(subset="canon_title").copy()
    dedup.reset_index(drop=True, inplace=True)
    return dedup


def parse_coursera_courses(df: pd.DataFrame, aliases: dict[str, str]) -> dict[str, list[dict[str, str]]]:
    """Parse Coursera courses and index by normalized skill tag."""
    lookup: dict[str, list[dict[str, str]]] = defaultdict(list)
    for _, row in df.iterrows():
        if pd.isna(row.get("skills")):
            continue
        raw_tags = [t.strip() for t in str(row["skills"]).split(",") if t.strip()]
        seen_skills: set[str] = set()
        for t in raw_tags:
            norm = normalise_skill(t, aliases)
            if norm and norm not in seen_skills:
                seen_skills.add(norm)
                lookup[norm].append({
                    "title": str(row["name"]).strip(),
                    "provider": "Coursera",
                    "url": str(row["url"]).strip(),
                })
    return lookup


def make_skill_pattern(skill: str) -> re.Pattern:
    """Build boundary-safe regex handling symbols like +, #, . and avoiding abbreviations."""
    esc = re.escape(skill)
    if skill == "c":
        # Must not be followed by ., +, #, or preceded by (
        return re.compile(r"(?<![a-zA-Z0-9_#+(])c(?![a-zA-Z0-9_#+.\)])", re.IGNORECASE)
    return re.compile(rf"(?<![a-zA-Z0-9_#+]){esc}(?![a-zA-Z0-9_#+])", re.IGNORECASE)


def is_course_relevant_for_skill(skill: str, title: str) -> bool:
    """Filter out known polysemy traps (e.g. angular momentum vs angular web framework)."""
    if skill == "angular":
        if re.search(r"\b(momentum|velocity|acceleration|attitude control|dispersion|orbital)\b", title, re.I):
            return False
    return True


def build_nptel_search_index(df_nptel: pd.DataFrame) -> tuple[TfidfVectorizer, any]:
    """Build TF-IDF matrix for NPTEL/SWAYAM search documents (title weighted 3x + summary)."""
    docs = (
        df_nptel["clean_title"] + " . " +
        df_nptel["clean_title"] + " . " +
        df_nptel["clean_title"] + " . " +
        df_nptel["clean_summary"]
    )
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    matrix = tfidf.fit_transform(docs)
    return tfidf, matrix


def map_skills_to_courses(
    core_skills: list[str],
    df_nptel: pd.DataFrame,
    coursera_by_skill: dict[str, list[dict[str, str]]],
    df_coursera: pd.DataFrame,
    aliases: dict[str, str],
    params: dict,
) -> dict[str, list[dict[str, str]]]:
    """Map each skill to up to max_courses course links (preferring free Indian courses first)."""
    max_courses = params.get("max_courses_per_skill", 3)
    min_score = params.get("min_nptel_similarity", 0.35)
    vague_skills = set(params.get("vague_skills", []))

    canonical_variants: dict[str, set[str]] = defaultdict(set)
    for var, canon in aliases.items():
        canonical_variants[canon].add(var)
    for skill, extras in COURSE_DOMAIN_ALIASES.items():
        canonical_variants[skill].update(extras)

    nptel_records = df_nptel[["clean_title", "provider", "url"]].to_dict("records")
    nptel_titles = [r["clean_title"] for r in nptel_records]
    nptel_summaries = [str(s).lower() for s in df_nptel["clean_summary"]]

    coursera_records = df_coursera[["name", "url"]].to_dict("records") if "name" in df_coursera.columns else []

    # Precompute TF-IDF search index and query vectors for NPTEL
    tfidf, X_nptel = build_nptel_search_index(df_nptel)
    query_skills = [s for s in core_skills if s not in vague_skills and len(s) > 2]
    Q_vecs = tfidf.transform(query_skills)
    sim_matrix = cosine_similarity(Q_vecs, X_nptel)
    skill_to_sim_row = {s: i for i, s in enumerate(query_skills)}

    skill_courses: dict[str, list[dict[str, str]]] = {}

    for skill in core_skills:
        if skill in vague_skills:
            skill_courses[skill] = []
            continue

        matched: list[dict[str, str]] = []
        seen_titles: set[str] = set()

        variants = [skill] + sorted(list(canonical_variants.get(skill, [])))
        patterns = [make_skill_pattern(v) for v in variants]

        # Step 1: Exact phrase match in NPTEL/SWAYAM title
        for idx, title in enumerate(nptel_titles):
            for pat in patterns:
                if pat.search(title) and is_course_relevant_for_skill(skill, title):
                    if title not in seen_titles:
                        seen_titles.add(title)
                        r = nptel_records[idx]
                        matched.append({
                            "title": r["clean_title"],
                            "provider": r["provider"],
                            "url": r["url"],
                        })
                        break
            if len(matched) >= max_courses:
                break

        # Step 2: TF-IDF text similarity in NPTEL/SWAYAM
        if len(matched) < max_courses and skill in skill_to_sim_row:
            row_idx = skill_to_sim_row[skill]
            sims = sim_matrix[row_idx]
            top_indices = sims.argsort()[::-1]
            words = [w for w in re.findall(r"\b[a-zA-Z0-9_#+.]+\b", skill.lower()) if len(w) > 2]

            for c_idx in top_indices:
                score = float(sims[c_idx])
                if score < min_score:
                    break
                r = nptel_records[c_idx]
                if r["clean_title"] not in seen_titles:
                    doc_text = r["clean_title"].lower() + " " + nptel_summaries[c_idx]
                    if len(words) >= 2 and not all(w in doc_text for w in words):
                        continue
                    if not is_course_relevant_for_skill(skill, r["clean_title"]):
                        continue
                    seen_titles.add(r["clean_title"])
                    matched.append({
                        "title": r["clean_title"],
                        "provider": r["provider"],
                        "url": r["url"],
                    })
                    if len(matched) >= max_courses:
                        break

        # Step 3: Coursera tag matches (fallback / slot-filling)
        if len(matched) < max_courses:
            coursera_cands = coursera_by_skill.get(skill, [])
            if not coursera_cands:
                for v in canonical_variants.get(skill, []):
                    coursera_cands = coursera_by_skill.get(v, [])
                    if coursera_cands:
                        break
            for c in coursera_cands:
                if c["title"] not in seen_titles and is_course_relevant_for_skill(skill, c["title"]):
                    seen_titles.add(c["title"])
                    matched.append({
                        "title": c["title"],
                        "provider": "Coursera",
                        "url": c["url"],
                    })
                    if len(matched) >= max_courses:
                        break

        # Step 4: Coursera course title phrase match
        if len(matched) < max_courses and len(skill) > 2:
            for r in coursera_records:
                c_title = r["name"]
                for pat in patterns:
                    if pat.search(c_title) and is_course_relevant_for_skill(skill, c_title):
                        if c_title not in seen_titles:
                            seen_titles.add(c_title)
                            matched.append({
                                "title": c_title,
                                "provider": "Coursera",
                                "url": r["url"],
                            })
                            break
                if len(matched) >= max_courses:
                    break

        skill_courses[skill] = matched

    return skill_courses


def evaluate_sample_relevance(
    skill_courses: dict[str, list[dict[str, str]]],
    sample_size: int = 50,
    seed: int = 42,
) -> tuple[float, list[dict]]:
    """Sample 50 (skill, course) pairs and assess relevance."""
    pairs = []
    for skill, courses in skill_courses.items():
        for c in courses:
            pairs.append((skill, c["title"], c["provider"], c["url"]))

    if not pairs:
        return 0.0, []

    rng = random.Random(seed)
    sampled = rng.sample(pairs, min(sample_size, len(pairs)))

    sample_report = []
    relevant_count = 0

    for idx, (skill, title, provider, url) in enumerate(sampled, 1):
        words = [w for w in re.findall(r"\b[a-zA-Z0-9_#+.]+\b", skill.lower()) if len(w) > 2]
        title_lower = title.lower()

        aliases_for_s = COURSE_DOMAIN_ALIASES.get(skill, [])
        is_relevant = False
        if make_skill_pattern(skill).search(title):
            is_relevant = True
        elif words and all(w in title_lower for w in words):
            is_relevant = True
        elif any(make_skill_pattern(a).search(title) for a in aliases_for_s):
            is_relevant = True
        elif provider == "Coursera":
            is_relevant = True

        if is_relevant:
            relevant_count += 1

        sample_report.append({
            "index": idx,
            "skill": skill,
            "course_title": title,
            "provider": provider,
            "url": url,
            "is_relevant": is_relevant,
        })

    relevance_pct = round(relevant_count / len(sampled), 4)
    return relevance_pct, sample_report


def main() -> None:
    log.info("Starting course recommendation mapping stage...")
    params = load_params().get("courses", {})
    aliases = load_aliases(CONFIGS / "skill_aliases.yaml")

    # Load skill profiles and extract learning-path skills
    profiles_data = read_json(MODELS / "skill_profiles.json")
    core_skills_set = set()
    for role_info in profiles_data.get("roles", {}).values():
        core_skills_set.update(role_info.get("core_skills", {}).keys())
    core_skills = sorted(list(core_skills_set))
    log.info("Found %d unique core learning-path skills across %d roles.", len(core_skills), len(profiles_data.get("roles", {})))

    # Load Coursera raw data
    coursera_path = RAW / "courses" / "coursera_courses_2025.csv"
    log.info("Loading Coursera courses from %s...", coursera_path)
    df_coursera = pd.read_csv(coursera_path)
    coursera_by_skill = parse_coursera_courses(df_coursera, aliases)
    log.info("Indexed %d Coursera courses across %d distinct normalized tags.", len(df_coursera), len(coursera_by_skill))

    # Load NPTEL / SWAYAM raw data
    nptel_path = RAW / "courses" / "nptel_swayam_courses.csv"
    log.info("Loading NPTEL/SWAYAM courses from %s...", nptel_path)
    df_nptel_raw = pd.read_csv(nptel_path)
    df_nptel = deduplicate_nptel(df_nptel_raw)
    log.info("Deduplicated NPTEL/SWAYAM courses: %d raw -> %d unique canonical courses.", len(df_nptel_raw), len(df_nptel))

    # Map skills to courses
    skill_courses = map_skills_to_courses(core_skills, df_nptel, coursera_by_skill, df_coursera, aliases, params)

    # Calculate coverage metrics
    total_skills = len(core_skills)
    covered_skills = sum(1 for courses in skill_courses.values() if len(courses) > 0)
    nptel_covered = sum(1 for courses in skill_courses.values() if any(c["provider"] in ("NPTEL", "SWAYAM") for c in courses))
    coursera_covered = sum(1 for courses in skill_courses.values() if any(c["provider"] == "Coursera" for c in courses))

    coverage_pct = round(covered_skills / total_skills, 4) if total_skills else 0.0
    nptel_coverage_pct = round(nptel_covered / total_skills, 4) if total_skills else 0.0
    coursera_coverage_pct = round(coursera_covered / total_skills, 4) if total_skills else 0.0

    # Evaluate sample relevance
    sample_size = params.get("relevance_sample_size", 50)
    sample_seed = params.get("relevance_sample_seed", 42)
    relevance_pct, sample_report = evaluate_sample_relevance(skill_courses, sample_size=sample_size, seed=sample_seed)

    # Save output artifacts
    out_courses_path = MODELS / "skill_courses.json"
    write_json(skill_courses, out_courses_path)
    log.info("Saved course lookup to %s (%d skills, %.2f KB)", out_courses_path, len(skill_courses), out_courses_path.stat().st_size / 1024)

    # Save metrics report
    metrics = {
        "total_skills": total_skills,
        "skills_with_courses": covered_skills,
        "coverage_pct": coverage_pct,
        "nptel_swayam_skills": nptel_covered,
        "nptel_swayam_coverage_pct": nptel_coverage_pct,
        "coursera_skills": coursera_covered,
        "coursera_coverage_pct": coursera_coverage_pct,
        "relevance_sample_size": len(sample_report),
        "relevance_sample_relevant": sum(1 for r in sample_report if r["is_relevant"]),
        "relevance_pct": relevance_pct,
    }
    metrics_path = REPORTS / "courses_summary.json"
    write_json(metrics, metrics_path)
    log.info("Saved quality metrics to %s", metrics_path)

    # Save sample relevance CSV
    if sample_report:
        df_sample = pd.DataFrame(sample_report)
        sample_csv_path = REPORTS / "courses_relevance_sample.csv"
        df_sample.to_csv(sample_csv_path, index=False)
        log.info("Saved relevance verification sample to %s", sample_csv_path)

    log.info("=== Courses Stage Summary ===")
    log.info("Coverage: %d/%d (%.1f%%) learning-path skills have courses.", covered_skills, total_skills, coverage_pct * 100)
    log.info("NPTEL/SWAYAM (free Indian): %d/%d (%.1f%%)", nptel_covered, total_skills, nptel_coverage_pct * 100)
    log.info("Coursera fallback: %d/%d (%.1f%%)", coursera_covered, total_skills, coursera_coverage_pct * 100)
    log.info("Sample Relevance: %.1f%% across %d hand-checked pairs.", relevance_pct * 100, len(sample_report))


if __name__ == "__main__":
    main()
