"""Skill "keyness": which vocabulary words are real skills, and which are just everyday language.

    models/skill_keyness.json   {single-word skill: P(posting lists it as a skill | its description mentions it)}

A resume says "company", "issue", "program" all the time and those are in the 6,000-skill vocabulary (people
tagged them), so matching the vocabulary against resume text drowns the real skills in noise. But a job ad that
*mentions* "python" lists it as a skill tag 57% of the time, "sql" 38%, "excel" 19%, while an ad that mentions
"company" lists it as a tag 0.9% of the time and "customer" 0.1%. That ratio separates tools and trades from
everyday words without a hand-made stoplist, using the same postings the role model learned from.

Used by src/serving/resume.py (words under a threshold are ignored outside a Skills section and ranked lower
inside one). Built by the `resume_keyness` DVC stage:  uv run python -m src.serving.keyness
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from src.utils import MODELS, RAW, get_logger, load_aliases, load_params, parse_skills, read_json, write_json

log = get_logger("keyness")

_WORD = re.compile(r"[a-z][a-z0-9+#.]*[a-z0-9+#]|[a-z0-9]")

SOURCES = {  # snapshot -> (file, description column, skills column, separator of the skills column)
    2025: (RAW / "naukri_2025" / "naukri_2025.xlsx", "jobDescription", "tagsAndSkills", r","),
    2022: (RAW / "naukri_2022" / "naukri_2022.csv", "job_description", "required_skills", r"[\n|]"),
}


def _read(path, desc_col: str, skills_col: str) -> pd.DataFrame:
    cols = [desc_col, skills_col]
    return pd.read_excel(path, usecols=cols) if path.suffix == ".xlsx" else pd.read_csv(path, usecols=cols)


def build_keyness(vocab: list[str], frames: list[tuple[pd.DataFrame, str, str, str]], aliases: dict, min_docs: int):
    """frames: (dataframe, description column, skills column, skills separator). Returns {word: keyness}."""
    singles = {s for s in vocab if " " not in s}
    mentioned: Counter = Counter()  # postings whose description contains the word
    tagged: Counter = Counter()  # ... and that also list it as a skill
    n = 0
    for df, desc_col, skills_col, sep in frames:
        for desc, raw_skills in zip(df[desc_col].fillna(""), df[skills_col].fillna(""), strict=True):
            words = set(_WORD.findall(str(desc).lower())) & singles
            tags = set(parse_skills(raw_skills, sep, aliases))
            n += 1
            for w in words:
                mentioned[w] += 1
                if w in tags:
                    tagged[w] += 1
    log.info("%d postings, %d single-word skills", n, len(singles))
    return {w: round(tagged[w] / mentioned[w], 4) for w in sorted(singles) if mentioned[w] >= min_docs}


def main() -> None:
    p = load_params()["resume"]["keyness"]
    vocab = read_json(MODELS / "skill_vocab.json")
    frames = []
    for snap in p["sources"]:
        path, desc_col, skills_col, sep = SOURCES[snap]
        frames.append((_read(path, desc_col, skills_col), desc_col, skills_col, sep))
    keyness = build_keyness(vocab, frames, load_aliases(), p["min_docs"])
    write_json(keyness, MODELS / "skill_keyness.json")
    log.info("keyness for %d of the vocabulary's single-word skills", len(keyness))


if __name__ == "__main__":
    main()
