"""Tests for src/serving/resume.py using made-up resumes (no real people, no network)."""

import json
import logging
import time

import pytest

from src.serving import resume
from src.serving.resume import ResumeError, SkillExtractor, extract_skills
from src.utils import MODELS
from tests.resume_samples import RESUME_LINES, make_docx, make_pdf

# Small deterministic vocabulary + synonyms so the tests do not depend on DVC-pulled files.
VOCAB = [
    "python", "sql", "react.js", "machine learning", "node.js", "tally", "excel", "c", "r", "c++", "javascript",
    "communication skills", "data", "core", "sales", "html", "css", "html/css", "aws", "kubernetes", "learning",
    "data analysis", "go", "ci/cd", "ms office", "e-commerce", "team", "case", "senior",
    "powerpoint", "bilingual", "spreadsheets", "company", "program", "leadership", "interpersonal", "excel powerpoint",
    "jee", "java", "english", "hindi", "bash", "flask", "monitoring", "events", "t", "redux", "git", "angular",
]  # fmt: skip
ALIASES = {
    "react": "react.js", "reactjs": "react.js", "react js": "react.js", "ml": "machine learning",
    "node js": "node.js", "nodejs": "node.js", "tally erp 9": "tally", "ms excel": "excel",
    "communication": "communication skills", "microsoft office": "ms office", "ecommerce": "e-commerce",
}  # fmt: skip


KEYNESS = {"company": 0.009, "program": 0.018, "python": 0.57, "sql": 0.38, "excel": 0.19, "leadership": 0.027,
           "interpersonal": 0.004, "sales": 0.43, "data": 0.002}  # fmt: skip


@pytest.fixture(scope="module")
def ex() -> SkillExtractor:
    return SkillExtractor(VOCAB, ALIASES)


@pytest.fixture(scope="module")
def kx() -> SkillExtractor:
    """Same vocabulary, with keyness scores like models/skill_keyness.json."""
    return SkillExtractor(VOCAB, ALIASES, keyness=KEYNESS)


def names(ex, text):
    return [s.skill for s in ex.from_text(text)]


# ── spelling and phrases ───────────────────────────────────────────────────────


def test_synonyms_come_back_in_model_spelling(ex):
    got = names(ex, "SKILLS\nReactJS, Machine Learning, Node JS, Tally ERP 9, MS Excel, Microsoft Office")
    assert {"react.js", "machine learning", "node.js", "tally", "excel", "ms office"} <= set(got)
    assert "reactjs" not in got and "ms excel" not in got


def test_longest_phrase_wins(ex):
    got = names(ex, "I use machine learning every day, and I like learning.")
    assert "machine learning" in got
    assert "learning" not in got  # the lone "learning" is inside the longer phrase / generic


def test_special_character_skills(ex):
    got = names(ex, "SKILLS\nC++, Node.js, CI/CD, E-Commerce, HTML/CSS")
    assert {"c++", "node.js", "ci/cd", "e-commerce", "html/css", "html", "css"} <= set(got)


def test_case_and_punctuation_noise(ex):
    assert "python" in names(ex, "Worked in (PYTHON), sql; and Excel.")
    assert {"sql", "excel"} <= set(names(ex, "Worked in (PYTHON), sql; and Excel."))


# ── false-positive control ─────────────────────────────────────────────────────


def test_short_skills_need_skills_context(ex):
    prose = "I reported to the C suite and joined Plan R. Let's Go!"
    assert not {"c", "r", "go"} & set(names(ex, prose))
    listed = names(ex, "SKILLS\nPython, C, R, Go")
    assert "c" in listed and "r" in listed


def test_short_skills_need_capital_letter(ex):
    assert "r" not in names(ex, "SKILLS\nr, python")  # a lone lower-case 'r' is noise


def test_ml_alias_accepted_only_in_caps(ex):
    assert "machine learning" in names(ex, "Research in ML and deep nets.")
    assert "machine learning" not in names(ex, "the ml of the story")


def test_generic_words_need_skills_section_or_repeats(ex):
    once = names(ex, "Strong communication and data skills with sales exposure on a core team.")
    assert not {"communication skills", "data", "sales", "core"} & set(once)
    listed = names(ex, "SKILLS\nCommunication, Data, Sales")
    assert {"communication skills", "data", "sales"} <= set(listed)
    repeated = names(ex, "Led sales calls. Closed sales deals. Owned sales targets.")
    assert "sales" in repeated


def test_english_words_only_inside_skills_section(ex):
    assert not {"case", "team", "senior"} & set(names(ex, "Senior engineer, case study, team player"))
    assert "team" in names(ex, "SKILLS\nTeam, Python")


# ── structure ──────────────────────────────────────────────────────────────────


def test_skills_section_ranks_first(ex):
    out = ex.from_text("\n".join(RESUME_LINES))
    by = {s.skill: s for s in out}
    assert by["python"].in_skills_section and not by["kubernetes"].in_skills_section
    assert [s.skill for s in out].index("python") < [s.skill for s in out].index("kubernetes")


def test_exam_names_are_not_skills_outside_a_skills_section(ex):
    # "jee" is Java EE in job ads, but on a student resume it is the entrance exam.
    got = names(ex, "EDUCATION\nJEE Main 2021: 98.2 percentile\n\nSKILLS\nPython, Java")
    assert "jee" not in got and {"python", "java"} <= set(got)
    assert "jee" in names(ex, "SKILLS\nJava, JEE, SQL")  # listed as a skill: kept


def test_languages_heading_is_a_skills_section(kx):
    # Students often list programming (and spoken) languages under "Languages": short ones like C / R need a
    # skills context, and spoken ones have low keyness, so both were dropped when the heading counted as "other".
    got = names(kx, "LANGUAGES\nC, R, Bash, SQL\n\nLanguages: English, Hindi")
    assert {"c", "r", "bash", "sql", "english", "hindi"} <= set(got)
    assert "c" not in names(kx, "EXPERIENCE\nReported to the C suite")


def test_split_first_letter_headings_are_repaired(ex):
    # Small caps / drop caps: PDF text says "T echnical Skills" or "S KILLS". The stray "T" is not a skill.
    for heading in ("T echnical Skills", "S KILLS", "T ECHNICAL S KILLS"):
        got = {s.skill: s for s in ex.from_text(f"{heading}\nPython, Git")}
        assert got["python"].in_skills_section and "t" not in got, heading
    got = names(ex, "EXPERIENCE\nFrameworks/T ools: Flask, Git")
    assert {"flask", "git"} <= set(got) and "t" not in got


def test_heading_cell_followed_by_list_without_colon(ex):
    got = {s.skill: s for s in ex.from_text("T echnical Proficiency JavaScript, Python, Redux")}
    assert all(got[k].in_skills_section for k in ("javascript", "python", "redux"))


def test_any_heading_that_names_skills_opens_a_skills_section(ex):
    for heading in ("COURSES AND SKILLS", "Skills & Interests", "Languages & Frameworks: C, R"):
        got = {s.skill: s for s in ex.from_text(f"{heading}\nC, R, Python")}
        assert got["c"].in_skills_section and got["r"].in_skills_section, heading


def test_tech_stack_line_inside_projects_does_not_turn_the_project_into_a_skills_section(ex):
    text = "PROJECTS\nTech Stack: Python, Flask\nBuilt monitoring for the events team"
    got = {s.skill: s for s in ex.from_text(text)}
    assert got["python"].in_skills_section and got["flask"].in_skills_section
    # the project description is prose again (in production, keyness then drops words like these)
    assert not got["monitoring"].in_skills_section and not got["events"].in_skills_section
    wrapped = {s.skill: s for s in ex.from_text("PROJECTS\nTech Stack: Python, Flask,\nRedux, Git\nNotes")}
    assert wrapped["redux"].in_skills_section  # a list that wraps keeps going on the next line


def test_extracurricular_heading_ends_the_skills_section(ex):
    got = {s.skill: s for s in ex.from_text("SKILLS\nPython\nExtracurricular Activity\nI enjoy social events.")}
    assert got["python"].in_skills_section and not got["events"].in_skills_section


def test_versioned_skill_names(ex):
    assert {"python", "angular"} <= set(names(ex, "SKILLS\nPython3, Angular2"))


def test_inline_label_and_section_end(ex):
    text = "Technical Skills: Python, SQL\nEXPERIENCE\nI ran a team and a case study."
    got = names(ex, text)
    assert {"python", "sql"} <= set(got)
    assert "team" not in got and "case" not in got  # the section ended at EXPERIENCE


def test_two_columns_do_not_join_phrases(ex):
    scrambled = "Machine          Python\nlearning         SQL"
    assert "machine learning" not in names(ex, scrambled)


def test_fuzzy_typos_only_in_skills_section(ex):
    assert "javascript" in names(ex, "SKILLS\nPython, Javasript")
    assert "javascript" not in names(ex, "I wrote Javasript for years.")
    off = SkillExtractor(VOCAB, ALIASES, fuzzy=False)
    assert "javascript" not in [s.skill for s in off.from_text("SKILLS\nJavasript")]


def test_every_skill_is_in_the_vocabulary(ex):
    out = ex.from_text("\n".join(RESUME_LINES) + "\nSKILLS\nfoo, bar, baz")
    assert out and all(s.skill in set(VOCAB) for s in out)


def test_top_n_and_empty_text(ex):
    assert len(ex.from_text("SKILLS\nPython, SQL, Excel, AWS, Kubernetes", top_n=2)) == 2
    assert ex.from_text("") == []


# ── files ──────────────────────────────────────────────────────────────────────


def test_pdf_end_to_end(ex):
    res = extract_skills(make_pdf([RESUME_LINES]), "cv.pdf", extractor=ex)
    assert res.code == "ok" and res.message is None
    assert {"python", "react.js", "machine learning", "node.js", "tally"} <= set(res.names)


def test_word_per_line_pdf_falls_back_to_layout_mode(ex, monkeypatch):
    # Google Docs exports write each word as its own text object; pypdf's default mode then returns one word per
    # line, so "TECHNICAL SKILLS" and "Languages : C, ..." fall apart. Simulated here at the pypdf boundary.
    import pypdf

    lines = ["TECHNICAL SKILLS", "Languages : C, Python, SQL", "Web Development : HTML, CSS, React"]
    fragmented = "\n".join(w for ln in lines for w in ln.split())
    calls = []

    def fake_extract(self, *args, extraction_mode="plain", **kwargs):
        calls.append(extraction_mode)
        return "\n".join(lines) if extraction_mode == "layout" else fragmented

    monkeypatch.setattr(pypdf.PageObject, "extract_text", fake_extract)
    out = extract_skills(make_pdf([["placeholder"]]), "cv.pdf", extractor=ex)
    assert calls == ["plain", "layout"]
    by = {s.skill: s for s in out.skills}
    assert {"c", "python", "sql", "html", "css", "react.js"} <= set(by)
    assert all(by[s].in_skills_section for s in ("c", "html", "css"))


def test_normal_pdf_does_not_use_layout_mode(ex, monkeypatch):
    import pypdf

    calls = []
    real = pypdf.PageObject.extract_text

    def spy(self, *args, **kwargs):
        calls.append(kwargs.get("extraction_mode", "plain"))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pypdf.PageObject, "extract_text", spy)
    extract_skills(make_pdf([RESUME_LINES]), "cv.pdf", extractor=ex)
    assert calls == ["plain"]


def test_list_items_on_their_own_line_are_not_headings(ex):
    # One skill per line (seen in word-per-line PDFs): "HTML," must not end the Skills section or be swallowed.
    got = ex.from_text("SKILLS\nC,\nHTML,\nCSS,\nPython")
    assert {"c", "html", "css", "python"} <= set(s.skill for s in got)
    assert all(s.in_skills_section for s in got)


def test_docx_end_to_end_with_table(ex):
    data = make_docx(["Priya Nair", "SKILLS", "Python"], table_rows=[["Tools", "Kubernetes, AWS"], ["Data", "SQL"]])
    res = extract_skills(data, "cv.docx", extractor=ex)
    assert {"python", "kubernetes", "aws", "sql"} <= set(res.names)


def test_scanned_pdf_returns_nothing_with_clear_message(ex):
    res = extract_skills(make_pdf([[]]), "scan.pdf", extractor=ex)
    assert res.skills == [] and res.code == "no_text"
    assert "scanned" in res.message.lower()


def test_no_skills_found_message(ex):
    res = extract_skills(
        make_pdf([["Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod"]]), "a.pdf", extractor=ex
    )
    assert res.skills == [] and res.code == "no_skills" and res.message


def test_long_pdf_is_cut_with_warning(ex):
    pages = [["SKILLS", "Python"]] * (resume.MAX_PAGES + 2)
    res = extract_skills(make_pdf(pages), "long.pdf", extractor=ex)
    assert res.warnings and "python" in res.names


@pytest.mark.parametrize(
    ("data", "name"),
    [
        (b"", "cv.pdf"),
        (b"hello world", "cv.txt"),
        (b"%PDF-1.4 this is not really a pdf", "cv.pdf"),
        (b"PK\x03\x04 not a docx", "cv.docx"),
        (b"PK\x03\x04 old", "cv.doc"),
        (b"not a pdf at all", "cv.pdf"),  # extension and content disagree
    ],
)
def test_bad_uploads_raise_resume_error(ex, data, name):
    with pytest.raises(ResumeError):
        extract_skills(data, name, extractor=ex)


def test_oversize_upload_rejected(ex):
    with pytest.raises(ResumeError, match="too large"):
        extract_skills(b"%PDF-" + b"0" * resume.MAX_BYTES, "big.pdf", extractor=ex)


def test_encrypted_pdf_rejected(ex):
    import io

    from pypdf import PdfReader, PdfWriter

    w = PdfWriter(clone_from=PdfReader(io.BytesIO(make_pdf([RESUME_LINES]))))
    w.encrypt("s3cret")
    buf = io.BytesIO()
    w.write(buf)
    with pytest.raises(ResumeError, match="password"):
        extract_skills(buf.getvalue(), "locked.pdf", extractor=ex)


# ── privacy ────────────────────────────────────────────────────────────────────


def test_resume_text_never_logged_or_in_errors(ex, caplog, tmp_path, monkeypatch):
    secret = "ZEBRA-SECRET-4711"
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.DEBUG):
        extract_skills(make_pdf([["SKILLS", "Python", secret]]), "cv.pdf", extractor=ex)
        with pytest.raises(ResumeError) as err:
            extract_skills(b"%PDF-1.4 " + secret.encode(), "cv.pdf", extractor=ex)
    assert secret not in caplog.text and secret not in str(err.value)
    assert list(tmp_path.iterdir()) == []  # nothing written to disk


def test_module_does_not_use_logging_or_network():
    src = resume.__loader__.get_source(resume.__name__)
    for banned in ("import logging", "requests", "urllib", "socket", "open(", "tempfile", "print("):
        assert banned not in src, banned


# ── speed ──────────────────────────────────────────────────────────────────────


def test_fast_enough_for_a_tiny_cpu(ex):
    data = make_pdf([RESUME_LINES] * 3)
    t = time.perf_counter()
    for _ in range(20):
        extract_skills(data, "cv.pdf", extractor=ex)
    assert (time.perf_counter() - t) / 20 < 0.5  # seconds per resume, far under the 'few seconds' budget


# ── the real vocabulary (skipped when the DVC models are not pulled) ───────────


@pytest.mark.skipif(not (MODELS / "skill_vocab.json").exists(), reason="run `dvc pull` first")
def test_real_vocabulary_spellings():
    real = resume.get_extractor()
    vocab = set(json.loads((MODELS / "skill_vocab.json").read_text(encoding="utf-8")))
    out = [s.skill for s in real.from_text("SKILLS\nReactJS, Machine Learning, Node JS, Tally ERP 9, MS Excel, C++, R")]
    assert {"react.js", "machine learning", "node.js", "tally", "excel", "c++", "r"} <= set(out)
    assert set(out) <= vocab


# ── regressions found while evaluating on real resumes ─────────────────────────


def test_phrases_never_span_commas(ex):
    # "excel powerpoint" is a vocabulary entry; a list "Excel, PowerPoint" must not match it
    got = names(ex, "SKILLS\nExcel, PowerPoint, Python")
    assert "excel powerpoint" not in got and {"excel", "powerpoint", "python"} <= set(got)


def test_spacing_hyphen_and_plural_variants(ex):
    got = names(ex, "SKILLS\nPower Point, Bi-lingual, Spreadsheet preparation")
    assert {"powerpoint", "bilingual", "spreadsheets"} <= set(got)


def test_single_capital_skill_line_is_not_a_heading(ex):
    # an all-caps one-word line such as "SQL" must not be mistaken for a section heading
    got = names(ex, "SKILLS\nPython\nSQL\nEXCEL")
    assert {"python", "sql", "excel"} <= set(got)


def test_highlights_and_qualifications_count_as_skills_sections(ex):
    assert "team" in names(ex, "Highlights\nTeam, Python")
    assert "team" in names(ex, "Core Qualifications\nTeam, Python")


# ── keyness: everyday words are ignored, real skills and soft skills in lists survive ──


def test_everyday_words_ignored_in_prose_even_when_repeated(kx):
    prose = "The company gave me a program. The company ran a program. Python and SQL were used."
    got = names(kx, prose)
    assert "company" not in got and "program" not in got
    assert {"python", "sql"} <= set(got)


def test_keyness_junk_in_skills_lists_dropped_but_soft_skills_kept(kx):
    got = names(kx, "SKILLS\nCompany, Data, Leadership, Interpersonal, Python")
    assert "company" not in got and "data" not in got
    assert {"leadership", "interpersonal", "python"} <= set(got)


def test_soft_skill_in_prose_needs_keyness(kx):
    assert "leadership" not in names(kx, "Showed leadership during the project.")


def test_keyness_ranks_real_skills_above_list_noise(kx):
    out = kx.from_text("SKILLS\nSales, Leadership, Python, SQL")
    order = [s.skill for s in out]
    assert order.index("python") < order.index("leadership")


def test_extractor_works_without_keyness_file(ex):
    assert "python" in names(ex, "SKILLS\nPython")  # keyness is optional


def test_build_keyness_counts_and_threshold():
    import pandas as pd

    from src.serving.keyness import build_keyness

    ads = pd.DataFrame(
        {
            "desc": ["python and sql wanted at the company", "company news; python", "company only", "python"],
            "skills": ["Python, SQL", "Python", "", "Python"],
        }
    )
    got = build_keyness(
        ["python", "sql", "company", "machine learning"], [(ads, "desc", "skills", ",")], {}, min_docs=2
    )
    assert got == {"company": 0.0, "python": 1.0}  # sql mentioned once -> below min_docs; phrases are skipped


# ── the evaluation labels ──────────────────────────────────────────────────────


def test_eval_gold_file_is_well_formed_and_contains_no_resume_text():
    import json as _json

    from src.utils import ROOT

    doc = _json.loads((ROOT / "tests/data/resume_eval_gold.json").read_text(encoding="utf-8"))
    assert doc["licence"].startswith("CC0")
    for part in ("resumes", "holdout"):
        assert doc[part], part
        for rid, rec in doc[part].items():
            assert rid.isdigit() and set(rec) == {"category", "skills"}  # ids and skills only, never text
            assert rec["skills"] == sorted(set(rec["skills"]))
    assert not set(doc["resumes"]) & set(doc["holdout"])  # the holdout never overlaps the tuning set


@pytest.mark.skipif(not (MODELS / "skill_vocab.json").exists(), reason="run `dvc pull` first")
def test_every_gold_skill_is_in_the_model_vocabulary():
    import json as _json

    from src.utils import ROOT

    vocab = set(_json.loads((MODELS / "skill_vocab.json").read_text(encoding="utf-8")))
    doc = _json.loads((ROOT / "tests/data/resume_eval_gold.json").read_text(encoding="utf-8"))
    bad = [
        (r, s)
        for part in ("resumes", "holdout")
        for r, rec in doc[part].items()
        for s in rec["skills"]
        if s not in vocab
    ]
    assert not bad, bad[:5]
