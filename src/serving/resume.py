"""Resume (PDF / DOCX) -> skills, in the role model's own spelling.

    from src.serving.resume import extract_skills
    result = extract_skills(file_bytes, "cv.pdf")
    result.names      # ['python', 'react.js', 'machine learning', ...]  (best first)
    result.message    # None, or a human-readable reason when nothing could be read
    result.skills     # same list with score / count / where it was found

Every returned skill is an entry of the model's vocabulary (models/skill_vocab.json). Spellings go through the
same normaliser and synonym rules as the training data (src.utils.normalise_skill, configs/skill_aliases.yaml),
so "ReactJS" comes back as "react.js" and "ML" as "machine learning".

How it works (no ML model, no network, small memory):
  1. extract text from the file in memory (pypdf / python-docx);
  2. find the "Skills" section so skills listed there rank higher and can be trusted when they are short or generic;
  3. match word n-grams against the vocabulary, longest phrase first, so "machine learning" beats "learning";
  4. gate risky matches: "C", "R", "Go" need a skills-section context; generic words ("communication",
     "management", "sales", "data", "core", ...) need the skills section or repeated use;
  5. optional fuzzy matching (rapidfuzz) for typos, only inside the skills section.

Privacy: the file is processed in memory only. Nothing is written to disk, and nothing from the resume is
logged or put in an exception message.
"""

from __future__ import annotations

import html
import io
import json
import re
import unicodedata
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from src.utils import CONFIGS, MODELS, load_aliases, normalise_skill

# ── limits (free Render server: small CPU, little memory) ──────────────────────
MAX_BYTES = 5_000_000  # reject bigger uploads before parsing
MAX_PAGES = 8  # PDF pages read; a resume longer than this is cut (a warning is returned)
MAX_CHARS = 150_000  # text kept after extraction
MAX_DOCX_UNCOMPRESSED = 40_000_000  # zip-bomb guard for .docx
MAX_NGRAM = 6  # longest vocabulary phrase has 6 words
MIN_TEXT_CHARS = 40  # fewer extracted characters than this = treated as a scan / empty file

# ── words that are skills in our vocabulary but also ordinary prose ────────────
# Accepted only when found in a skills section, or when the resume uses them repeatedly.
GENERIC = frozenset(
    """
    communication management sales data core development design testing analysis research planning support service
    services training leadership teaching marketing operations business customer project team quality security
    technical technology software system systems application applications engineering process processes
    reporting english hindi education learning problem solving teamwork knowledge computer computers
    administration coordination consulting execution implementation strategy strategic digital online
    """.split()
)
GENERIC_MIN_COUNT = 2

# models/skill_keyness.json: for a single-word skill, how often a job ad that mentions the word also lists it as a
# skill tag (see src/serving/keyness.py). 'python' 0.57, 'sql' 0.38, 'company' 0.009. Words below this value are
# everyday language and are ignored outside a Skills section.
KEYNESS_OUTSIDE = 0.12
KEYNESS_INSIDE = 0.01  # inside a Skills section a soft skill ("leadership") is trusted; raise to drop junk list words

# Ordinary English words that happen to be vocabulary entries. Prose is full of them ("case study", "set up",
# "team"), so they only count when they sit inside a skills section.
STRICT = frozenset(
    """
    act art auto bank bot boot box bulk bus busy can cap case cash chat cold com cook core cost data debt demo
    desk dice dies dot draw end feed fire fix flex flow gas gen glue gold grid hi key lab lead lean line load
    loan mean mice min mob news oil opt part pay post pre raw rest risk room safe sale set side site six skin
    snow soft sol sub target team thai time top unit usa uk word wire yoga senior junior manager executive associate
    analyst engineer developer intern fresher trainee strong good excellent experience experienced
    """.split()
) | frozenset(
    # Exam and school-board names. Some are vocabulary skills with another meaning ("jee" = Java EE in job ads),
    # but on a resume they almost always sit under Education ("JEE Main 2021: 98 percentile").
    "jee neet gate cat gre gmat cet ssc hsc cbse icse isc upsc ielts toefl sat clat".split()
)

# Soft skills that job-ad tags rarely carry (so keyness is low) but that people do list in a Skills section.
SOFT_SKILLS = frozenset(
    "leadership interpersonal teamwork innovation adaptability mentoring coaching creativity collaboration negotiation"
    " motivation initiative flexibility".split()
)

# Very short surface forms ("C", "R", "Go", "ML") are only trusted with the right casing in a skills context.
SHORT_LEN = 2
SHORT_OK_ANYWHERE = frozenset({"ML", "AI", "JS", "BI", "QA", "HR", "UI", "UX", "CI", "CD", "TS"})

# ── section headers ─────────────────────────────────────────────────────────────


def _phrases(block: str) -> frozenset[str]:
    return frozenset(p.strip() for p in block.replace("\n", " ").split("|") if p.strip())


SKILL_HEADERS = _phrases(
    """
    skills|skill|skill set|skillset|key skills|core skills|technical skills|professional skills|it skills|
    soft skills|hard skills|skills summary|summary of skills|skills and abilities|skills & abilities|
    skills and expertise|skills & expertise|areas of expertise|expertise|technical expertise|core competencies|
    competencies|key competencies|technical proficiencies|technical proficiency|proficiencies|technologies|
    technology|technology skills|tools|tools and technologies|tools & technologies|technical summary|
    technical knowledge|computer skills|software skills|skills and qualifications|skills & qualifications|
    skill highlights|highlights|key qualifications|core qualifications|qualifications|programming languages|
    frameworks|libraries|databases|platforms|operating systems|
    methodologies|cloud|web technologies|technical environment|tech stack|
    languages|languages known|known languages|spoken languages|language skills|language proficiency|
    coding languages|programming|programming skills|computer languages
    """
)
OTHER_HEADERS = _phrases(
    """
    experience|work experience|professional experience|employment|employment history|work history|
    career history|professional background|relevant experience|summary|professional summary|career summary|
    executive summary|objective|career objective|profile|professional profile|about me|about|education|
    academic background|education and training|education & training|academic qualifications|projects|
    academic projects|personal projects|certifications|certificates|licenses|licenses & certifications|
    awards|achievements|accomplishments|publications|references|interests|hobbies|volunteering|activities|
    extracurricular|personal details|personal information|declaration|contact|contact information|training|
    internships|internship|positions of responsibility|courses
    """
)
SKILLS_SECTION_MAX_LINES = 30  # an unterminated "Skills" section stops being trusted after this many lines
_HEADER_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z &/,\-]{1,60}?)\s*[:\-–—]?\s*$")
_INLINE_LABEL = re.compile(r"^\s*([A-Za-z][A-Za-z &/\-]{1,40}?)\s*[:–—]\s*(\S.*)$")
# A skill phrase never spans a comma, semicolon, bracket, bullet, column gap or sentence end,
# so "Excel, PowerPoint" can not match a vocabulary entry called "excel powerpoint".
_SEGMENT_SPLIT = re.compile(r"[,;:()\[\]{}\t\u2022\u00b7\u25cf\u25aa\u25a0\u25e6|]|\s{3,}|(?<=[a-z0-9])\.\s")
_TOKEN_EDGE = "\"'`,;:()[]{}<>*•·|!?“”‘’"
_SLASH_SPACES = re.compile(r"(?<=\w)\s+/\s+(?=\w)")


class ResumeError(ValueError):
    """The upload cannot be processed (wrong type, too big, corrupt, encrypted). The message is safe to show."""


@dataclass(frozen=True)
class ExtractedSkill:
    skill: str  # canonical, in the model's vocabulary
    score: float  # higher = more confident; the list is sorted by it
    count: int  # how many times it was found
    in_skills_section: bool
    fuzzy: bool = False  # found through typo-tolerant matching


@dataclass
class ResumeResult:
    skills: list[ExtractedSkill] = field(default_factory=list)
    message: str | None = None  # why nothing (or little) was returned, in plain words
    code: str = "ok"  # ok | no_text | no_skills | truncated
    warnings: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [s.skill for s in self.skills]


# ── file -> text ────────────────────────────────────────────────────────────────


def _kind(data: bytes, filename: str) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext == ".pdf" and data[:5] == b"%PDF-":
        return "pdf"
    if ext == ".docx" and data[:2] == b"PK":
        return "docx"
    if ext == ".doc":
        raise ResumeError("Old .doc files are not supported. Please save the resume as .docx or PDF.")
    raise ResumeError("Unsupported file. Please upload a PDF or a .docx file.")


FRAGMENTED_WORDS_PER_LINE = 1.5  # a normal PDF gives ~6 words per line; a word-per-line extraction gives ~1


def _words_per_line(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return sum(len(ln.split()) for ln in lines) / len(lines) if lines else 0.0


def _pdf_text(data: bytes, warnings: list[str]) -> str:
    from pypdf import PdfReader  # imported lazily: keeps start-up memory small
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ResumeError("This PDF is password-protected. Please upload an unlocked copy.")
        pages = reader.pages
        if len(pages) > MAX_PAGES:
            warnings.append(f"Only the first {MAX_PAGES} pages were read.")
        n = min(len(pages), MAX_PAGES)
        text = "\n".join((pages[i].extract_text() or "") for i in range(n))
        if text.strip() and _words_per_line(text) < FRAGMENTED_WORDS_PER_LINE:
            # Some exporters (Google Docs) write every word as its own text object, and the default mode then
            # returns one word per line: headings and "Label: a, b" lines fall apart. Layout mode rebuilds the
            # lines from where the words sit on the page. Not the default: it can merge two-column layouts.
            text = "\n".join((pages[i].extract_text(extraction_mode="layout") or "") for i in range(n))
        return text
    except ResumeError:
        raise
    except (PyPdfError, ValueError, KeyError, TypeError, OSError, RecursionError) as e:
        raise ResumeError("This PDF could not be read. It may be damaged.") from e


def _docx_text(data: bytes) -> str:
    import docx  # python-docx, lazy import

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if sum(i.file_size for i in z.infolist()) > MAX_DOCX_UNCOMPRESSED:
                raise ResumeError("This document is too large to process.")
        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs]
        for table in d.tables:  # skills are often laid out in tables
            for row in table.rows:
                seen = set()
                for cell in row.cells:
                    if id(cell._tc) not in seen:  # merged cells repeat
                        seen.add(id(cell._tc))
                        parts.append(cell.text)
        return "\n".join(parts)
    except ResumeError:
        raise
    except (zipfile.BadZipFile, KeyError, ValueError, OSError) as e:
        raise ResumeError("This document could not be read. It may be damaged.") from e


# ── the matcher ─────────────────────────────────────────────────────────────────


class SkillExtractor:
    """Holds the vocabulary index. Build once and reuse (see get_extractor)."""

    def __init__(
        self,
        vocab: list[str],
        aliases: dict[str, str] | None = None,
        fuzzy: bool = True,
        expand_compounds: bool = False,
        keyness: dict[str, float] | None = None,
        keyness_outside: float = KEYNESS_OUTSIDE,
        keyness_inside: float = KEYNESS_INSIDE,
    ):
        self.vocab = set(vocab)
        self.index: dict[str, str] = {s: s for s in self.vocab}
        for variant, canon in (aliases or {}).items():
            if canon in self.vocab:
                self.index.setdefault(variant, canon)
        for key, canon in list(self.index.items()):  # "e commerce" should find "e-commerce"
            if "-" in key:
                self.index.setdefault(key.replace("-", " "), canon)
        self.squashed: dict[str, str] = {}  # "power point" / "bi-lingual" -> "powerpoint" / "bilingual"
        for key, canon in self.index.items():
            sq = re.sub(r"[\s\-]", "", key)
            if len(sq) >= 6 and sq != key:
                self.squashed.setdefault(sq, canon)
        self.fuzzy = fuzzy
        self.expand_compounds = expand_compounds  # "python developer" -> also "python" (off: hurts precision)
        self.keyness = keyness or {}
        self.keyness_outside = keyness_outside
        self.keyness_inside = keyness_inside
        self._fuzzy_pool = sorted(s for s in self.vocab if s.isalpha() and len(s) >= 6 and s not in GENERIC | STRICT)

    # -- text -> tokens ----------------------------------------------------------
    @staticmethod
    def _clean(text: str) -> str:
        text = unicodedata.normalize("NFKC", html.unescape(text))
        text = text.replace("\u00ad", "").replace("\u200b", "")
        text = _SLASH_SPACES.sub("/", text)
        return text[:MAX_CHARS]

    @staticmethod
    def _tokens(segment: str) -> tuple[list[str], list[str]]:
        raw, norm = [], []
        for tok in segment.split():
            t = tok.strip(_TOKEN_EDGE)
            if t.endswith(".") and len(t) > 1 and not t.startswith("."):
                t = t.rstrip(".")  # sentence-final full stop ("... in Python.")
            if t:
                raw.append(t)
                norm.append(t.lower())
        return raw, norm

    def _header(self, line: str) -> tuple[str | None, str]:
        """(kind, remainder). kind is 'skills', 'other' or None; remainder is text after an inline 'Label:'."""
        m = _INLINE_LABEL.match(line)
        if m:
            label = re.sub(r"[^a-z &/]", "", m.group(1).lower()).strip()
            if label in SKILL_HEADERS:
                return "skills", m.group(2)
            if label in OTHER_HEADERS:
                return "other", m.group(2)
            return None, line
        m = _HEADER_LINE.match(line)
        if m:
            label = re.sub(r"[^a-z &/]", "", m.group(1).lower()).strip()
            if label in SKILL_HEADERS:
                return "skills", ""
            if label in OTHER_HEADERS:
                return "other", ""
            words = line.split()
            bare = line.strip().strip(_TOKEN_EDGE + ".").lower()
            if (
                line.isupper()
                and 1 <= len(words) <= 4
                and not any(ch.isdigit() for ch in line)
                and not line.rstrip().endswith(",")  # "HTML," is an item of a list, never a heading
                and bare not in self.index  # "SQL" / "AWS," on its own line is a skill, not a heading
            ):
                return "other", ""  # unknown ALL-CAPS heading ends the skills section
        return None, line

    # -- matching ----------------------------------------------------------------
    def _lookup(self, norm: list[str], i: int) -> tuple[str, int, str] | None:
        """Longest vocabulary phrase starting at token i -> (canonical skill, tokens used, matched key)."""
        for n in range(min(MAX_NGRAM, len(norm) - i), 0, -1):
            key = " ".join(norm[i : i + n])
            hit = self.index.get(key)
            if hit is None and n == 1:
                hit, key = self._single_variants(key)
            if hit is not None:
                return hit, n, key
        for n in (3, 2, 1):  # spacing / hyphen differences: "Power Point", "Share Point", "Bi-lingual"
            if len(norm) - i >= n:
                sq = re.sub(r"[\s\-]", "", "".join(norm[i : i + n]))
                canon = self.squashed.get(sq) or (self.index.get(sq) if n > 1 or sq != norm[i] else None)
                if canon is not None and len(sq) >= 6:
                    return canon, n, canon
        return None

    def _single_variants(self, tok: str) -> tuple[str | None, str]:
        if "/" in tok:
            return None, tok
        if tok.endswith("s") and len(tok) >= 6 and tok[:-1] in self.index:  # "spreadsheets"
            return self.index[tok[:-1]], tok[:-1]
        if len(tok) >= 8 and tok + "s" in self.index:  # "spreadsheet" -> "spreadsheets"
            return self.index[tok + "s"], tok + "s"
        n = normalise_skill(tok)
        if n != tok and n in self.index:
            return self.index[n], n
        return None, tok

    def _grade(self, canon: str, key: str, surface: str, in_skills: bool) -> str | None:
        """'solid' (trust it), 'generic' (needs repeats) or None (drop), for one sighting."""
        if len(surface) <= SHORT_LEN:  # "C", "R", "ML", ...
            ok = surface[0].isupper() and (in_skills or (surface.isupper() and surface in SHORT_OK_ANYWHERE))
            return "solid" if ok else None
        if " " not in key:
            k = self.keyness.get(key, self.keyness.get(canon))
            if k is not None and k < (self.keyness_inside if in_skills else self.keyness_outside):
                if not (in_skills and key in SOFT_SKILLS):
                    return None  # everyday word ("company", "issue", "program"), see KEYNESS_OUTSIDE
        if key in STRICT:
            return "solid" if in_skills else None
        if key in GENERIC:
            return "solid" if in_skills else "generic"
        return "solid"

    def _record(self, found: dict, canon: str, key: str, surface: str, in_skills: bool, ngram: int = 1) -> None:
        grade = self._grade(canon, key, surface, in_skills)
        if grade is None:
            return
        f = found[canon]
        f[grade] += 1
        f["skills"] = f["skills"] or in_skills
        f["multi"] = f["multi"] or ngram > 1 or not canon.isalpha()

    def _scan(self, raw: list[str], norm: list[str], in_skills: bool, found: dict) -> None:
        i = 0
        while i < len(norm):
            hit = self._lookup(norm, i)
            if hit is None:
                if "/" in norm[i]:  # "HTML/CSS" -> html, css
                    for part, rpart in zip(norm[i].split("/"), raw[i].split("/"), strict=False):
                        if part in self.index:
                            self._record(found, self.index[part], part, rpart, in_skills)
                elif in_skills and self.fuzzy:
                    self._fuzzy(norm[i], found)
                i += 1
                continue
            canon, n, key = hit
            surface = raw[i] if n == 1 else " ".join(raw[i : i + n])
            self._record(found, canon, key, surface, in_skills, n)
            if n == 1 and "/" in key and canon == key:  # "html/css" also says html and css
                for part, rpart in zip(key.split("/"), raw[i].split("/"), strict=False):
                    if part in self.index and self.index[part] != canon:
                        self._record(found, self.index[part], part, rpart, in_skills)
            if n > 1 and self.expand_compounds:
                for sub in norm[i : i + n]:
                    c = self.index.get(sub)
                    if c and c != canon and len(sub) > SHORT_LEN:
                        self._record(found, c, sub, sub, in_skills)
            i += n

    def _fuzzy(self, tok: str, found: dict) -> None:
        if not (len(tok) >= 6 and tok.isalpha()) or tok in self.index:
            return
        from rapidfuzz import fuzz, process

        best = process.extractOne(tok, self._fuzzy_pool, scorer=fuzz.ratio, score_cutoff=90)
        if best and abs(len(best[0]) - len(tok)) <= 2 and best[0][0] == tok[0]:
            self._record(found, best[0], best[0], tok, True)
            if best[0] in found:
                found[best[0]]["fuzzy"] = True

    def _keyness_bonus(self, canon: str) -> float:
        """Up to +1.5 for words that behave like real skills in job ads; +0.5 when unknown (phrases, rare words)."""
        k = self.keyness.get(canon)
        return 0.5 if k is None else 1.5 * min(k / 0.3, 1.0)

    # -- public ------------------------------------------------------------------
    def from_text(self, text: str, top_n: int = 30) -> list[ExtractedSkill]:
        found: dict[str, dict] = defaultdict(
            lambda: {"solid": 0, "generic": 0, "skills": False, "multi": False, "fuzzy": False}
        )
        in_skills, lines_left = False, 0
        for line in self._clean(text).splitlines():
            if not line.strip():
                continue
            kind, rest = self._header(line)
            if kind is not None:
                in_skills, lines_left = kind == "skills", SKILLS_SECTION_MAX_LINES
                line = rest
                if not line.strip():
                    continue
            elif in_skills:
                lines_left -= 1
                in_skills = lines_left > 0
            for segment in _SEGMENT_SPLIT.split(line):
                raw, norm = self._tokens(segment)
                self._scan(raw, norm, in_skills, found)

        out = []
        for canon, f in found.items():
            total = f["solid"] + f["generic"]
            if not (f["solid"] or f["generic"] >= GENERIC_MIN_COUNT):
                continue
            score = min(total, 3) + (2.0 if f["skills"] else 0.0) + (0.5 if f["multi"] else 0.0)
            score += self._keyness_bonus(canon)
            score -= 0.5 if f["fuzzy"] else 0.0
            out.append(ExtractedSkill(canon, round(score, 2), total, f["skills"], f["fuzzy"]))
        out.sort(key=lambda s: (-s.score, s.skill))
        return out[:top_n]


# ── module-level convenience ────────────────────────────────────────────────────


@lru_cache(maxsize=2)
def get_extractor(
    vocab_path: str | None = None,
    aliases_path: str | None = None,
    fuzzy: bool = True,
    expand_compounds: bool = False,
    keyness_path: str | None = None,
) -> SkillExtractor:
    vocab = json.loads(Path(vocab_path or MODELS / "skill_vocab.json").read_text(encoding="utf-8"))
    aliases = load_aliases(Path(aliases_path or CONFIGS / "skill_aliases.yaml"))
    kp = Path(keyness_path or MODELS / "skill_keyness.json")
    keyness = (
        json.loads(kp.read_text(encoding="utf-8")) if kp.exists() else None
    )  # optional: rules still work without it
    return SkillExtractor(vocab, aliases, fuzzy, expand_compounds, keyness)


def extract_skills_from_text(text: str, top_n: int = 30, extractor: SkillExtractor | None = None) -> list[str]:
    """Skill names from plain text. Handy for tests and evaluation."""
    return [s.skill for s in (extractor or get_extractor()).from_text(text, top_n)]


def extract_skills(
    data: bytes, filename: str, top_n: int = 30, extractor: SkillExtractor | None = None
) -> ResumeResult:
    """Main entry point: file bytes + filename -> ResumeResult.

    Raises ResumeError for unsupported / oversized / corrupt / encrypted files (the message is safe to show to
    the user). A readable file that simply has no usable text (a scanned PDF) returns an empty result with a
    message instead of raising.
    """
    if not data:
        raise ResumeError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise ResumeError(f"The file is too large (limit {MAX_BYTES // 1_000_000} MB).")
    kind = _kind(data, filename)
    result = ResumeResult()
    text = _pdf_text(data, result.warnings) if kind == "pdf" else _docx_text(data)
    if len(text.strip()) < MIN_TEXT_CHARS:
        result.code = "no_text"
        result.message = (
            "We could not find any text in this file. It looks like a scanned image. "
            "Please upload a PDF or Word file with selectable text, or type your skills in manually."
        )
        return result
    result.skills = (extractor or get_extractor()).from_text(text, top_n)
    if not result.skills:
        result.code = "no_skills"
        result.message = "We could not find skills from our list in this resume. You can add them manually."
    return result
