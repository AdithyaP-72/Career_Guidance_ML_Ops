# Resume -> skills (`src/serving/resume.py`)

Upload a PDF or DOCX, get back the person's skills **in the role model's own spelling** (an entry of
`models/skill_vocab.json`), best first. The list pre-fills the skill box; the user edits it, then hits Analyze.

```python
from src.serving.resume import extract_skills, ResumeError

result = extract_skills(file_bytes, filename, top_n=30)  # -> ResumeResult
result.names  # ['python', 'react.js', 'machine learning', ...]  best first
result.skills  # same, as ExtractedSkill(skill, score, count, in_skills_section, fuzzy)
result.message  # None, or a plain-words reason when nothing usable was found (safe to show the user)
result.code  # 'ok' | 'no_text' (scanned PDF) | 'no_skills'
result.warnings  # e.g. 'Only the first 8 pages were read.'
# raises ResumeError (message is safe to show) for: wrong type / .doc, empty, >5 MB, corrupt, password-protected
```
Suggested API mapping: `ResumeError` -> HTTP 422 (413 for "too large"); `result.code == "no_text"` -> 200 with an empty
list and `result.message` so the UI can say "this looks like a scan, type your skills instead".
Keep the 5-10 you pre-fill from `result.names[:10]`; the order is meaningful.

`get_extractor()` builds the index once (cached); `extract_skills_from_text(text)` skips file parsing.

## How it works
1. Text out of the file **in memory** (`pypdf`, `python-docx` incl. tables). Scanned PDFs have no text: empty result + message.
   Google Docs PDFs come out of pypdf's default mode one word per line; when the text averages < 1.5 words per line it
   is re-read in pypdf's layout mode, which rebuilds the lines from word positions.
2. Find the "Skills" / "Technical Skills" / "Highlights" / "Qualifications" section (also inline `Skills: a, b`); skills
   there rank higher and short or generic skills are trusted there. Headings are matched after repairing split first
   letters from small caps ("T echnical Skills", "S KILLS"); any short heading naming skills counts ("Courses and
   Skills"), as do combined ones ("Frameworks/Tools") and a heading cell followed by a list without a colon. A labelled
   list such as `Tech Stack: a, b` under Projects is a skills list for that line only (plus a wrapped next line), not a
   whole section. A one-word line such as `HTML,` is a list item, never a heading.
3. Longest-phrase matching of word n-grams against the vocabulary, after the training pipeline's own normaliser and
   `configs/skill_aliases.yaml` (ReactJS -> react.js, ML -> machine learning, Tally ERP 9 -> tally). Phrases never span a
   comma, bracket, bullet or column gap, so two-column layouts do not glue words together. Spacing / hyphen / plural
   variants are matched ("Power Point", "Bi-lingual", "spreadsheet").
4. False-positive control:
   * `C`, `R`, `Go`, `ML`... only with a capital letter and a skills context;
   * **keyness** (`models/skill_keyness.json`, built by `src/serving/keyness.py`): for a single-word skill, how often a job
     ad that mentions the word also lists it as a skill tag. `python` 0.57, `sql` 0.38, `excel` 0.19 versus `company`
     0.009, `issue` 0.002, `customer` 0.001. Everyday words are ignored in running prose and ranked low inside lists,
     with no hand-made stoplist; a short soft-skill list (`leadership`, `interpersonal`...) is exempt inside a Skills section;
   * a small list of generic words (`communication`, `management`, `sales`, `data`, `core`...) needs a skills section or
     repetition; ordinary English words that are vocabulary entries (`case`, `team`, `set`...) need a skills section.
5. Typo tolerance (`rapidfuzz`, only inside a Skills section, >= 90 similarity, same first letter).

No network, no AI API, no files written, nothing from the resume is logged or put in an error message (unit-tested).

## Pipeline pieces added
* `models/skill_keyness.json` via DVC stage `resume_keyness` (`params.yaml: resume.keyness`). The extractor still works
  without it (rules only; precision drops, see below). **Run `dvc push`** so teammates / the Docker image get it.
* Dependencies: `uv sync --group serve` (pypdf, python-docx, rapidfuzz). The image needs `models/skill_vocab.json`,
  `models/skill_keyness.json` and `configs/skill_aliases.yaml`.

## Evaluation (public resumes, hand-labelled)
Data: Kaggle "Resume Dataset" (`snehaanbhawal/resume-dataset`, 2,484 livecareer.com example resumes, no names).
**Licence: CC0 / public domain, as stated on the Kaggle page** (the content itself was scraped from livecareer.com, so
check with whoever approves datasets). No resume text is stored in this repo; `tests/data/resume_eval_gold.json` holds only
IDs and labelled skills. The extractor is run on the real PDFs.

Labelling rule: a vocabulary skill counts if the resume claims it as a skill / tool / technology / domain competency (in a
skills list or used in experience); job titles, employers, degree names and generic nouns in prose do not; skills outside the
6,000-entry vocabulary are ignored. **Labels were made by Claude (an AI assistant) reading each resume, not by independent
humans: spot-check a few before presenting the numbers as ground truth.**

| Set | Resumes / labels | Precision | Recall | F1 | Precision@10 |
|---|---|---|---|---|---|
| Development (thresholds tuned here) | 25 / 483 | 49.3% | 95.2% | 65.0% | 73.6% |
| **Holdout (labelled after freezing the design)** | **12 / 255** | **49.1%** | **90.2%** | **63.6%** | **75.0%** |

Ablations on the holdout (F1, with the precision in brackets): plain longest-phrase vocabulary match **38.5%** (24%);
section awareness + hand-written gating rules but no keyness **41.5%** (27%); the full extractor with keyness **63.6%** (49%).
Without Skills-section awareness F1 is similar (65.4%) but recall falls to 71% and precision@10 to 68%, so sections matter
mostly for ranking and recall; compound expansion ("python developer" -> python) lowers F1 to 60.4%; fuzzy matching changes
nothing on these resumes (no typos in them) and is kept because it is cheap.

Precision is moderate by design: the vocabulary has thousands of everyday words that people tagged as skills, and the user
reviews the list. Recall is high so real skills are rarely missing; ranking puts the best first (3 of 4 of the top 10 are
correct).

Resources (process that already has pandas + scikit-learn loaded): index + keyness **+1 MB**; parsing real PDFs peaks at
**+44 MB**; about **80 ms** per resume (the 8-page limit and 5 MB cap bound the worst case).

Reproduce: unzip the dataset, then
`uv run python -m scripts.eval_resume_extractor --data /path/to/unzipped/dataset` (writes `reports/resume_extractor_eval.json`).

## Known limits
* Scanned / image PDFs: no OCR (too heavy for the server); the user gets a clear message.
* Skills the model does not know are not returned (by design); abbreviations that are also words (`C`, `R`) need a skills section.
* Plural/stem variants beyond trailing "s" ("lead generating" vs "lead generation") are not matched.
* Keyness is estimated from Naukri 2022/2025 postings, so it favours skills common in Indian job ads.
* Resumes that bury skills in prose without a Skills section get lower recall; section-less text relies on keyness alone.
