# Career Guide: AI-based career guidance (MLOps project)

Enter 5 to 10 skills, your experience and education. The app returns the **top-3 specific job roles** (105 roles in 22 families, every sector, not only software), a **readiness score**, the role's **core skills marked have / missing**, **why** the role matched, and an **ordered learning path** linked to free NPTEL/SWAYAM courses (Coursera as fallback).

Everything runs on your own machine: no cloud, no Docker, no external service at runtime.

**How to set up and run (Windows / macOS): see [SETUP.md](SETUP.md).** In short: clone, `git checkout Adi6k-build-main`, then `run.bat` (Windows) or `./run.sh` (macOS/Linux). The original project plan is in [docs/PLAN.md](docs/PLAN.md); results log in [journey.md](journey.md).

## What you get
| Tab | Purpose |
|---|---|
| **Find my career** | Search box over the model's own vocabulary (min 5, max 10 skills), top-3 roles with match strength, readiness ring, "why this role", have/missing skills, learning path with course links and the **+pts match** each skill would add, rising/fading trend badges, family-level fallback when confidence is low, feedback, PDF export |
| **Explore roles** | Every role: core skills with demand share, trends, courses, common titles, typical experience |
| **Insights** | Model quality vs baselines, accuracy by number of skills, independent checks, 2019→2025 drift, live usage and drift monitoring, feedback-driven retrain signal, model versions |

## Objectives (from the project plan) and how each is met
| # | Objective | Where | Result |
|---|---|---|---|
| O1 | Recommend specific roles: ≥5 skills + experience → top-3 of ~100 roles under 22 families; target ≥75% top-3 on 5-skill inputs; fall back to family when confidence is low | `src/models/train.py`, `recommend.py` | **105 roles. 76.9% top-3 on 5 random skills per posting** (79.7% when skills are drawn from the search-box vocabulary, 82.2% on full postings). Profile-matching baseline 65.1%, majority class 18.3%. Low-confidence results show the closest field |
| O2 | Skill gap + readiness: ~10 core skills per role (asked by ≥8% of its postings), have/missing, readiness weighted by demand | `src/models/profiles.py` | `models/skill_profiles.json`, versioned by DVC |
| O3 | Ordered learning path (demand × co-occurrence with your skills, boost for skills that grew 2019→2025), each linked to a free NPTEL/SWAYAM course, Coursera fallback | `recommend.gap()`, `lookups.py` | Courses for 1,175 of 6,000 skills; **54% of the top-200 skills** have a match (plan said about half) |
| O4 | Education as a separate prior from two Indian sources | `lookups.py`, `configs/education.yaml`, `configs/family_nco.yaml` | What employers ask (Naukri 2015-17) × what people actually do (PLFS 2023-24 via NCO-2015 codes). All 11 UI degrees covered. E.g. LLB → Legal 13×, MBBS/Pharma → Healthcare 7×, B.Ed → Teaching 7×, B.Com → Accounting 5×. Nudges the ranking, never overrides skills |
| O5 | Reproducible pipeline | `dvc.yaml`, `params.yaml`, `uv.lock`, MLflow | `uv sync` + `python manage.py download` + `python manage.py pipeline` reproduces every metric (about 2 minutes); runs tagged with git commit + data hash |
| O6 | Real drift, retrain, promote | `src/monitoring/`, `promote.py` | A model trained on 2019 only: 67.1% → 61.9% top-3 on 2025 (−5.2 pts), skill drift JS 0.41 (alert). `retrain` promotes a new model only if it is not worse than the serving champion |

### Data used
Naukri 2025 (97,929 postings, main training and test), Naukri 2019 (30,000; label validation and drift reference), Naukri 2022 (32,738; extra training data), Internshala 2025 (8,483 internships; fresher-mode training data), Naukri 2015-17 (education prior), PLFS 2023-24 (education prior), NPTEL/SWAYAM and Coursera catalogues (courses). The test set is always a held-out part of Naukri 2025; extra sources only add to training and are de-duplicated against the test set.

### How we know the numbers are not circular
Role labels come from hand-written title rules, so accuracy against them alone would be partly self-referential. Three checks that do not depend on those rules:
- **Recruiter-chosen roles:** the model (trained on 2025 rule labels) is scored on 17,790 *2019* postings against Naukri's own `Role` field: **85.8% top-3 hit** (acceptable-role map in `configs/naukri_role_map.yaml`).
- **Naukri categories:** rule labels agree with Naukri's `Functional Area` in 86.0% (plausible-family mapping), 70.9% (IT families grouped) and 59.8% (strict one-to-one) of 22,651 postings; with Naukri's `Role` 74.7% (15,567 postings).
- **Hand-labelled gold set:** `scripts/make_gold_sample.py` writes a 300-row sheet; label it and the evaluate stage reports accuracy against your labels automatically.

### Accuracy needs skills
Top-3 accuracy with 3 / 5 / 8 skills: 68.5% / 76.9% / 81.4%. That is why the form requires at least 5.

## Commands (cross-platform: `uv run python manage.py <command>`)
`serve` · `download` · `pipeline` · `retrain` · `compare` · `test` · `simulate`. On macOS/Linux `make <same>` also works.

## Layout
```
configs/    role_rules.yaml · naukri_role_map.yaml · skill_aliases.yaml · education.yaml · family_nco.yaml
src/data/   ingest · label · eda · split        src/features/  build (multi-hot skills, skill dropout)
src/models/ train · evaluate · compare · profiles · lookups · recommend · promote
src/monitoring/ drift · report                  src/api/  FastAPI service (also serves the UI)
app/static/ index.html (single file, no CDN)    models/champion/  the bundle the app serves (model + profiles + lookups + reports)
run.py · run.bat · run.sh · manage.py           scripts/ download.py · make_gold_sample.py · simulate_traffic.py
```
Pipeline (DVC): `ingest → label → eda → split → train → evaluate → profiles → lookups → drift`.

## API (http://127.0.0.1:8000/docs)
`GET /skills?q=` · `POST /recommend` · `GET /roles`, `/roles/{role}` · `POST /feedback` · `GET /drift`, `/stats`, `/model-info`, `/meta`, `/health`

## Where feedback goes
"Was this helpful?" is saved locally in `data/logs/feedback.jsonl`, joined to the prediction it rated. Insights shows the helpful rate and the least helpful roles, and recommends a retrain when helpfulness drops below 60% (≥20 ratings) or live skill drift reaches alert.

## Deliberate deviations from the plan, and limitations
- **Vocabulary is 6,000 skills, not 1,500.** The plan said "start at 1,500" and tune. With 105 roles, 6,000 lifts top-3 on 5 skills from 65% to 77% (1,500 skills covers 87% of postings with ≥3 known skills, 6,000 covers 97%). The search box still never shows a full list.
- **Class balancing is off.** `class_weight=balanced` cost about 6 points of top-3; macro-F1 still rose without it.
- **Extra training data (2022, Internshala) is about neutral on the 2025 test.** It is included for coverage of fresher roles and older skills, not because it raised the score.
- **Runtime tooling is local only** (no Docker, Prometheus or Grafana); monitoring is built into the app instead. Quantisation, Optuna tuning, SHAP and CodeCarbon from the plan's later stages are not done; "why this role" uses the linear model's coefficients instead of SHAP.
- The data are job **postings**, not people: a match means your skills resemble what postings for that role ask for, not a personal probability.
- Role labels are rule-based; neighbouring roles (Marketing vs Digital Marketing, generic Software Engineer) are confused most (`reports/per_role.csv`).
- Datasets are CC BY-NC-SA / CC0: fine for coursework, not commercial use.
