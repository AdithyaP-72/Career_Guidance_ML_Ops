# AI-Based Career Guidance Assistant — MLOps Project

> Working plan for the project: what we're building, which data we use and why, how we tackle Phase 1, and the full MLOps roadmap.

---

## 1. What we're building

**Input:** at least 5 skills you have (languages, tools, frameworks and non-tech skills such as accounting or sales), your education and your years of experience.
**Output:** the top‑3 **specific job roles** (e.g. *Frontend Developer*, *Accountant*, *Staff Nurse*, *Digital Marketing Executive*), each with a confidence score. For the role you pick:
- a **readiness score**;
- the role's ~10 **core skills**, marked have / missing;
- an **ordered learning path**: which skill to learn first, second and third, each linked to a course.

Each role sits under one of 22 broader role families (Software, Sales, Accounting…).

The product covers **all careers, not just developers.** Sales, accounting, HR, teaching, healthcare, design, manufacturing, BPO and software all count. It's a **tabular** problem with no NLP: skills come from a fixed list and the model is a classifier.

### How the user enters skills (no endless checklist)
A checklist of every skill doesn't scale: the job data has 44,000 distinct skill tags. Instead:
- **Skills:** a search box (like LinkedIn's) over a **fixed vocabulary of the ~6,000 most common skills**. The user types `pyth…`, picks from suggestions and must choose **at least 5** (up to ~10). They never see the full list. Because the box can only offer skills the model was trained on, what the app collects always matches what the model expects. The minimum of 5 is there because accuracy falls sharply with fewer skills (section 2.4). With only `html, css`, even a person couldn't tell PHP developer from frontend developer from SEO.
- **Experience:** a number.
- **Education:** a dropdown with ~12 options (12th, Diploma, B.Tech, B.Com, BBA/MBA, B.Sc, BA, CA/CS, MBBS/Pharma/Nursing, LLB, B.Ed…). It feeds a separate education prior, not the skills model; see O4 below.

```
┌─ CareerCompass ─────────────────────────────────────────┐
│  Your skills (min 5):  [ ta▌                       ]    │
│     ↳ tally · tally erp · tableau · tax                 │
│     [accounting ×] [tally ×] [excel ×] [ms office ×]    │
│     [communication skills ×]                            │
│  Experience: [ 1 ] yrs    Education: [ B.Com ▾ ]        │
│  [ Analyze ]                                            │
├─────────────────────────────────────────────────────────┤
│  Top matches                                            │
│   1. Accountant ................ 44%  (Accounting)      │
│   2. Chartered Accountant ......  8%  (Accounting)      │
│   3. Accounts Payable Exec. ....  6%  (Accounting)      │
│                                                         │
│  Accountant: readiness 39%                              │
│   ✔ accounting  ✔ tally                                 │
│   ✘ gst  ✘ tds  ✘ tally erp  ✘ bank reconciliation      │
│                                                         │
│  Your learning path                                     │
│   1. GST    (26% of postings) → NPTEL: "GST: Law & …"   │
│   2. TDS    (25%)             → SWAYAM: "Income Tax …"  │
│   3. Tally ERP (16%)          → …                       │
└─────────────────────────────────────────────────────────┘
   (roles, %, readiness and path order are from our test
    run; course names illustrative)
```

Behind the UI: a Streamlit or React frontend calls a FastAPI service, which runs a Dockerized model pulled from the MLflow registry ("champion"). Predictions are logged to a monitoring stack (Prometheus, Grafana and a drift check). When drift shows up, GitHub Actions runs `dvc repro` to retrain, and the new "challenger" model is compared against the champion.

### Objectives (reframed after checking the data)
We tested these on the actual datasets (section 2.4) before committing to them.

| # | Objective | How we measure it | Realistic target |
|---|---|---|---|
| O1 | **Recommend specific roles.** ≥5 skills + experience → top‑3 of **~100 specific roles** (each under one of 22 families) across all sectors | **Top‑3 accuracy measured on 5-skill inputs** (what users actually enter), plus top‑3 on full postings, macro‑F1 and a per-role report | Baseline LogReg: **67% top‑3 on 5-skill inputs** (75% on full postings; 1 in 12 if guessing the biggest role). Target: ≥ 75% on 5-skill inputs. If the top role's confidence is low, fall back to showing the family and ask for more skills |
| O2 | **Skill gap + readiness.** For each role: its ~10 **core skills** (asked for by ≥ 8% of that role's postings), marked have / missing, plus a readiness % (have-skills weighted by how often postings ask for them) | Per-role skill shares from 2025 postings | Deterministic pandas output, versioned by DVC. Role-level lists are focused (Frontend: javascript 42%, react 36%, css 33%, html 24%), unlike family-level ones (Software: css 21%, java 20%, c# 12%…) |
| O3 | **Ordered learning path.** The next 3–4 missing skills, ranked by *how often the role asks for it × how often it appears alongside skills the user already has*, plus a boost for skills that grew 2019→2025. Each is linked to a free **NPTEL/SWAYAM** course (Coursera as fallback) | Co-occurrence counts from 2025 postings; growth from the Naukri snapshots; skill → course by title/skill-tag matching | Example from our test: Accountant with accounting + tally → **GST → TDS → Tally ERP → bank reconciliation**. About half of the top‑200 skills have an NPTEL/SWAYAM course match |
| O4 | **Education as a separate prior, not a skills-model input** | Two Indian sources: what jobs *ask for* (Naukri 2015–17 has a structured `education` field) and what graduates *actually do* (PLFS, government labour survey) | The 2025/2019 postings have no education column, and regex on descriptions is mostly noise. A separate education → family table instead nudges the ranking and adds notes: e.g. B.Com → Accounts 56% of postings, MBBS → Medical 93%, LLB → Legal 63%, B.Ed → Teaching 75% |
| O5 | **Reproducible pipeline** | `git clone && uv sync && dvc pull && dvc repro` gives the same metrics; every run is in MLflow | Phase 1 deliverable |
| O6 | **Real drift, not simulated** | 2019 snapshot = reference, 2025 = "live" traffic. Measure skill-distribution drift and the drop in model performance, retrain, and promote the challenger | Already visible: a 2019-trained model drops from 75% → 69% top‑3 on 2025 data |

**Out of scope (on purpose):**
- predicting one of 55k exact job titles (we predict ~100 canonical roles and show common real titles for each);
- resume upload (would need NLP);
- markets outside India;
- salary prediction. That one is a possible stretch goal, since 2025 postings have min/max salary, but many are "Not disclosed".

**Framing to keep honest:** the data is job **postings**, not people. The model learns "postings that ask for these skills belong to this family". So the product says *"your skills match what Data & Analytics jobs ask for"*, not *"people like you became data analysts"*.

### Options we considered
- **A, skills profile → role match + skill gap (chosen).**
- **A′, resume upload.** Needs NLP; possible later add-on via keyword matching against our skill vocabulary.
- **B, student stream recommender.** Every Kaggle dataset we found for it is small or synthetic.
- **C, LLM chatbot.** Nothing to train, track or retrain.
- **D, A plus an LLM that explains the output.** A stretch goal for later phases.

---

## 2. Datasets

### 2.1 What we use

Every dataset below was downloaded and checked: first rows, columns, and whether its relationships look real or synthetic. Everything is Indian except O\*NET and Coursera.

**Core: needed for the Phase 1 model and pipeline**

| # | Dataset | Size | Licence | What it gives us |
|---|---|---|---|---|
| 1 | [Indian Job Market Dataset 2025 (Naukri)](https://www.kaggle.com/datasets/shivamshrivastava21/indian-job-market-dataset-2025-2026) | 97,929 postings × 17 cols, 18.6k companies, ~Sept 2025. Ships as `.xlsx` | CC BY‑NC‑SA 4.0 (coursework OK; derived data keeps the licence; no commercial use) | **Main training data ("today").** All sectors. `tagsAndSkills` is a comma-separated skill list (median 8 per posting), so no NLP. No role label, so we create one (2.3) |
| 2 | [Jobs on Naukri.com 2019 (PromptCloud)](https://www.kaggle.com/datasets/promptcloud/jobs-on-naukricom) | 30,000 postings (sample of 485k), Jul–Aug 2019 | CC0 | **Past snapshot + label check.** Pipe-separated `Key Skills` plus Naukri's own labels (`Functional Area` 72, `Role Category` 206, `Role` 649). Validates our title rules and is the drift reference |

**Supporting: Indian data for education, occupations and courses** (pandas lookups, not model training)

| # | Dataset | Size | Licence | What it gives us |
|---|---|---|---|---|
| 3 | [Jobs on Naukri.com 2015–17 (PromptCloudHQ)](https://www.kaggle.com/datasets/PromptCloudHQ/jobs-on-naukricom) | 22,000 postings, Jan 2015 – Jan 2017 | CC BY‑NC‑SA 4.0 | **Education employers ask for, per family.** Structured `education` ("UG: B.Tech/B.E. PG: …", parsed for 89%), and the misnamed `skills` column is actually Naukri's functional area (45 values), i.e. a label. Note: it has no real skill tags |
| 4 | [PLFS unit-level data 2017‑18 → 2023‑24](https://www.kaggle.com/datasets/pradnyakalvikatte/plfs-india-2017-18-to-2023-24) (Govt. of India, MoSPI). Official source: [microdata.gov.in](https://microdata.gov.in/NADA/index.php/catalog) (free registration) | 7 yearly files, ~120 MB each; 2023‑24 alone has 418k people, ~165k with an occupation code | Kaggle mirror says CC0; cite MoSPI, and use the official download for faculty approval | **What Indian graduates actually do.** General education level + technical-education field → 3‑digit NCO‑2015 occupation, with survey weights. The only "people, not postings" source. Columns are coded (`b4q8_perv1` = education level, `b5pt1q6_perv1` = occupation), so decode them with MoSPI's layout document |
| 5 | [NCO‑2015 codes](https://www.kaggle.com/datasets/nemaleshwarh/national-classification-of-organisationnco-2015) + [NCO‑2015 descriptions](https://www.kaggle.com/datasets/shriabhinandansharma/nco-occupation-descriptions-dataset) (Govt. of India, DGE) | 3,445 Indian occupations with codes, titles, descriptions and division/family hierarchy | MIT (Kaggle uploads of government data) | **India's official occupation taxonomy.** Decodes PLFS occupation codes, and gives Indian descriptions for each of our role families via a hand-made family ↔ NCO crosswalk (~22 rows) |
| 6 | [NPTEL‑SWAYAM course catalog](https://www.kaggle.com/datasets/lakshyyaaaa/nptel-swayam-course-catalog) | 21,761 courses (18.3k SWAYAM, 3.5k NPTEL), 667 domains, with URLs | MIT | **"Learn next" → a free Indian course.** 93 of the top‑200 Naukri skills match at least one course title |
| 7 | [Coursera Courses Metadata 2025](https://www.kaggle.com/datasets/longnguyen3774/coursera-courses-metadata-for-analytics-2025) | 5,411 courses with a structured `skills` list | CC BY‑NC‑SA 4.0 | **Fallback course links** when there's no NPTEL/SWAYAM match. 95 of the top‑200 Naukri skills match a Coursera skill tag exactly |
| 8 | [O\*NET 31.0](https://www.onetcenter.org/database.html) (US Dept. of Labor, Aug 2026) | 1,016 occupations; 31.8k technology-skill examples with *Hot Technology* / *In Demand* flags | CC BY 4.0 | **"Hot technology" flags for tech tools only.** Just 24 of the top‑300 Naukri skills appear in it, and it's US-centric |

**Optional extensions (only if time allows)**

| # | Dataset | Size | Licence | Use / caveat |
|---|---|---|---|---|
| 9 | [Internship Opportunities in India 2025 (Internshala)](https://www.kaggle.com/datasets/jayaantanaath/internship-opportunities-in-india-2025) | 8,483 internships, one week of Sept 2025 | MIT | A "fresher / internship" mode: `profile` + clean, standardised `Skills` (793 unique). Only 32% of its skill mentions are in the Naukri top‑500 (more soft skills such as "English proficiency", "Canva"), so it needs its own alias mapping. Its `Education` column is junk |
| 10 | [Latest 30K Jobs Data (Naukri, May 2022)](https://www.kaggle.com/datasets/kuchhbhi/latest-30k-jobs-data) | 32,738 postings with skill lists | CC0 | Extra point on the drift timeline (2019 → 2022 → 2025), but **55% software + 17% data roles**, so compare within families only, or the sector mix itself looks like drift |

### 2.2 What we looked at and rejected

| Dataset | Why not |
|---|---|
| Stack Overflow Developer Survey (previous plan) | Developers only |
| [Indeed India 2021 (PromptCloud, 2 × 30k)](https://www.kaggle.com/datasets/promptcloud/indeed-india-job-data) | Has category labels but **no skills column**, only free-text descriptions (would need NLP) |
| [Naukri Job Listings Jan–Mar 2020 (PromptCloud)](https://www.kaggle.com/datasets/promptcloud/naukri-job-listings-2020) | Same problem: no skills column |
| [Indian Fresher Job Market](https://www.kaggle.com/datasets/shambhurajejagadale/indian-fresher-job-market-who-gets-hired-and-why) | **Synthetic:** ~3.6% per job role and roles independent of branch |
| [Indian Tech Job Market 2026](https://www.kaggle.com/datasets/shree0910/india-tech-job-market-2026-23k-records) | Data/AI roles only (6 categories) |
| [LinkedIn Jobs India](https://www.kaggle.com/datasets/michellemiranda/linkedin-jobs-dataset-for-india) | 949 rows |
| [Internshala Jobs 5k](https://www.kaggle.com/datasets/khushipitroda/internshala-jobs-dataset-with-5000-rows) | No skills column |
| NSDC Qualification Packs / NQR (India's ~3,000 NSQF job roles with standards) | Great content, but only as individual PDFs with no bulk dataset. Could be scraped later for vocational roles |
| National Career Service (NCS) | data.gov.in only has aggregate vacancy statistics, not per-occupation skills |
| [1.3M LinkedIn Jobs & Skills 2024](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) | Real and all sectors, but 6.2 GB and US/UK/CA/AU only. Keep in mind as a stretch "international" extension |
| [LinkedIn Job Postings 2023–24](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | Its "skills" are ~35 LinkedIn job-function codes (e.g. "IT", "Sales"), not real skills, so no skill gap is possible |
| [Naukri Job Listing 2020](https://www.kaggle.com/datasets/promptcloud/naukri-job-listing-dataset-2020) | No skills column, only free-text descriptions (would need NLP) |
| [Naukri Jobs Dec 2023](https://www.kaggle.com/datasets/promptcloud/naukri-jobs-data-dec-2023) | 49 KB, too small |
| [Naukri postings (iqbal303)](https://www.kaggle.com/datasets/iqbal303/job-postings-dataset-from-naukri-com) | Data-science roles only; skills are glued together with no separator; unknown licence |
| [Naukri Jobs (muhammetakkurt)](https://www.kaggle.com/datasets/muhammetakkurt/naukri-jobs-dataset) | Software engineers and data scientists only |
| [India Tech Jobs 2024–26](https://www.kaggle.com/datasets/sridipbasu/india-tech-jobs-2024-2026-salary-and-skills) | Simulated |
| ESCO (EU occupation–skill taxonomy) | Skill names are long phrases ("use spreadsheets software"), so matching them to Naukri tags would need NLP |
| Lightcast Open Skills | API only (registration). Could help normalise skill synonyms later |
| Friend's 4 other files: AMEO engineering grads, Indian student placement, students.csv, AI Job Market | AMEO is real but its target is salary (job title column removed) and it covers engineers only. The other three are **synthetic**: roles evenly split and independent of degree, GPA identical for placed vs. unplaced students, skill flags at ~50% for every job title |

### 2.3 The label problem and how we solve it
The 2025 data has 55k unique free-text titles and no role column. Labels come from **two levels of keyword rules** in `configs/role_rules.yaml`. Each rule maps a title pattern to a **specific role** and its **family**, e.g. `android|kotlin` → *Android Developer* (Software Development), `tally|gst|tds` titles → *Tax / GST Executive* (Accounting & Finance). Rules are ordered from specific to general, and the first match wins.

Why not just use the most common raw titles? We tried. Automatic matching produces useless generic labels: the top "roles" come out as *manager*, *engineer*, *developer*, *executive*. The role list has to be hand-curated.

The rules define **113 candidate roles**. They are *not* automatically the classes. Which roles the model learns is decided by **comparing every snapshot** (2017, 2019, 2022, 2025) in the `taxonomy` pipeline stage, using thresholds in `params.yaml` (results in 2.5):
- **stable**: enough postings in 2025 (to learn) *and* in 2019 (so we can measure drift for that role);
- **emerging**: enough in 2025 only (new or fast-growing roles; the product recommends them, but a 2019-trained model can't know them);
- **excluded**: too rare in 2025 to learn.

Checking each role's skill list catches rule bugs, and the tests in `tests/test_labels.py` keep fixed bugs fixed. Four found so far:
- `unity` matched "Opport**unity**", so generic ads became *Game Developer*;
- `hiring` put BPO ads under *Recruiter*;
- `secretary` caught *Company Secretary* titles and filed them under *Executive Assistant*;
- the building-*Architect* rule caught "Solution Architect" (found by a unit test).

Expect more. Reviewing each role's top skills (notebook section 9) is part of the labelling work.

The 22 families come from grouping 2019's 72 Naukri `Functional Area` values:

> Software Development · QA & Testing · Data & Analytics · IT Infra & Support · ERP / Enterprise Apps · Sales & BD · Customer Service / BPO · HR & Recruitment · Accounting & Finance · Banking & Financial Services · Marketing · Content & Media · Teaching & Training · Design & Creative · Healthcare & Pharma · Manufacturing & Core Engg · Construction & Site Engg · Supply Chain & Logistics · Office Admin & Data Entry · Hospitality · Legal · Strategy & Consulting

**Why this is checkable:** 2019 has *both* titles and Naukri's own labels (`Functional Area`, plus a 649-value `Role` column). So we run our rules on the 2019 titles and measure how often the **family** they assign agrees with Naukri's. That agreement rate is a tracked pipeline metric (`reports/label_quality.json`), alongside coverage and postings per role. Improving the rules is a normal commit: `dvc repro` reruns only the labelling stage and everything after it.

Current rules, as measured by the pipeline:

| Snapshot | Titles given a role | Family agreement with Naukri |
|---|---|---|
| 2017 | 78.9% | 67.3% (16,491 postings compared) |
| 2019 | 79.9% | 67.2% (22,684 postings compared) |
| 2022 | 87.0% | n/a |
| 2025 | 78.5% | n/a |

**Read the agreement number with care.** `reports/labels/disagreements.csv` shows that the biggest "disagreements" are Naukri's taxonomy being coarser than ours. Naukri files DevOps engineers, project managers, business analysts, SAP consultants and testers all under *IT Software – Application Programming*. So the Phase 1 goal is not a fixed percentage. It's to fix the *real* rule errors in that file, and the most common uncaught titles in `reports/labels/unlabelled_titles.csv`.

### 2.4 Feasibility check (done before committing)
| Test | Result |
|---|---|
| Skill vocabulary size vs. coverage | 44k raw tags. With a top‑300 vocabulary, 58% of postings have ≥ 3 known skills; with top‑500, 69% |
| Skills → **family** (2019, 22 families, LogReg, top‑500 skills + experience, 80/20 split) | Top‑3 75.3%, top‑1 51%, macro‑F1 0.42. Majority-class baseline is 33%. Too coarse to be useful: family skill lists are a mix of unrelated skills |
| Skills → **specific role** (2025, 102 roles, ~69k deduplicated postings, LogReg) | With top‑1,500 skills: **top‑3 75%**, top‑5 83%, top‑1 52%, macro‑F1 0.51 (majority baseline 8%). With only top‑500 skills: top‑3 65%. Specific roles need specific skills, so the vocabulary is 1,500 |
| Role model on **realistic sparse input** (test postings cut down to 5 or 3 random skills) | Top‑3: full postings 75% → **5 skills 66%** → 3 skills 52%. Training on randomly cut-down copies of the postings ("skill dropout") lifts these to **67% / 56%** |
| Ordered learning-path demo | Accountant with accounting + tally (44% confidence) → GST → TDS → Tally ERP → bank reconciliation. `html, css` alone → PHP 11% / Frontend 9%, genuinely ambiguous. Add `javascript, react.js` → Frontend 59% |
| 2019 family model applied to 2025 (rule-labelled) | Top‑3 falls to **69.4%**, macro‑F1 0.41. This mixes real drift with label noise from our rules; we'll separate the two in the monitoring phase |
| Real skill drift 2019 → 2025 (% of postings asking) | jQuery 3.6% → 0.6% · PHP 2.4% → 0.2% · Kubernetes 0.01% → 1.5% · Angular 0.05% → 1.0% · Power BI 0.14% → 0.9% · AWS 1.0% → 1.9% · Generative AI 0% → 0.2% |
| Top‑300 skills shared by both years | 169 of 300 |
| Education → family (Naukri 2015–17, share of postings asking for the degree) | B.Com → Accounts 56% · B.Tech → Software 49% · B.A → Sales 22% / Journalism 15% · MBBS → Medical 93% · LLB → Legal 63% · B.Ed → Teaching 75% |
| Skill → course coverage (top‑200 Naukri skills) | NPTEL/SWAYAM title match: 93 · Coursera skill-tag exact match: 95 |

(These were quick experiments before the pipeline existed. The numbers below come from the pipeline itself.)

### 2.5 Which roles become classes: the cross-snapshot comparison
Produced by `uv run dvc repro` → `reports/taxonomy/role_support.csv` (every role's usable postings per snapshot, its share change 2019 → 2025, and what Naukri's own 2019 `Role` label calls those postings). Explore it in notebook section 7.

With the current thresholds (≥ 100 postings in 2025, ≥ 20 in 2019): **113 candidates → 103 classes (100 stable + 3 emerging), 10 excluded.** The classes cover all 22 families and 77.8% of usable 2025 postings. The 100 stable classes cover 78.9% of usable 2019 postings, giving a **20,183-posting 2019 reference set** for the drift experiment.

| Finding | Detail |
|---|---|
| Emerging (enough in 2025, too rare in 2019) | ServiceNow Developer (14 → 303 postings), Trust & Safety / Content Moderator (1 → 251), MEP / HVAC Engineer (18 → 127) |
| Excluded (< 100 in 2025) | Quantity Surveyor (99), Node.js Developer (97), **iOS Developer (95, steadily falling: 165 → 143 → 117 → 95 from 2017 to 2025)**, Data Entry Operator, Medical Representative, Instrumentation Engineer, Lab Technician, Pharmacist, Physiotherapist, Game Developer |
| Role-level drift 2019 → 2025 (change in share of postings) | ML / AI Engineer **×6.0** (59 → 1,186 postings), Data Engineer ×2.5, Accounts Payable/Receivable ×2.3, SAP Consultant ×2.2 · .NET Developer **×0.32**, Java Developer ×0.52, Content Writer ×0.52, Graphic Designer ×0.54 |
| 2022 snapshot is IT-only in practice | e.g. Content Writer 15, Graphic Designer 7, Site Engineer 3 postings (vs. 365 / 423 / 469 in 2025). Use 2022 for tech roles only |
| 2017 behaves like 2019 | Similar counts per role, so it's a second "past" point and a check that 2019 isn't a fluke |
| Weakest roles by Naukri-label agreement (review these rules first) | R&D / Process Engineer (Naukri's top label is "Other", 7%), Video Editor / Animator (10%), Data Analyst (12%; the catch-all `analyst` rule is too broad) |
| Thin 2019 baseline (< 60 postings) | ML / AI Engineer, Staff Nurse, Chef / Cook, Interior Designer, Payroll Executive and 10 more. Their drift numbers will be noisy |
| Class balance (train split) | Largest: Software Engineer (general), 4,669 postings (8.5%). Smallest: 84 |

**Decision so far:** keep these thresholds for the first baselines (103 classes). The team reviews notebook sections 6–9 and records any change in its "Decisions" cell. Changing a threshold is one line in `params.yaml` plus `uv run dvc repro`. Or try one without editing anything: `uv run dvc exp run -S taxonomy.min_reference_postings=50`.

**Starter rule work found by the reports:**
- *application lead* is the most common uncaught 2025 title (1,145 postings, an Accenture template), followed by *application support engineer*, *technical lead – l1*, *application designer*, *security architect* and *solution architect*. Decide which roles these belong to;
- `net` (108 tags) is probably `.net` and should go in `skill_aliases.yaml`.

---

## 3. Phase 1: Development & Reproducibility

> Dataset acquisition/approval; data exploration; feature engineering; baseline models; Git repository; DVC dataset versioning; MLflow experiment tracking; initial model evaluation.

**What "done" means for Phase 1:** a teammate can run

```bash
git clone <repo> && uv sync && dvc pull && dvc repro
```

and get the **same metrics**, with every run visible in MLflow. Every step below works toward that.

### Status
| Steps | State |
|---|---|
| 0–4 (setup, repo, data + DVC, EDA, features) | ✅ Done and pushed (checklist A–H) |
| 5–8 (baselines, MLflow tracking, initial evaluation) | ✅ **Code written and tested end-to-end** (43 unit tests; full `dvc repro` ~23 min, mostly LightGBM). **Your run still to do:** checklist I |

**That completes Phase 1** as the course defines it: dataset acquisition/approval, data exploration, feature engineering, baseline models, Git repository, DVC dataset versioning, MLflow experiment tracking, initial model evaluation. Everything else in section 4 belongs to later phases.

### Your setup checklist (do this yourselves, per the seniors)
Every command below was run on a clean machine with the versions uv resolves today. Run them from the repo root.

**A. Install the tools (each teammate, once)**
1. **uv**:
   - Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`, then reopen PowerShell. Full Windows walkthrough in **H (Windows)** below.

   Check with `uv --version`. uv downloads Python 3.12 by itself; you don't install Python separately.
2. **Git**, plus a **GitHub** account.
3. A **DagsHub** account (sign up with GitHub). It gives us free DVC storage now and a hosted MLflow server in Step 7.

**B. Create the Python environment (one person, once; others just run `uv sync`)**
```bash
uv init --bare --python 3.12 --name career-guidance --pin-python   # only writes pyproject.toml + .python-version
uv add pandas pyarrow openpyxl scikit-learn pyyaml matplotlib fg-data-profiling "dvc[s3]"
uv add --dev pytest ruff jupyter
uv run pytest -q                                                    # 28 passed at this point (43 after step I)
git add pyproject.toml uv.lock .python-version && git commit -m "uv project + lockfile"
```
- Plain `uv init` (without `--bare`) would create a `src/career_guidance/` package layout that clashes with our `src/`.
- `lightgbm` and `mlflow` get added in Step 5. I've checked that they resolve together with everything above.

**C. Download the raw data (one person, once)**
```bash
uv run python -m src.data.download --list   # what will be fetched, with licences
uv run python -m src.data.download          # ~260 MB, ~1–2 min, into data/raw/<name>/
```

**D. Version the data with DVC**
```bash
uv run dvc init
uv run dvc add data/raw/naukri_2025 data/raw/naukri_2019 data/raw/naukri_2017 data/raw/naukri_2022 data/raw/nco_2015 data/raw/courses data/raw/onet_31_0
git add .dvc .dvcignore data/raw/*.dvc data/raw/.gitignore
git commit -m "Track raw datasets with DVC"
```
**PLFS is deliberately left out.** It's MoSPI survey microdata, and our DagsHub repo is public. Pushing it there would mean redistributing it, and MoSPI's data-use terms may not allow that. It stays on your machine only (`data/raw/plfs/` is in `.gitignore`), and nothing before the education-prior stage uses it. Once the terms are checked, track it with `dvc add data/raw/plfs` after removing that `.gitignore` line. Publishing aggregated tables derived from it is the safer route either way.

**E. Connect DagsHub as the storage remote**
1. On DagsHub, create a **public** repository **connected to the GitHub repo**. Code stays on GitHub; DagsHub adds storage and MLflow. Add the three teammates as collaborators with **write** access.
   - Why public: the free plan allows only 2 collaborators and 100 tracked experiments on *private* repos. Public repos have no limit on either.
   - What that means: anyone can see what you push (data, metrics, models), and a leaked token gives write access. So never commit a token, and revoke one immediately if it slips.
2. Get a token: DagsHub → your avatar → **User Settings → Tokens**.
3. In the DagsHub repo, click **Remote → Data → DVC**. It shows the exact commands. They look like this:
   ```bash
   uv run dvc remote add -d origin s3://dvc
   uv run dvc remote modify origin endpointurl https://dagshub.com/<user>/<repo>.s3
   uv run dvc remote modify origin --local access_key_id <your-token>
   uv run dvc remote modify origin --local secret_access_key <your-token>
   ```
   **`--local` matters:** it puts the token in `.dvc/config.local`, which DVC keeps out of Git. The first two lines go into `.dvc/config`, which *is* committed.
4. Commit and push:
   ```bash
   git add .dvc/config && git commit -m "DagsHub DVC remote"
   uv run dvc push
   ```

**F. Run the pipeline**
```bash
uv run dvc repro          # ingest → profiling → label → taxonomy → split → profiles (~2 min)
uv run dvc metrics show   # the numbers in sections 2.3 and 2.5
git status                # dvc.lock, reports/*.json, reports/labels, reports/taxonomy, new .gitignore files
git add -A && git commit -m "Run Phase 1 data pipeline"
uv run dvc push && git push
```

**G. Explore**
- Open `notebooks/01_eda.ipynb` in VS Code (kernel: `.venv`) or with `uv run jupyter lab`.
- The profiling reports are in `reports/profiling/naukri_<year>.html`. Open them in a browser.

**H. Teammates: reproduce everything on your machine**

> **What's on GitHub right now:** Steps 0–4 (data pipeline, EDA, features) plus this README.
> **Coming next:** the Steps 5–8 model code (baselines, MLflow, `journey.md`). Wherever a step below says *after the model push*, wait until the repo owner has pushed it. The commands stay the same.

*H1. Accounts and access (once)*
1. Create a **GitHub** account and a **DagsHub** account (sign up on DagsHub *with* GitHub; it's easiest).
2. Send the repo owner your GitHub and DagsHub usernames. They invite you **twice**:
   - GitHub repo → Settings → Collaborators, so you can push code;
   - DagsHub repo → Settings → Collaborators, with **write** access, so you can push data and log MLflow runs.

   DagsHub requires a login even for reading, so you need this before `dvc pull` works.
3. Accept both invites (check your email or notifications).
4. Create your own token: DagsHub → avatar → **User Settings → Tokens → New token**. Keep it private: **never paste it into chat, screenshots or commits**. If it leaks, revoke it there and make a new one.

*H2. Install the tools (once)*

| | Linux / macOS (terminal) | Windows (PowerShell, no admin needed) |
|---|---|---|
| Git | usually preinstalled (`git --version`) | `winget install --id Git.Git -e --source winget` |
| uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| Check | reopen the terminal, then `git --version` and `uv --version` | **close and reopen PowerShell**, then `git --version` and `uv --version` |

You don't install Python yourself: uv downloads Python 3.12 for the project. On Windows, keep the project **out of OneDrive-synced folders** (e.g. use `C:\dev`), because syncing `.venv` and `.dvc/cache` is slow and can lock files.

*H3. Get the project and its environment*
```bash
git clone https://github.com/AdithyaP-72/Career_Guidance_ML_Ops.git
cd Career_Guidance_ML_Ops
uv sync                      # creates .venv with the exact versions in uv.lock
```
(Same three lines in PowerShell. Windows PowerShell 5.1 doesn't understand `&&`, so keep them on separate lines.)

*H4. Connect DVC to DagsHub with your token*

Linux / macOS / Git Bash:
```bash
read -rs DAGSHUB_TOKEN       # type exactly this, Enter, then paste your token, Enter (nothing shows: normal)
uv run dvc remote modify origin --local access_key_id "$DAGSHUB_TOKEN"
uv run dvc remote modify origin --local secret_access_key "$DAGSHUB_TOKEN"
unset DAGSHUB_TOKEN
```
Windows PowerShell:
```powershell
$sec = Read-Host "Paste your DagsHub token" -AsSecureString
$tok = [System.Net.NetworkCredential]::new("", $sec).Password
uv run dvc remote modify origin --local access_key_id $tok
uv run dvc remote modify origin --local secret_access_key $tok
Remove-Variable sec, tok
```
`--local` saves the token in `.dvc/config.local`, which Git never sees. Don't use `read -rs <your-token>`: the token goes on the *next* line, not after the command.

*H5. Download the data and check you can reproduce*
```bash
uv run dvc pull              # raw data + every pipeline output (and, after the model push, the trained models)
uv run pytest -q             # now: 28 passed · after the model push: 43 passed
uv run dvc repro             # must print: "Data and pipelines are up to date."
```
**That last line is the Phase 1 reproducibility check.** It means your code, data and outputs match `dvc.lock` exactly, with nothing rerun. If instead it starts rerunning stages:
- **On Windows:** you probably cloned before `.gitattributes` existed; delete the folder and clone again.
- **Otherwise:** check `git status` for local edits.

*H6. MLflow (after the model push)*

You only need this when you train or retrain models yourself:
```bash
cp .env.example .env         # PowerShell: Copy-Item .env.example .env
```
Edit `.env`: put your DagsHub username in `MLFLOW_TRACKING_USERNAME` and your token in `MLFLOW_TRACKING_PASSWORD`. `.env` is gitignored, so never commit it. To *view* runs you need nothing locally: open the repo on DagsHub → **Experiments** (or `https://dagshub.com/AdithyaP-72/Career_Guidance_ML_Ops.mlflow`). You'll see the experiment `career-role-baselines` with one run per baseline.

*H7. Look around*
- **EDA notebook:** open `notebooks/01_eda.ipynb` in VS Code with the **Python** and **Jupyter** extensions, choose the `.venv` kernel (Linux/macOS `.venv/bin/python`, Windows `.venv\Scripts\python.exe`) and click **Run All**. Or run `uv run jupyter lab`.
- **Data profiling reports:** open `reports/profiling/naukri_<year>.html` in a browser.
- **Metrics:** `uv run dvc metrics show`.
- **After the model push:** `reports/model_comparison.csv`, `reports/eval/<model>/` and [journey.md](journey.md).

*H8. Everyday workflow*
- **Get teammates' latest work:** `git pull`, then `uv sync` and `uv run dvc pull`.
- **Change something** (a rule in `configs/role_rules.yaml`, a value in `params.yaml`, code in `src/`):
  1. `uv run pytest -q`, then `uv run dvc repro` (it reruns only what your change affects);
  2. `git add -A && git commit -m "what you changed and why"`;
  3. `uv run dvc push && git push`. **Both** are needed: Git carries the code and `dvc.lock`, DVC carries the data and models.
- **Before committing:** clear notebook outputs, and run `git status` to make sure no `.env` or token file is listed.
- **PLFS** isn't on DagsHub (step D: possible redistribution limits). You only need it for a later phase: `uv run python -m src.data.download --only plfs`.

**I. Models + MLflow (Steps 5–8)**: once Steps A–H work
1. Add the model packages (one person; everyone else just runs `uv sync` after pulling):
   ```bash
   uv add lightgbm mlflow python-dotenv
   uv run pytest -q                                   # now 43 passed (the model tests need these packages)
   git add pyproject.toml uv.lock && git commit -m "Add model + MLflow packages"
   ```
2. Point MLflow at DagsHub (**each person**, with their own token):
   ```bash
   cp .env.example .env      # Windows PowerShell: Copy-Item .env.example .env
   ```
   Open `.env` and set your DagsHub username and token. `.env` is gitignored, so it never gets committed. Without a `.env`, runs go to a local `mlflow.db` instead (view with `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`).
3. Train, evaluate and compare the baselines:
   ```bash
   uv run dvc repro            # data stages rerun once (ingest changed), then train@<model> ×5 and select: ~23 min
   uv run dvc metrics show     # every model's metrics side by side
   cat reports/model_comparison.csv
   ```
   You should get the numbers in Step 5 below (same seed and data; LightGBM may differ in the last decimal).
4. See it in MLflow: on DagsHub, open the repo → **Experiments** tab (or `https://dagshub.com/AdithyaP-72/Career_Guidance_ML_Ops.mlflow`). The experiment `career-role-baselines` has 5 runs:
   - each run has its params, metrics, git commit + data hashes, and its `eval/` charts and CSVs;
   - the best one (tag `best_baseline = true`, logistic regression) also carries the saved model.
   Sort by `top3_5skills` to compare.
5. Save everything:
   ```bash
   git add -A && git commit -m "Phase 1 Steps 5-8: baselines, MLflow tracking, initial evaluation"
   uv run dvc push && git push
   ```

### Step 0: Decisions locked in (seniors' "versions!!!" advice)
- **One package manager: `uv`.** It writes `uv.lock`, which pins every package including sub-dependencies. Never `pip install` into this project.
- **Python 3.12**, pinned in `.python-version`.
- **pandas 2.3, not 3.x.** The profiling library requires `pandas<3` and `numpy<2.4`, and uv picks the newest versions that fit. All our code is tested on these versions.
- **`fg-data-profiling`, not `ydata-profiling`.** We read the docs before installing and hit two problems:
  - ydata-profiling 4.18 is deprecated (its own import warning says to switch to `fg-data-profiling`, from the same team);
  - it crashes on import with current setuptools, because it uses `pkg_resources`, which setuptools 81+ removed.

  `fg-data-profiling` (imported as `data_profiling`) is its maintained continuation and has neither problem. This is exactly the kind of version error the seniors warned about.
- **No neural model in Phase 1.** scikit-learn + LightGBM on sparse skill features. A Keras MLP may come later as a challenger (see section 1).
- **Dataset approval:** get the core and supporting datasets in 2.1 (#1–8) approved by faculty. Mention the CC BY‑NC‑SA licences (Naukri 2025, Naukri 2015–17, Coursera) and that PLFS comes from MoSPI.

### Step 1: Repository structure
```
├── configs/
│   ├── sources.yaml             # every raw dataset: URL, licence, which files to keep
│   ├── skill_aliases.yaml       # "reactjs" / "react js" / "react" → "react.js"
│   ├── role_rules.yaml          # title regex → candidate role + family (first match wins)
│   └── functional_area_map.yaml # Naukri's 2017/2019 labels → our families (to score the rules)
├── data/                        # all DVC-managed, never in Git
│   ├── raw/<source>/            # as downloaded; one .dvc file per folder
│   ├── interim/                 # postings.parquet (all snapshots, tidy), labelled.parquet
│   └── processed/               # classes.json, train / test / reference_2019 .parquet
├── models/                      # skill profiles, baselines/<model>.joblib, model.joblib (best baseline)
├── notebooks/01_eda.ipynb
├── reports/                     # metrics JSON, labels/ taxonomy/ eval/ CSVs + charts, profiling/ HTML
├── src/
│   ├── utils.py                 # paths, config loading, skill + experience parsing
│   ├── data/                    # download, ingest, profile, label, taxonomy, split
│   ├── features/build.py        # the sklearn feature pipeline + skill dropout
│   └── models/                  # profiles, estimators, train, evaluate, tracking (MLflow), select
├── tests/                       # pytest: skills, labelling rules, features, models
├── dvc.yaml  params.yaml        # the pipeline and every tunable value
├── pyproject.toml  uv.lock  .python-version   # created by you in step B
├── ruff.toml  conftest.py  .gitignore  .gitattributes
├── .env.example                 # MLflow → DagsHub settings template (copy to .env)
├── journey.md                   # Phase 1 initial model evaluation
└── README.md
```
Every pipeline script runs as `uv run python -m src.<module>` from the repo root.

### Step 2: Data, Git and DVC (remember: versioning ≠ storage)
- **Getting the data:** `src/data/download.py` reads `configs/sources.yaml` and keeps only the files we need from each zip. For example, it keeps 3 of O\*NET's ~40 files and only the 2023‑24 PLFS year, so storage stays at ~260 MB. Kaggle's public API needs no login today; if that changes, use the `kaggle` CLI with a token.
- **Versioning** = the small `data/raw/*.dvc` files and `dvc.lock`, which hold content hashes. These go in **Git**.
- **Storage** = the DagsHub **DVC remote**, which holds the actual bytes (`dvc push` / `dvc pull`).
- **Pipeline outputs** (`data/interim`, `data/processed`, `models/`, `reports/profiling`) are DVC-tracked too, so `dvc pull` gives teammates the outputs without re-running anything. Small review files (`reports/*.json`, `reports/labels/`, `reports/taxonomy/`) are committed to Git so they show up in diffs and on GitHub.
- **Adding a new snapshot later** (e.g. a 2026 scrape) = add it to `sources.yaml` and a loader in `ingest.py`, then `dvc add`, `dvc repro`, commit. That's our data-versioning story, and the new snapshot becomes "live traffic" for monitoring.

### Step 3: Data exploration
Two layers:
- **Automated profiling** (`profiling` stage): one `fg-data-profiling` report per snapshot in `reports/profiling/`.
- **Project-specific EDA** (`notebooks/01_eda.ipynb`): it reads pipeline outputs, so it runs in seconds. It covers snapshot summary, skills per posting and vocabulary size, unmerged synonyms, skill drift, experience, rule quality, the role comparison (2.5), class balance, and a skill-profile sanity check.

What cleaning does (`ingest` stage), and what we found:

| | 2017 | 2019 | 2022 | 2025 |
|---|---|---|---|---|
| Postings after removing exact duplicates | 21,321 | 29,249 | 26,252 | 89,052 |
| Duplicates removed | 679 | 176 | **6,486 (20%)** | **8,877 (9%)** |
| Median skills per posting | n/a (no tags) | 8 | 8 | 8 |
| Postings with no skills | 100% | 3.6% | 0.02% | 0.6% |
| Distinct skills (after aliases) | n/a | 14,462 | 14,982 | 43,833 |

- **Separators differ per snapshot:** `,` in 2025, `|` in 2019, newlines in 2022. Normalisation also removes the spaces Naukri puts around punctuation ("c + +" → "c++", "ci / cd" → "ci/cd") and maps ~50 synonym groups (`configs/skill_aliases.yaml`, mined from the 4,000 most common tags).
- **Vocabulary size:** with the top 1,500 skills, 87% of 2025 postings keep ≥ 3 known skills (top 500: 69%, top 5,000: 97%). We first chose 1,500, but in model tests **6,000 beat 1,500 by ~5 points of top‑3 accuracy** (79.4% vs 74.6%, logistic regression), because specific roles need specific skills. So `top_k_skills = 6000`. It's still a search box, never a list the user scrolls.
- **Skill drift 2019 → 2025** (% of postings): javascript 8.0 → 3.2, html 6.4 → 1.7, jquery 3.8 → 0.6, mysql 3.3 → 0.7 · sap 1.1 → 5.1, software testing 0.3 → 2.1, continuous integration 0.3 → 1.8, kubernetes 0.01 → 1.6.
- **Experience:** median minimum experience is 2–3 years in every snapshot. The model uses the posting's **minimum** (`exp_min`), since a user enters one number.
- **Train/serve consistency:** only features the form collects (skills, experience). Company, location and salary stay out of the model.

### Step 4: Features (`src/features/build.py`)
- `make_preprocessor(top_k, exp_clip)` is an sklearn `ColumnTransformer` with two parts:
  - **skills:** `CountVectorizer(analyzer=identity, binary=True, max_features=6000)` straight on the skill lists, giving a sparse 0/1 vector;
  - **experience:** median-impute → clip to 0–20 → scale to 0–1.

  It's fit on the train split only, so there's no leakage. The fitted vocabulary *is* the search box's list. Skills outside it are ignored at serving time (tested). `identity` and `clip_experience` are top-level functions so the fitted pipeline can be pickled (tested).
- `skill_dropout(df, copies, min_k, max_k, seed)`: training-time augmentation that adds copies of each posting cut to 2–6 random skills. It's applied in Step 5 to the training split only.
- `sample_skills(...)`: gives each test posting (and each 2019 reference posting) a fixed 5-skill `skills_eval` column in the `split` stage. **The headline metric uses `skills_eval`**, because users type ~5 skills while postings list ~8.
- **Skill profiles** (`profiles` stage, from the *train* split only, since the profile-matching baseline predicts from them):
  - per role, the core skills (≥ 8% of the role's postings, max 12), the top 30 skills with shares, common titles, and experience quartiles (`models/skill_profiles.json`);
  - skill growth 2019 → 2025;
  - a 6,000×6,000 skill co-occurrence matrix (sparse) that orders the learning path (`models/skill_cooccurrence.npz` + `skill_vocab.json`).
- **Splits** (`split` stage): 2025 class postings with ≥ 3 skills → **54,752 train / 13,688 test** (stratified by role, seed 42). Plus **20,183** 2019 postings in the 100 stable classes as the drift reference.

### Step 5: Baseline models (results)
All five share the same feature pipeline (Step 4) and training data: 54,752 postings plus 2 skill-dropout copies each, so 164,256 rows over 103 roles. Each model's settings live in `params.yaml → train.<model>`. The headline metric is **top‑3 accuracy on 5-skill test inputs**: 13,688 held-out 2025 postings, each cut to 5 random skills, which is what a user types.

| Model | What it does | Top‑3 | Top‑1 | Top‑5 | Macro‑F1 | Right family | Top‑3, full postings | Train | Predict 1k | Size |
|---|---|---|---|---|---|---|---|---|---|---|
| Dummy | Ignores skills; always the 3 most common roles | 19.8% | 8.5% | 25.7% | 0.002 | 17.5% | 19.8% | 2 s | 19 ms | 0.07 MB |
| Profile matching | Cosine similarity to each role's average posting (no learning) | 66.3% | 43.5% | 75.5% | 0.455 | 62.9% | 73.9% | 2 s | 35 ms | 0.3 MB |
| **Logistic regression** ✅ | One weight per skill per role | **79.4%** | 56.5% | **85.8%** | 0.553 | **72.1%** | 84.6% | 3 min | **21 ms** | 4.6 MB |
| Random forest | 100 trees, leaf ≥ 5 | 73.8% | 49.3% | 80.7% | 0.440 | 66.8% | 79.3% | 42 s | 149 ms | 107 MB |
| LightGBM | 120 boosting rounds × 103 trees each | 78.2% | **56.8%** | 84.5% | **0.560** | 72.0% | **84.8%** | 15.5 min | 807 ms | 7.8 MB |

What the columns mean:
- **Top‑k:** is the true role among the model's k best guesses? The product shows 3.
- **Macro‑F1:** per-role F1 averaged so small roles count as much as big ones.
- **Right family:** is the #1 role at least in the correct family?
- **Full postings:** the same top‑3 test, but with all of a posting's skills.

Settings that mattered, measured with logistic regression on the 5-skill test (both first spotted in a teammate's exploratory branch, then confirmed on our data):

| Setting | Top‑3 |
|---|---|
| 1,500 skills, `class_weight: balanced` | 68.9% |
| 1,500 skills, no class weights | 74.6% |
| **6,000 skills, no class weights (used)** | **79.4%** |

LightGBM needed small, regularised settings. Library defaults diverge with 103 classes, and 200 larger rounds took over 20 minutes.

### Step 6: The DVC pipeline (`dvc.yaml`)
All of Phase 1:

| Stage | Command | Reads | Writes |
|---|---|---|---|
| `ingest` | `src.data.ingest` | `data/raw/naukri_*`, `skill_aliases.yaml` | `data/interim/postings.parquet`, metric `reports/ingest_summary.json` |
| `profiling` | `src.data.profile` | postings | `reports/profiling/*.html` |
| `label` | `src.data.label` | postings, `role_rules.yaml`, `functional_area_map.yaml` | `data/interim/labelled.parquet`, `reports/labels/*.csv`, metric `reports/label_quality.json` |
| `taxonomy` | `src.data.taxonomy` | labelled; params `taxonomy`, `data.min_skills` | `data/processed/classes.json`, `reports/taxonomy/role_support.csv`, metric `reports/taxonomy_summary.json` |
| `split` | `src.data.split` | labelled, classes; params `split`, `features.eval_n_skills` | `train` / `test` / `reference_2019.parquet`, metric `reports/split_summary.json` |
| `profiles` | `src.models.profiles` | train, labelled, classes; params `profiles`, `features.top_k_skills` | `models/skill_profiles.json`, `skill_cooccurrence.npz`, `skill_vocab.json` |
| `train@<model>` ×5 | `src.models.train <model>` | train, test, reference_2019; params `features`, `train.<model>` | `models/baselines/<model>.joblib`, `reports/eval/<model>/`, `reports/runs/<model>.json`, metric `reports/metrics/<model>.json`, + an MLflow run |
| `select` | `src.models.select` | every model's metrics + run | `models/model.joblib` (best), `reports/model_comparison.csv`, metric `reports/metrics.json`, + the best model uploaded to its MLflow run |

`dvc repro` reruns only stages whose inputs changed:
- editing `role_rules.yaml` reruns label → taxonomy → split → profiles, but not ingest or profiling;
- changing `taxonomy.min_train_postings` reruns taxonomy onward;
- changing one model's block, e.g. `train.lightgbm`, retrains only that model (then `select`).

Useful commands:
- `uv run dvc dag` draws the graph;
- `uv run dvc params diff` shows what changed;
- `uv run dvc exp run -S taxonomy.min_reference_postings=50` followed by `uv run dvc exp show` gives the experiment table the seniors showed on slide 14.

### Step 7: MLflow experiment tracking
`src/models/tracking.py` points MLflow at the server in `.env`, which is our DagsHub MLflow, or falls back to a local `mlflow.db`. Each `train@<model>` stage logs one run to the `career-role-baselines` experiment:

| Logged | What |
|---|---|
| Tags | `git_commit`, `code_dirty` (uncommitted code/config edits), `data.train_md5` / `data.test_md5` / `data.reference_2019_md5` (the same hashes DVC uses), `model`, `dvc_stage` |
| Params | the model's block, the `features` block, the taxonomy thresholds, training rows, number of classes |
| Metrics | everything in Step 5 plus the 2019-reference scores, fit time, model size, prediction latency |
| Artifacts | `eval/`: per-role report, top confusions, family confusion matrix (CSV + chart) |
| Model | **best baseline only**, uploaded by `select`: the full pipeline in skops format with our `src/` code, so it loads anywhere |

How each row helps:
- **Tags make a run traceable:** from any run you can get back to the exact code (git) and data (DVC hashes) that produced it.
- **The model is uploaded for the best baseline only** because the random forest alone is ~375 MB in MLflow's format, and all five model files are versioned in DVC anyway.
- **skops** is MLflow's safer replacement for pickle. It only loads types listed in `TRUSTED_TYPES` (`src/models/train.py`), and a test checks that every model reloads with that list.

Registering the model in the MLflow Model Registry (champion/challenger aliases) is a later-phase step.

### Step 8: Initial evaluation
Written up in **[journey.md](journey.md)**: results, what moved the numbers, where the best model goes wrong, and limitations. In short:
- **Logistic regression is the best baseline:** 79.4% top‑3 on 5-skill input, **+13.1 points over profile matching**, 4× the dummy baseline. It's also the fastest to predict (21 ms per 1,000 users).
- **LightGBM is close** on top‑1 and macro‑F1 but loses on top‑3, takes 5× longer to train and is 40× slower to predict. Random forest is worse and 23× bigger.
- **Most errors are between neighbouring roles:** Customer Service ↔ Voice Support, BDE ↔ Sales Executive, specific developer roles → Software Engineer (general). At family level, Banking postings often land in Sales (26%), and Strategy & Consulting is the hardest family (48% correct).
- **More skills help:** 84.6% top‑3 with full postings vs 79.4% with 5 skills, which supports the "at least 5 skills" rule in the form.

### Suggested team split
1. **Data:** `sources.yaml` / download, `ingest.py`, `skill_aliases.yaml` (start with notebook section 3), EDA notebook
2. **Labels:** `role_rules.yaml`, starting from the starter list in 2.5 and the two `reports/labels/*.csv` files; review notebook sections 6, 7 and 9; propose the taxonomy thresholds
3. **Models (Step 5):** baselines including profile matching, the `train` / `evaluate` stages, MLflow logging
4. **Infra + lookups:** uv / DVC / DagsHub setup (each person still does their own), the DagsHub MLflow server, keeping `dvc.yaml` tidy, and later the `lookups` stage (education prior, course links)

The infrastructure person shouldn't do everyone's setup. The seniors said to set up the tools yourselves, so each person should do their own `uv sync` and `dvc pull` at least once.

---

## 4. Roadmap summary

| # | MLOps stage | Tools for this project | Status |
|---|---|---|---|
| 1 | Problem definition & data collection | Naukri 2025 + 2019 (core), 2017 + 2022 (comparison); PLFS, NCO‑2015, NPTEL/SWAYAM, Coursera, O\*NET (supporting) | ✅ Built: `download.py` + `sources.yaml` |
| 2 | Data cleaning & preprocessing | pandas, fg-data-profiling, rule-based labelling, cross-snapshot taxonomy, scikit-learn feature pipeline | ✅ Built: `ingest` / `profiling` / `label` / `taxonomy` / `split` / `profiles` stages |
| 3 | Data versioning & storage | Git, DVC with DagsHub remote | 🟡 Code ready; your setup (checklist E) |
| 4a | Model development: baselines & tracking | scikit-learn, LightGBM, MLflow (DagsHub) | ✅ Built: 5 baselines, MLflow runs, logistic regression best at 79.4% top‑3 |
| 4b | Model development: tuning (and optional neural model) | Optuna, optional Keras MLP | ⏳ Later |
| 5 | Validation & testing | pytest (43 tests: skills, rules, features, models), MLflow registry (champion/challenger), SHAP, CodeCarbon | 🟡 Tests built; rest later |
| 6 | Packaging & CI/CD | Docker or Podman, TF Serving (if neural), **quantization** (ONNX for sklearn/LightGBM, TFLite if neural), GitHub Actions | ⏳ Later |
| 7 | Deployment | FastAPI with Render or Hugging Face Spaces (or SageMaker) | ⏳ Later |
| 8 | Monitoring | Prometheus + Grafana **run as Docker images**. For drift, **not Evidently** (per seniors): use NannyML, Alibi Detect or custom chi-square/JS-distance tests on skill frequencies. 2019 = reference (`reference_2019.parquet` is already built), 2025 = live; later a fresh scrape | ⏳ Later |
| 9 | Continuous training & feedback | GitHub Actions → `dvc repro` on a new snapshot → challenger vs. champion, plus user feedback ("was this helpful?") | ⏳ Later |

**What Phase 1 covers:** stages 1–4a and the start of 5. That means approved and explored data, versioned with DVC, run through a reproducible pipeline, with tracked baseline experiments in MLflow and an initial evaluation. Everything after that (tuning, packaging, serving, monitoring, retraining) builds on this foundation.

---

## Seniors' advice (notes)
- Do the setup of MLOps tools on your own.
- Data versioning is different from data storage.
- Read each tool's documentation before installing to avoid version errors.
- Use only one package manager for reproducibility.
- Docker alternatives: TensorFlow Serving, Podman, TensorRT.
- Try quantization.
- Don't use Evidently AI.
- Prometheus and Grafana: use Docker images for everything instead of .exe installers.
