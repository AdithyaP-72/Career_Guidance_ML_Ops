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
- **Skills:** a search box (like LinkedIn's) over a **fixed vocabulary of the ~1,500 most common skills**. The user types `pyth…`, picks from suggestions and must choose **at least 5** (up to ~10). They never see the full list. Because the box can only offer skills the model was trained on, what the app collects always matches what the model expects. The minimum of 5 is there because accuracy falls sharply with fewer skills (section 2.4). With only `html, css`, even a person couldn't tell PHP developer from frontend developer from SEO.
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

First attempt at roles (~113 rules, ~2 hours):
- 80% of 2025 postings get a specific role;
- **102 roles have ≥ 100 postings** (70 have ≥ 300). Those 102 are the classes.

Checking the resulting skill lists catches rule bugs. Three we found and fixed:
- `unity` matched "Opport**unity**", so generic ads became *Game Developer*;
- `hiring` put BPO ads under *Recruiter*;
- `secretary` caught *Company Secretary* titles and filed them under *Executive Assistant*.

Expect more bugs like these. Reviewing each role's top skills is part of the labelling work.

The 22 families come from grouping 2019's 72 Naukri `Functional Area` values:

> Software Development · QA & Testing · Data & Analytics · IT Infra & Support · ERP / Enterprise Apps · Sales & BD · Customer Service / BPO · HR & Recruitment · Accounting & Finance · Banking & Financial Services · Marketing · Content & Media · Teaching & Training · Design & Creative · Healthcare & Pharma · Manufacturing & Core Engg · Construction & Site Engg · Supply Chain & Logistics · Office Admin & Data Entry · Hospitality · Legal · Strategy & Consulting

**Why this is checkable:** 2019 has *both* titles and Naukri's own labels (`Functional Area`, plus a 649-value `Role` column). So we run our rules on the 2019 titles and measure how often the **family** they assign agrees with Naukri's. That agreement rate is a tracked pipeline metric (`reports/label_quality.json`), alongside coverage and postings per role. Improving the rules is a normal commit: `dvc repro` reruns only the labelling stage and everything after it.

First attempt at family-level agreement (22 rules): 78% coverage on 2019 titles, **68% agreement** with Naukri's labels. Phase 1 goal: ≥ 80%. Some disagreement is Naukri's own noise, since recruiters pick the functional area themselves.

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

---

## 3. Phase 1: Development & Reproducibility

> Dataset acquisition/approval; data exploration; feature engineering; baseline models; Git repository; DVC dataset versioning; MLflow experiment tracking; initial model evaluation.

**What "done" means for Phase 1:** a teammate can run

```bash
git clone <repo> && uv sync && dvc pull && dvc repro
```

and get the **same metrics**, with every run visible in MLflow. Every step below works toward that.

### Step 0: Decisions to lock in on day 1 (seniors' "versions!!!" advice)
- **One package manager: `uv`.** It's fast, it writes a lockfile (`uv.lock`) and it pins the Python version. Never mix in `pip install` or conda. Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **Python 3.12.**
- **No neural model in Phase 1.** On sparse skill features, scikit-learn and LightGBM are the right tools. If a Keras MLP is added later (for the TF Serving / TFLite quantization path the seniors used), read TensorFlow's NumPy compatibility notes before adding it.
- **Dataset approval:** get the core and supporting datasets in 2.1 (#1–8) approved by faculty. Mention the CC BY‑NC‑SA licences (Naukri 2025, Naukri 2015–17, Coursera) and that PLFS comes from MoSPI.

### Step 1: Repository structure (cookiecutter-data-science style)
```
career-guidance/
├── configs/
│   ├── role_rules.yaml      # title regex → specific role + family (versioned, reviewed like code)
│   ├── skill_aliases.yaml   # "reactjs" / "react js" / "react.js" → "react"
│   └── family_nco.yaml      # role family ↔ NCO-2015 codes ↔ Naukri functional areas
├── data/
│   ├── raw/                 # DVC-tracked, never edited by hand
│   ├── interim/             # tidy parquet, skills as normalised lists
│   └── processed/           # labelled + train/test splits
├── notebooks/               # 01_eda.ipynb, 02_baselines.ipynb
├── scripts/download.sh      # fetches raw data (Kaggle public API + O*NET)
├── src/
│   ├── data/{ingest.py, label.py, split.py}
│   ├── features/build.py    # sklearn ColumnTransformer
│   └── models/{train.py, evaluate.py, profiles.py}
├── models/                  # DVC outputs: model.pkl, skill_profiles.json
├── reports/                 # metrics.json, label_quality.json, plots, profiling HTML
├── params.yaml              # every tunable value lives here
├── dvc.yaml                 # the pipeline
├── pyproject.toml + uv.lock
├── .gitignore  .dvcignore
└── README.md / journey.md
```

```bash
uv init --python 3.12
uv add pandas pyarrow openpyxl scikit-learn lightgbm mlflow dvc ydata-profiling pyyaml
uv add --dev ruff jupyter pytest
```
`openpyxl` is needed because the 2025 data comes as `.xlsx`. "Pandas Profiler" is now called **`ydata-profiling`**. It has lagged behind new NumPy releases before, so check its supported versions before adding it.

### Step 2: Getting the data, Git and DVC (remember: versioning ≠ storage)
These Kaggle downloads currently work **without a Kaggle login** through the public API. If that ever changes, use the `kaggle` CLI with an API token.
```bash
# scripts/download.sh
K=https://www.kaggle.com/api/v1/datasets/download
curl -L -o /tmp/n2025.zip   $K/shivamshrivastava21/indian-job-market-dataset-2025-2026   # core
curl -L -o /tmp/n2019.zip   $K/promptcloud/jobs-on-naukricom                             # core
curl -L -o /tmp/n2017.zip   $K/PromptCloudHQ/jobs-on-naukricom                           # education prior
curl -L -o /tmp/plfs.zip    $K/pradnyakalvikatte/plfs-india-2017-18-to-2023-24           # education prior (~800 MB unzipped; or official MoSPI download)
curl -L -o /tmp/nco.zip     $K/nemaleshwarh/national-classification-of-organisationnco-2015
curl -L -o /tmp/ncod.zip    $K/shriabhinandansharma/nco-occupation-descriptions-dataset
curl -L -o /tmp/nptel.zip   $K/lakshyyaaaa/nptel-swayam-course-catalog
curl -L -o /tmp/cour.zip    $K/longnguyen3774/coursera-courses-metadata-for-analytics-2025
curl -L -o /tmp/onet.zip    https://www.onetcenter.org/dl_files/database/db_31_0_csv.zip
# unzip each into data/raw/<name>/
```
```bash
dvc init
dvc add data/raw/naukri_2025 data/raw/naukri_2019 data/raw/naukri_2017 data/raw/plfs \
        data/raw/nco_2015 data/raw/courses data/raw/onet_31_0      # → commit the .dvc files
dvc remote add -d storage <remote>
dvc push
```
PLFS is the big one (~120 MB per year). Start with just the 2023‑24 file, and only keep the columns we use (education, technical education, occupation, weight) in `data/interim`.
- **Versioning** means small `.dvc` and `dvc.lock` files that hold content hashes. These go in Git.
- **Storage** is the DVC *remote* that holds the actual bytes.
- **Remote choice:** the Google Drive remote now needs your own Google Cloud OAuth client or service account, because the default DVC app is blocked. Read DVC's gdrive docs first. **DagsHub** is a simpler option for a team: it gives a free DVC remote *and* a hosted MLflow server.
- Raw data never goes into Git. Add `.gitignore` before the first commit (100 MB push limit).
- Each new snapshot (e.g. a 2026 scrape) is just another `dvc add` and commit. That's our data-versioning demonstration, and it becomes "live traffic" for monitoring.

### Step 3: Data exploration (`notebooks/01_eda.ipynb`)
- Run a `ydata-profiling` report per year and save it to `reports/`.
- **Skills:**
  - check the separators (`,` in 2025, `|` in 2019) and the leading/trailing spaces;
  - look at casing and synonyms (`react` / `react.js` / `reactjs`), which go into `skill_aliases.yaml`;
  - count tags per posting and plot vocabulary coverage against vocabulary size;
  - treat postings with no skills (0.6% in 2025, 4% in 2019) as unusable.
- **Titles → families:**
  - look at the top 200 titles and at what the rules miss (~24%);
  - find where rules and Naukri's labels disagree on 2019, and why.
- **Class balance:** Software Development is ~25–33% of postings, while Legal and Hospitality are under 1%. Use macro‑F1 and `class_weight="balanced"`, and set a minimum family size (e.g. 100) in `params.yaml`.
- **Experience:** 2025 has `minimumExperience`/`maximumExperience`; 2019 has strings like `"5 - 10 yrs"`. Parse both to a number. Use the posting's **minimum** experience as the feature, since the user enters a single number.
- **Duplicates:** the same job is often reposted. Deduplicate on (title, company, skills).
- **Train/serve consistency:** only use features the form collects, i.e. skills and experience. Company, location and salary may predict the family, but the app doesn't ask for them, so they stay out of the model.

### Step 4: Features (`src/features/build.py`)
- **Skills → multi-hot vector over the top‑K vocabulary.** `CountVectorizer(analyzer=identity, binary=True, max_features=K)` works directly on a column of skill lists. `identity` must be a top-level function so the pipeline can be pickled. K (start at 1,500) lives in `params.yaml`.
- **Experience:** clip to 0–20 and scale.
- **Skill dropout (training only):** add copies of each training posting reduced to 2–6 random skills. Users enter ~5 skills while postings have ~8, so this makes training look like real use. Never apply it to the test set. The test set gets its own fixed 5-skill version for the headline metric.
- Put everything in an **sklearn `Pipeline` + `ColumnTransformer`**. The vocabulary is then learned from the training split only (no leakage), and the whole thing is one artifact to serve. The serving vocabulary *is* the search box's list.
- **Skill profiles (for O2/O3, pandas, not a model):** for each role, compute:
  - the share of postings asking for each skill, giving the core skills (≥ 8%) and the readiness weights;
  - skill co-occurrence counts, which order the learning path;
  - growth 2019 → 2025;
  - the 5 most common raw titles;
  - the typical experience range.

  This is written to `models/skill_profiles.json` and versioned by DVC like the model.

### Step 5: Baseline models
| Model | Why |
|---|---|
| `DummyClassifier` (majority class) | The floor: 8% for roles |
| **Profile matching** (no training) | Cosine similarity between the user's skills and each role's skill profile. The ML model has to beat this, or ML isn't adding anything |
| Logistic Regression | Simple, interpretable: 66–67% top‑3 on 5-skill inputs in our check |
| Random Forest | Handles feature interactions without tuning |
| LightGBM | Usually the strongest model on tabular data and handles sparse input well |

Use a stratified split with a fixed seed, both set in `params.yaml`.
**Metrics:**
- **Headline: top‑3 accuracy on 5-skill test inputs**, since the product shows three roles and users enter about five skills;
- top‑3 accuracy on full postings;
- top‑3 accuracy of the family implied by the predicted role;
- macro‑F1;
- a per-role report;
- a confusion matrix at family level (102×102 is unreadable).

Expect confusion between neighbouring roles: Java vs. Full Stack, Data Analyst vs. BI Developer, Sales Executive vs. BDE.

### Step 6: The DVC pipeline (`dvc.yaml`)
```yaml
stages:
  ingest:      # raw → tidy parquet; skills split, normalised, aliased
    cmd: uv run python -m src.data.ingest
    deps: [src/data/ingest.py, configs/skill_aliases.yaml, data/raw/naukri_2019.csv, data/raw/naukri_2025.xlsx]
    outs: [data/interim/postings_2019.parquet, data/interim/postings_2025.parquet]
  label:       # title → role family; scores the rules against Naukri's 2019 labels
    cmd: uv run python -m src.data.label
    deps: [src/data/label.py, configs/role_rules.yaml, data/interim/postings_2019.parquet, data/interim/postings_2025.parquet]
    params: [label]
    outs: [data/processed/labelled_2019.parquet, data/processed/labelled_2025.parquet]
    metrics: [reports/label_quality.json: {cache: false}]
  split:
    cmd: uv run python -m src.data.split
    deps: [src/data/split.py, data/processed/labelled_2025.parquet]
    params: [split]
    outs: [data/processed/train.parquet, data/processed/test.parquet]
  train:
    cmd: uv run python -m src.models.train
    deps: [src/models/train.py, src/features/build.py, data/processed/train.parquet]
    params: [features, train]
    outs: [models/model.pkl]
  evaluate:
    cmd: uv run python -m src.models.evaluate
    deps: [src/models/evaluate.py, models/model.pkl, data/processed/test.parquet]
    metrics: [reports/metrics.json: {cache: false}]
    plots: [reports/confusion_matrix.png]
  profiles:    # skill shares, 2019→2025 growth, typical titles per family
    cmd: uv run python -m src.models.profiles
    deps: [src/models/profiles.py, data/processed/labelled_2019.parquet, data/processed/labelled_2025.parquet]
    params: [profiles]
    outs: [models/skill_profiles.json]
  lookups:     # education prior (Naukri 2017 + PLFS via NCO), skill → course links (NPTEL/SWAYAM, Coursera)
    cmd: uv run python -m src.models.lookups
    deps: [src/models/lookups.py, configs/family_nco.yaml, data/raw/naukri_2017, data/raw/plfs, data/raw/nco_2015, data/raw/courses, models/skill_profiles.json]
    outs: [models/education_prior.json, models/course_links.json]
```
```yaml
# params.yaml
label:    {min_role_size: 100}
split:    {test_size: 0.2, seed: 42}
features: {top_k_skills: 1500, exp_clip: 20, dropout_copies: 2, dropout_min: 2, dropout_max: 6, eval_n_skills: 5}
train:    {model: logreg, C: 1.0, class_weight: balanced}
profiles: {top_n_skills: 15, top_n_titles: 5}
```
`dvc repro` reruns only the stages whose inputs changed. Editing `role_rules.yaml` reruns label → split → train → evaluate → profiles but not ingest. `dvc exp run -S features.top_k_skills=1000` plus `dvc exp show` gives the same experiment table the seniors showed on slide 14.

### Step 7: MLflow experiment tracking
```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```
```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("career-baselines")
with mlflow.start_run(run_name="logreg"):
    mlflow.set_tags({"git_commit": commit, "data_md5": raw_dvc_md5})  # ← links MLflow to DVC
    mlflow.log_params({**params["features"], **params["train"]})
    pipe.fit(X_train, y_train)
    mlflow.log_metrics({"top3_acc": top3, "macro_f1": f1, "label_agreement_2019": agree})
    mlflow.log_artifact("reports/confusion_matrix.png")
    mlflow.sklearn.log_model(pipe, name="model")
```
Tagging each run with the **git commit and DVC data hash** is what makes a run reproducible: we can always trace which code and which data produced it. Logging label agreement next to model metrics shows whether a gain came from better labels or a better model. Register the best baseline in the Model Registry now; the champion/challenger aliases are needed in later phases.

### Step 8: Initial evaluation
Compare runs in the MLflow UI and pick the best baseline. Write the following up in `journey.md`:
- its top‑3 accuracy and macro‑F1 against the profile-matching baseline;
- which families get confused (e.g. Software vs. ERP vs. QA, Sales vs. Banking);
- the labelling agreement and how it moved as the rules improved;
- the class imbalance.

Also run the 2019-trained model on 2025 once and record the drop. It's the first evidence for the monitoring phase.

### Suggested team split
1. **Data:** `download.sh`, ingest, `skill_aliases.yaml`, EDA
2. **Labels:** the role + family taxonomy, `role_rules.yaml` (~100 roles), reviewing each role's top skills for rule bugs, label-quality metric (target ≥ 80% family agreement on 2019)
3. **Models:** features, baselines (including profile matching), evaluation, MLflow logging
4. **Infra + lookups:** repo, uv, DVC remote, MLflow server, `dvc.yaml`, the skill-profiles and lookups stages (education prior from Naukri 2017 + PLFS, course links)

The infrastructure person shouldn't do everyone's setup. The seniors said to set up the tools yourselves, so each person should do their own `uv sync` and `dvc pull` at least once.

---

## 4. Roadmap summary

| # | MLOps stage | Tools for this project | Phase 1? |
|---|---|---|---|
| 1 | Problem definition & data collection | Naukri 2025 + 2019 (core); Naukri 2015–17, PLFS, NCO‑2015, NPTEL/SWAYAM, Coursera, O\*NET (supporting) | ✅ Done |
| 2 | Data cleaning & preprocessing | pandas, ydata-profiling, rule-based labelling, scikit-learn encoders | ✅ Done |
| 3 | Data versioning & storage | Git, DVC with a GDrive or DagsHub remote | ✅ Done (dataset versioning plus the pipeline) |
| 4a | Model development: baselines & tracking | scikit-learn, LightGBM, MLflow | ✅ Done |
| 4b | Model development: tuning (and optional neural model) | Optuna, optional Keras MLP | ⏳ Later |
| 5 | Validation & testing | MLflow registry (champion/challenger), pytest (rule and schema tests), SHAP (which skills drove a recommendation), CodeCarbon | 🟡 Partly (initial evaluation only) |
| 6 | Packaging & CI/CD | Docker or Podman, TF Serving (if neural), **quantization** (ONNX for sklearn/LightGBM, TFLite if neural), GitHub Actions | ⏳ Later |
| 7 | Deployment | FastAPI with Render or Hugging Face Spaces (or SageMaker) | ⏳ Later |
| 8 | Monitoring | Prometheus + Grafana **run as Docker images**. For drift, **not Evidently** (per seniors): use NannyML, Alibi Detect or custom chi-square/JS-distance tests on skill frequencies. 2019 = reference, 2025 = live; later a fresh scrape | ⏳ Later |
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
