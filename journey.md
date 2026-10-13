# Phase 1 · Initial model evaluation

What the first baselines show, what moved the numbers, and where the best model goes wrong.
Every number here comes from `uv run dvc repro` (seed 42). The sources are `reports/model_comparison.csv`, `reports/metrics/*.json` and `reports/eval/<model>/`, and the same runs are in MLflow (experiment `career-role-baselines`).

## The task
- **Input:** the skills a person has (5 in the headline test) and their years of experience.
- **Output:** the top‑3 of **103 specific roles**, grouped into 22 families, across all sectors.
- **Data:** Naukri 2025 job postings, labelled from their titles by rules (`configs/role_rules.yaml`).
  - **Train:** 54,752 postings, plus 2 cut-down copies of each ("skill dropout"), so the model sees short inputs like a real user's.
  - **Test:** 13,688 held-out postings, each cut to **5 random skills**.
- **Headline metric:** top‑3 accuracy on that 5-skill test, because the product shows three roles and users type about five skills.

## Results

| Model | Top‑3 | Top‑1 | Macro‑F1 | Right family | Top‑3 with all skills | Train | Predict 1k users |
|---|---|---|---|---|---|---|---|
| Dummy (3 most common roles) | 19.8% | 8.5% | 0.002 | 17.5% | 19.8% | 2 s | 19 ms |
| Profile matching (no learning) | 66.3% | 43.5% | 0.455 | 62.9% | 73.9% | 2 s | 35 ms |
| **Logistic regression** | **79.4%** | 56.5% | 0.553 | **72.1%** | 84.6% | 3 min | **21 ms** |
| Random forest | 73.8% | 49.3% | 0.440 | 66.8% | 79.3% | 42 s | 149 ms |
| LightGBM | 78.2% | **56.8%** | **0.560** | 72.0% | **84.8%** | 15.5 min | 807 ms |

**Best baseline: logistic regression.**
- **Headline:** it beats the no-ML approach (profile matching) by **+13.1 points of top‑3**, so the learned skill weights clearly add value.
- **Against LightGBM:** LightGBM is within 0.3 points on top‑1 and macro‑F1 but 1.2 points behind on top‑3, takes 5× longer to train and is 40× slower to predict.
- **Against random forest:** it does worse here and its model file is 23× bigger (107 MB).

The likely reason the linear model wins: "which skills point to which role" is mostly a sum of independent clues (`gst` → accounting, `react.js` → frontend). With only ~5 active skills out of 6,000, tree models see too little per split to learn interactions that pay off.

## What moved the numbers
Measured with logistic regression on the 5-skill test:

| Change | Top‑3 |
|---|---|
| Start: 1,500-skill vocabulary, `class_weight: balanced` | 68.9% |
| No class weights | 74.6% (+5.7) |
| 6,000-skill vocabulary | **79.4%** (+4.8) |

- **Class weights:** `balanced` pushes predictions toward rare roles, which costs top‑k accuracy on the real role mix.
- **Vocabulary:** specific roles need specific skills (`kotlin`, `tds`, `gnm`), and many of those sit outside the top 1,500.
- **Credit:** both ideas came from a teammate's exploratory branch and were re-measured on our pipeline before adopting them.

Data fixes that came out of the same review:
- HTML entities in titles (`Sales &amp; Marketing`) hid 215 postings from the title rules. They're now decoded in `ingest`.
- Earlier rule fixes are recorded in README §2.3: the "unity" in "opportunity" trap, "hiring" BPO ads labelled as Recruiter, Company Secretary filed as Executive Assistant, and "Solution Architect" caught by the building-Architect rule.

## Where the best model goes wrong
From `reports/eval/logreg/`:
- **Most errors are between neighbouring roles** (`top_confusions_5skills.csv`):
  - Customer Service Executive ↔ Customer Support (Voice) (109 / 83 test postings);
  - Business Development Executive ↔ Sales Executive (99 / 55);
  - Area/Territory Sales Manager, Relationship Manager (Banking) and Inside Sales → Sales Executive;
  - Full Stack, Java, Frontend and ML/AI Engineer → Software Engineer (general).

  Many of these are arguably *also right*. The generic catch-all roles (Sales Executive, Software Engineer (general)) attract postings that share their skills.
- **Family level** (`family_confusion_5skills.png`):
  - Software Development (86%), Customer Service/BPO (82%), HR (81%) and Sales (81%) are reliable;
  - **Banking postings often land in Sales (26%)**, since bank RM/branch jobs are sales jobs with banking words;
  - **Strategy & Consulting is the weakest family (48%)**: consultants, project managers and business analysts share generic skills with Data & Analytics and Software.
- **Weakest roles** (`per_role_5skills.csv`, F1 < 0.4, 8 of 103 roles): Brand/Product Marketing Manager, Financial Analyst, Finance Manager, Cloud Engineer, Management Consultant, Automation Test Engineer, Warehouse/Inventory, Microsoft Dynamics Consultant. 12 roles reach F1 ≥ 0.7, e.g. Trust & Safety, School Teacher, Recruiter, Staff Nurse, SAP Consultant.
- **More skills help:** 79.4% top‑3 with 5 skills vs 84.6% with all of a posting's skills (~8). That supports the "at least 5 skills" rule in the form.
- **On 2019 postings** (the 100 roles that existed then), the same 2025-trained model scores 66.8% top‑3, 12.6 points lower. Older postings use partly different skills (e.g. jQuery and PHP then, Kubernetes and Power BI now). This is the first sign of the drift the monitoring phase will track.

## Limitations to keep in mind
- **Labels come from title rules, not people.** Family-level agreement with Naukri's own 2017/2019 labels is ~67%, but much of the gap is Naukri's coarser taxonomy (README §2.3). Some test "errors" are label noise, and some model "hits" could be too.
- **Postings, not people:** the model learns what job ads ask for. It says "your skills match what these jobs ask for", not "people like you became…".
- **Only top‑3 was used to choose the best model.** Top‑1 and macro‑F1 are near-tied between logistic regression and LightGBM.

## Next phases (not Phase 1)
- **Tuning:** Optuna on logistic regression and LightGBM, using these runs as the baseline.
- **Labelling:** tighter rules for the catch-all roles (Sales Executive, Software Engineer (general), Management Consultant).
- **Model Registry:** champion/challenger aliases for the selected model.
- **Possible neural-net challenger:** a skill-embedding layer, which could help the sparse 5-skill inputs.
- **Monitoring:** the 2019 vs 2025 comparison above as the drift baseline.
