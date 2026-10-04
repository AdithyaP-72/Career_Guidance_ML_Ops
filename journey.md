# Build log

All numbers come from `dvc repro` (seed 42) and `reports/*.json`; every run is in MLflow (`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`).

## Data
| Dataset | Rows (usable) | Use |
|---|---|---|
| Naukri 2025 | 97,929 (87,259) | main training + the fixed held-out test set |
| Naukri 2019 | 30,000 (28,288) | label validation, drift reference, skill growth |
| Naukri 2022 | 32,738 (26,047) | extra training data (software-heavy: 55%+) |
| Internshala 2025 | 8,483 (7,587) | fresher / internship training data |
| Naukri 2015-17 | 22,000 | education prior (what employers ask) |
| PLFS 2023-24 | 164,523 respondents with an occupation | education prior (what people actually do) |
| NPTEL/SWAYAM, Coursera | 21,761 and 5,411 courses | course links |

The four files shared by friends (AI Job Market, Engineer Grad outcomes, Indian student placement, students.csv) are not used: three are synthetic, the other targets salary.

## Labelling
- 115 ordered regex rules -> **105 roles with at least 100 postings**, 22 families. Role rules are unit-tested (golden titles, substring traps like "unity" in "opportunity", "no generic labels").
- Bugs found and fixed along the way: `&amp;` in titles hid "Sales & Marketing"; "Back Office Executive" was swallowed by the admin rule (caught by a golden test); generic catch-all roles removed.
- Agreement with Naukri 2019 (independent of our rules): family 86.0% plausible / 70.9% IT-grouped / 59.8% strict; role 74.7% against the recruiter-chosen `Role`.

## Modelling experiments (5 random skills per held-out posting, top-3)
| Change | Top-3 |
|---|---|
| 76 roles, 1,500 skills, balanced (first model) | 70.2% |
| 105 roles, same settings (harder task) | 66.3% |
| + Naukri 2022 / Internshala training data | 65.0% / 65.1% (no help on 2025 test) |
| class_weight balanced -> none | 71.2% (+6 pts) |
| vocabulary 1,500 -> 2,500 -> 4,000 -> 6,000 | 73.8% -> 76.2% -> 77.0% |
| C 0.5, 2 dropout copies (kept) | **76.9%** |
LightGBM first collapsed to 3.7% (default settings diverge with 105 classes); after tuning it reaches 75.6% top-3 and a slightly higher macro-F1 than logistic regression, but logistic regression stays champion on top-3.

## Final results
| | Top-3 |
|---|---|
| Majority class | 18.3% |
| Profile matching (no training) | 65.1% |
| **Logistic regression (champion)** | **76.9%** (full postings 82.2%, in-vocabulary 5 skills 79.7%) |
| With 3 / 8 skills | 68.5% / 81.4% |
| Independent: vs recruiter-chosen 2019 roles | 85.8% |

## Drift
A logistic regression trained on 2019 only: 67.1% top-3 on 2019 held-out, 61.9% on 2025 (-5.2 pts); skill-distribution JS distance 0.41 (alert). Rising: SAP, project management, sales, software testing; fading: JavaScript, HTML, jQuery, MySQL. Live monitoring compares what users enter to training skills on the 60 most common skills (binned JS distance; unit-tested).

## Known weak spots
Marketing Executive vs Digital Marketing Executive, generic Software Engineer, and other neighbouring roles (`reports/per_role.csv`). The gold-set sheet (`scripts/make_gold_sample.py`) is ready for hand labelling to give a rules-free accuracy.
