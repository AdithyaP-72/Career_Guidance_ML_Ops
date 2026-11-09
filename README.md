# AI-Based Career Guidance Assistant — MLOps Project

An end-to-end MLOps pipeline for Indian career path guidance. Given a user's skills and experience, the system recommends top-3 specific job roles, quantifies skill gaps and readiness, and recommends an ordered learning path mapped to free online courses.

---

## 1. Project Overview & Objectives

* **Input:** $\ge 5$ skills (from a fixed vocabulary of ~6,000 common skills) and years of experience.
* **Output:**
  1. Top-3 specific job roles (out of 103 canonical roles across 22 families) with probability confidence.
  2. Role readiness score & core skill gap analysis (have vs. missing).
  3. Ordered learning path with direct links to free NPTEL / SWAYAM / Coursera courses.

### Objectives & Evaluation Targets

| # | Objective | Measurement | Target / Baseline |
|---|---|---|---|
| **O1** | Specific role recommendation | Top-3 accuracy on 5-skill inputs (simulating real user interaction) | Calibrated Complement Naive Bayes candidate: **80.22% top-3** (Target: $\ge 75\%$) |
| **O2** | Skill gap & readiness | Weighted core skill overlap per role (from 2025 Naukri data) | Deterministic pandas calculations versioned by DVC |
| **O3** | Ordered learning path | Co-occurrence rank $\times$ market growth ($2019 \to 2025$) mapped to course catalog | NPTEL/SWAYAM matches for top skills |
| **O4** | Education prior | Separate education-to-role prior table (PLFS & Naukri historical data) | Independent lookup table (not direct NLP on degrees) |
| **O5** | Reproducible pipeline | Clone $\to$ sync $\to$ DVC pull $\to$ DVC repro gives identical metrics; all runs in MLflow | **Phase 1 Deliverable (Completed)** |
| **O6** | Real drift detection | $2019 \to 2025$ natural distribution drift benchmarking | 2019 reference set evaluated across all models |

---

## 2. Reproducibility Commands

To reproduce the exact dataset, feature pipeline, baselines, and evaluation metrics:

### Quick Reproduction (Linux / macOS)

```bash
# 1. Clone repository and set up Python 3.12 environment
git clone https://github.com/AdithyaP-72/Career_Guidance_ML_Ops.git
cd Career_Guidance_ML_Ops
uv sync

# 2. Configure the HTTPS DVC remote with local-only credentials
uv run dvc remote add --local -d dagshub-https https://dagshub.com/AdithyaP-72/Career_Guidance_ML_Ops.dvc
uv run dvc remote modify --local dagshub-https auth basic
uv run dvc remote modify --local dagshub-https user adithyap15122005
read -rs DAGSHUB_TOKEN
uv run dvc remote modify --local dagshub-https password "$DAGSHUB_TOKEN"
unset DAGSHUB_TOKEN

# 3. Pull versioned data and models
uv run dvc pull

# 4. Configure hosted MLflow before training
cp .env.example .env
# Edit .env: MLFLOW_TRACKING_USERNAME=adithyap15122005
# Set MLFLOW_TRACKING_PASSWORD to your DagsHub token.
# Keep MLFLOW_TRACKING_URI set to the DagsHub .mlflow URL in the example.

# 5. Test, reproduce, and inspect metrics and quality gate
uv run pytest -q
uv run dvc repro
uv run dvc metrics show
cat reports/model_card.md
```

### Quick Reproduction (Windows PowerShell)

```powershell
# 1. Clone repository and set up the environment
git clone https://github.com/AdithyaP-72/Career_Guidance_ML_Ops.git
cd Career_Guidance_ML_Ops
uv sync

# 2. Configure the HTTPS DVC remote with local-only credentials
uv run dvc remote add --local -d dagshub-https https://dagshub.com/AdithyaP-72/Career_Guidance_ML_Ops.dvc
uv run dvc remote modify --local dagshub-https auth basic
uv run dvc remote modify --local dagshub-https user adithyap15122005
$sec = Read-Host "Paste your DagsHub token" -AsSecureString
$tok = [System.Net.NetworkCredential]::new("", $sec).Password
uv run dvc remote modify --local dagshub-https password $tok
Remove-Variable sec, tok

# 3. Pull versioned data and models
uv run dvc pull

# 4. Configure hosted MLflow before training
Copy-Item .env.example .env
# Edit .env: MLFLOW_TRACKING_USERNAME=adithyap15122005
# Set MLFLOW_TRACKING_PASSWORD to your DagsHub token.
# Keep MLFLOW_TRACKING_URI set to the DagsHub .mlflow URL in the example.

# 5. Test, reproduce, and inspect metrics and quality gate
uv run pytest -q
uv run dvc repro
uv run dvc metrics show
Get-Content reports/model_card.md
```

DVC credentials stay in Git-ignored `.dvc/config.local`; MLflow credentials stay in Git-ignored `.env`. Use your own DagsHub username if your token belongs to another account. Without `MLFLOW_TRACKING_URI`, training logs to local `mlflow.db` and its runs will not appear in the hosted dashboard. After reproduction, each `reports/runs/*.json` should have the DagsHub URL in `tracking`; the selected run's `mlflow_model_uri` in `reports/metrics.json` must start with `models:/`.

---

## 3. Phase 1 Model Results & Tracking

Nine candidate models were trained across 164,256 augmented postings (54,752 unique train postings with skill dropout) and evaluated against a held-out test set of 13,688 postings cut to 5 skills.

### Candidate Comparison (`reports/model_comparison.csv`)

| Model | Top-3 (5 skills) | Top-1 (5 skills) | Top-5 (5 skills) | Macro-F1 | Right Family | ECE (5 skills) | Top-3 (Full) | Top-3 (2019, 5 skills) |
|---|---|---|---|---|---|---|---|---|
| **Calibrated Complement Naive Bayes** (Selected candidate) | 80.22% | 56.54% | 86.59% | 0.5436 | 72.40% | 0.0719 | 83.42% | 70.16% |
| Complement Naive Bayes (Raw) | 80.03% | 55.97% | 86.61% | 0.5194 | 72.03% | 0.4336 | 83.59% | 69.92% |
| Logistic Regression | 79.39% | 56.47% | 85.84% | 0.5527 | 72.09% | 0.0476 | 84.58% | 66.75% |
| LightGBM | 78.19% | 56.75% | 84.53% | 0.5595 | 72.00% | 0.0681 | 84.81% | 67.09% |
| MLP (Neural Network) | 76.40% | 53.84% | 83.30% | 0.5238 | 70.54% | 0.2831 | 81.30% | 61.27% |
| SGD (Modified Huber) | 76.22% | 56.75% | 81.12% | 0.5492 | 71.83% | 0.0293 | 83.16% | 62.05% |
| Random Forest | 73.75% | 49.26% | 80.65% | 0.4401 | 66.79% | 0.1249 | 79.28% | 62.08% |
| Profile Matching (Cosine) | 66.33% | 43.51% | 75.47% | 0.4549 | 62.89% | 0.1871 | 73.86% | 55.97% |
| Dummy (Prior baseline) | 19.78% | 8.53% | 25.70% | 0.0015 | 17.47% | 0.0000 | 19.78% | 17.55% |

### Recorded Training and Inference Costs

| Model | Fit Time | Predict 1k | Single Request | Model Size |
|---|---|---|---|---|
| **Calibrated Complement Naive Bayes** (Selected candidate) | 25.6 s | 100.16 ms | 60.49 ms | 1.90 MB |
| Complement Naive Bayes (Raw) | 5.5 s | 43.63 ms | 18.48 ms | 1.80 MB |
| Logistic Regression | 70.9 s | 7.11 ms | 2.33 ms | 4.80 MB |
| LightGBM | 454.0 s | 591.10 ms | 12.73 ms | 8.15 MB |
| MLP (Neural Network) | 2469.7 s | 12.86 ms | 3.30 ms | 36.67 MB |
| SGD (Modified Huber) | 5.8 s | 14.38 ms | 17.95 ms | 2.89 MB |
| Random Forest | 14.0 s | 66.97 ms | 33.73 ms | 111.96 MB |
| Profile Matching (Cosine) | 0.7 s | 9.92 ms | 5.98 ms | 0.31 MB |
| Dummy (Prior baseline) | 1.1 s | 7.68 ms | 3.70 ms | 0.07 MB |

**Isotonic-calibrated Complement Naive Bayes is the selected candidate**, with **80.22% top-3 accuracy** and **0.0719 ECE** on five-skill test inputs. Selection uses the highest `top3_5skills` across configured models; `models/model.joblib` holds that candidate and `reports/metrics.json` records its metrics and MLflow model URI. A selected candidate is distinct from a deployed registry champion; registration and deployment are outside this change.

The raw ComplementNB remains in the comparison: it reaches **80.03% top-3**, but its low confidence produces roughly **0.43 ECE**, exceeding the quality gate. The calibrated challenger wraps it in `CalibratedClassifierCV(method="isotonic", cv=3, ensemble=False)`. With `ensemble=False`, inference uses one fitted classifier plus calibrators rather than three models. Class-wise calibration can change role rankings, so both ranking accuracy and ECE are measured. Calibration CV operates on augmented training rows; related variants of one source posting may span folds. The held-out test postings remain separate.

The `validate` DVC stage checks the selected candidate against `params.yaml`: **top-3 accuracy ≥ 75%**, **ECE ≤ 0.10**, and **median warmed single-request latency ≤ 100 ms**. It writes `reports/validation.json` and `reports/model_card.md`, then fails the pipeline if a measurement is missing, invalid, or outside its limit. Check for `Quality gate: PASSED` in the model card before proposing promotion. These checks do not register or promote the model.

Top-3 accuracy means the correct role appears among the three highest-ranked recommendations. Macro-F1 gives equal weight to each role, while Right Family checks the family of the top-ranked role. ECE measures the weighted accuracy-confidence gap for top-ranked predictions in 10 uniform bins; lower is better. The 2019 column evaluates 20,183 historical reference postings and provides a distribution-shift benchmark. Single Request is the median of 20 warmed one-posting predictions including feature preprocessing; Predict 1k is batch throughput. Timings depend on hardware. The machine-readable sources are `reports/metrics/*.json`, `reports/model_comparison.csv`, and `reports/metrics.json`.

* Detailed evaluation findings and confusion patterns are documented in **[journey.md](journey.md)**.
* **Hosted MLflow Experiment Tracker:** View parameter curves, confusion plots, and model artifacts at:
  👉 [https://dagshub.com/AdithyaP-72/Career_Guidance_ML_Ops.mlflow](https://dagshub.com/AdithyaP-72/Career_Guidance_ML_Ops.mlflow)

---

## 4. Repository Structure

```
├── configs/
│   ├── sources.yaml             # Raw dataset URLs, licences & file filter
│   ├── skill_aliases.yaml       # Normalisation rules for synonymous skills
│   ├── role_rules.yaml          # Rule-based title taxonomy mapping to 103 roles
│   └── functional_area_map.yaml # Functional Area alignment to 22 families
├── data/                        # DVC-managed data files (gitignored)
│   ├── raw/<source>/            # Raw snapshots (2017, 2019, 2022, 2025, NCO, Courses)
│   ├── interim/                 # Cleaned postings.parquet, labelled.parquet
│   └── processed/               # classes.json, train/test/reference_2019 parquets
├── models/                      # DVC-managed model checkpoints & skill profiles
│   ├── baselines/               # Trained candidate models (.joblib)
│   ├── model.joblib             # Selected candidate (calibrated ComplementNB)
│   └── skill_profiles.json      # Precomputed role skill profiles & co-occurrence matrix
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory Data Analysis notebook
├── reports/                     # Metrics, taxonomy support, and evaluation reports
│   ├── metrics.json             # Selected model performance
│   ├── model_comparison.csv     # Metrics summary table for all candidates
│   ├── validation.json          # Selected candidate quality checks
│   ├── model_card.md            # Quality gate result and evaluation notes
│   ├── eval/                    # Per-model confusion matrices and error reports
│   └── profiling/               # fg-data-profiling HTML reports
├── src/
│   ├── data/                    # Ingestion, profiling, labelling, taxonomy & splitting
│   ├── features/build.py        # CountVectorizer + experience ColumnTransformer
│   └── models/                  # Estimators, training, MLflow tracking & model selection
├── tests/                       # Pytest test suite (features, models, serialization & quality gates)
├── dvc.yaml & params.yaml       # DVC multi-stage pipeline definition & hyperparams
├── pyproject.toml & uv.lock     # Fully-pinned dependencies managed with uv
└── journey.md                   # Initial baseline analysis and findings
```

---

## 5. Project Details & Background

### 5.1 Datasets
* **Naukri 2025 (~98k postings):** Main current market training dataset. Provides multi-skill tags and experience range across all Indian sectors.
* **Naukri 2019 (30k postings):** Historical snapshot with explicit functional roles used for drift evaluation and label taxonomy verification.
* **Naukri 2017 & 2022:** Supplementary snapshots used for multi-year trend and taxonomy stability checks.
* **Supporting Taxonomies & Courses:** NCO-2015 occupation codes, NPTEL/SWAYAM course catalogs, Coursera course metadata, and O*NET technology tags.

### 5.2 Labelling & Taxonomy Resolution
Because raw postings contain over 55,000 messy free-text titles, labels are curated using deterministic regex rules in `configs/role_rules.yaml`. The taxonomy stage classifies roles into:
* **Stable:** Sufficient support in both 2025 ($\ge 100$) and 2019 ($\ge 20$) snapshots.
* **Emerging:** Sufficient support in 2025 only (e.g. *ServiceNow Developer*, *Trust & Safety*).
* **Excluded:** Infrequent roles pruned from training classes.

### 5.3 Feature Engineering
A scikit-learn `ColumnTransformer` combines:
1. **Binary Skill Matrix:** `CountVectorizer(binary=True, max_features=6000)` fitted exclusively on train data.
2. **Experience Scaler:** Median imputation $\to$ clip to $[0, 20]$ years $\to$ min-max scaling $[0, 1]$.
3. **Skill Dropout Augmentation:** Simulates sparse human inputs during training by generating variants with 2–6 skills sampled randomly from posting tags.

### 5.4 MLOps Architecture & Roadmap

| Stage | Focus | Tools Used / Planned | Status |
|---|---|---|---|
| **1–2** | Ingestion & Features | pandas, pyarrow, scikit-learn, fg-data-profiling | ✅ Done |
| **3** | Versioning & Storage | Git, DVC, DagsHub Remote Storage (HTTPS DVC) | ✅ Done |
| **4a** | Baselines & Tracking | scikit-learn, LightGBM, MLflow (DagsHub hosted) | ✅ Done |
| **4b** | Hyperparameter Tuning | Optuna Bayesian optimization | ⏳ Next |
| **5** | Validation & Testing | pytest, skops safe loading, ECE/accuracy/latency gates; SHAP planned | 🟡 In progress |
| **6** | Packaging | Docker / Podman, ONNX model quantization | ⏳ Planned |
| **7** | Serving & API | FastAPI, Render / Hugging Face Spaces | ⏳ Planned |
| **8** | Monitoring | Prometheus, Grafana, NannyML / custom JS divergence | ⏳ Planned |
| **9** | Continuous Retraining | GitHub Actions automated `dvc repro` workflow | ⏳ Planned |
