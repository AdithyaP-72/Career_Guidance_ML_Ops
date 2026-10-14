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
| **O1** | Specific role recommendation | Top-3 accuracy on 5-skill inputs (simulating real user interaction) | Champion Complement Naive Bayes: **80.03% top-3** (Target: $\ge 75\%$) |
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
# 1. Clone repository
git clone https://github.com/AdithyaP-72/Career_Guidance_ML_Ops.git
cd Career_Guidance_ML_Ops

# 2. Setup environment (Python 3.12 managed via uv)
uv sync

# 3. Configure DagsHub DVC remote credentials
read -rs DAGSHUB_TOKEN
uv run dvc remote modify origin --local access_key_id "$DAGSHUB_TOKEN"
uv run dvc remote modify origin --local secret_access_key "$DAGSHUB_TOKEN"
unset DAGSHUB_TOKEN

# 4. Pull versioned data and models from DagsHub
uv run dvc pull

# 5. Verify reproducibility and test suite
uv run pytest -q
uv run dvc repro             # Prints: "Data and pipelines are up to date."
uv run dvc metrics show
```

### Quick Reproduction (Windows PowerShell)

```powershell
# 1. Clone repository
git clone https://github.com/AdithyaP-72/Career_Guidance_ML_Ops.git
cd Career_Guidance_ML_Ops

# 2. Setup environment
uv sync

# 3. Configure DagsHub DVC remote credentials
$sec = Read-Host "Paste your DagsHub token" -AsSecureString
$tok = [System.Net.NetworkCredential]::new("", $sec).Password
uv run dvc remote modify origin --local access_key_id $tok
uv run dvc remote modify origin --local secret_access_key $tok
Remove-Variable sec, tok

# 4. Pull data & model artifacts
uv run dvc pull

# 5. Verify reproduction
uv run pytest -q
uv run dvc repro             # Prints: "Data and pipelines are up to date."
uv run dvc metrics show
```

---

## 3. Phase 1 Model Results & Tracking

Eight candidate models were trained across 164,256 augmented postings (54,752 unique train postings with skill dropout) and evaluated against a held-out test set of 13,688 postings cut to 5 skills.

### Baseline Comparison (`reports/model_comparison.csv`)

| Model | Top-3 (5 skills) | Top-1 (5 skills) | Top-5 (5 skills) | Macro-F1 | Right Family | Top-3 (Full) | Top-3 (2019, 5 skills) | Fit Time | Predict 1k | Model Size |
|---|---|---|---|---|---|---|---|---|---|---|
| **Complement Naive Bayes** (Champion) | 80.03% | 55.97% | 86.61% | 0.5194 | 72.03% | 83.59% | 69.92% | 1.4 s | 11.54 ms | 1.8 MB |
| Logistic Regression | 79.39% | 56.47% | 85.84% | 0.5527 | 72.09% | 84.58% | 66.75% | 73.1 s | 8.46 ms | 4.8 MB |
| LightGBM | 78.19% | 56.75% | 84.53% | 0.5595 | 72.00% | 84.81% | 67.09% | 321.7 s | 379.64 ms | 8.15 MB |
| MLP (Neural Network) | 76.40% | 53.84% | 83.30% | 0.5238 | 70.54% | 81.30% | 61.27% | 3002.9 s | 20.66 ms | 36.67 MB |
| SGD (Modified Huber) | 76.22% | 56.75% | 81.12% | 0.5492 | 71.83% | 83.16% | 62.05% | 4.7 s | 9.31 ms | 2.89 MB |
| Random Forest | 73.75% | 49.26% | 80.65% | 0.4401 | 66.79% | 79.28% | 62.08% | 19.9 s | 82.33 ms | 111.96 MB |
| Profile Matching (Cosine) | 66.33% | 43.51% | 75.47% | 0.4549 | 62.89% | 73.86% | 55.97% | 0.8 s | 11.76 ms | 0.31 MB |
| Dummy (Prior baseline) | 19.78% | 8.53% | 25.70% | 0.0015 | 17.47% | 19.78% | 17.55% | 0.9 s | 5.71 ms | 0.07 MB |


**Complement Naive Bayes is the current champion**, selected by the highest top-3 accuracy on five-skill test inputs (`select.metric: top3_5skills`). Its 80.03% score exceeds Logistic Regression by 0.64 percentage points and Profile Matching by 13.70 percentage points, meeting the 75% target. The selected artifact is `models/model.joblib`, with metrics recorded in `reports/metrics.json`.

The newly added challengers are **MLP** (76.40% top-3), **SGD with modified Huber loss** (76.22%), and **Complement Naive Bayes** (80.03%). MLP uses two hidden layers (256 and 128 units); SGD provides a fast linear classifier for sparse skill features.

Top-3 accuracy means the correct role appears among the model's three highest-ranked recommendations. Macro-F1 measures top-1 performance across roles, while Right Family measures whether the top-ranked role belongs to the correct family. The 2019 column evaluates the historical reference set of 20,183 postings; the champion scores 69.92%, compared with 80.03% on the held-out 2025 test set. This difference provides a historical distribution-shift benchmark. Training and inference timings reflect the recorded run environment and can vary by hardware.

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
│   ├── model.joblib             # Selected champion model (Complement Naive Bayes)
│   └── skill_profiles.json      # Precomputed role skill profiles & co-occurrence matrix
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory Data Analysis notebook
├── reports/                     # Metrics, taxonomy support, and evaluation reports
│   ├── metrics.json             # Selected model performance
│   ├── model_comparison.csv     # Metrics summary table for all baselines
│   ├── eval/                    # Per-model confusion matrices and error reports
│   └── profiling/               # fg-data-profiling HTML reports
├── src/
│   ├── data/                    # Ingestion, profiling, labelling, taxonomy & splitting
│   ├── features/build.py        # CountVectorizer + experience ColumnTransformer
│   └── models/                  # Estimators, training, MLflow tracking & model selection
├── tests/                       # Pytest test suite (43 passed unit tests)
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
| **3** | Versioning & Storage | Git, DVC, DagsHub Remote Storage (S3 API) | ✅ Done |
| **4a** | Baselines & Tracking | scikit-learn, LightGBM, MLflow (DagsHub hosted) | ✅ Done |
| **4b** | Hyperparameter Tuning | Optuna Bayesian optimization | ⏳ Next |
| **5** | Validation & Testing | pytest (43 tests), skops safe loading, SHAP analysis | 🟡 In progress |
| **6** | Packaging | Docker / Podman, ONNX model quantization | ⏳ Planned |
| **7** | Serving & API | FastAPI, Render / Hugging Face Spaces | ⏳ Planned |
| **8** | Monitoring | Prometheus, Grafana, NannyML / custom JS divergence | ⏳ Planned |
| **9** | Continuous Retraining | GitHub Actions automated `dvc repro` workflow | ⏳ Planned |
