# AI-Based Career Guidance Assistant — MLOps Project

> Working plan for the project: what we're building, how we tackle Phase 1, and the full MLOps roadmap.

---

## 1. What the final product could look like

The seniors' project is a useful template: **user input goes in, a trained model predicts, the prediction is shown in a UI, and the MLOps tooling wraps around it.** They used NLP because sentiment analysis is a text problem. Career guidance doesn't have to be one: if users describe themselves with a structured form (skills, tools, education, experience), it becomes a **tabular** problem. We use NLP only if we decide we need free-text input such as resumes.

| # | Product | Input → Output | Model type | Good datasets | Verdict |
|---|---|---|---|---|---|
| **A** | **Skills profile → career match + skill-gap report** | Checklist of languages/tools/frameworks, education, years of experience → top‑3 job roles with confidence, skills you have vs. skills people in that role typically have, what to learn next | Tabular classifier (LogReg / Random Forest / XGBoost / LightGBM), no NLP | **Stack Overflow Developer Survey** (real responses, ~50–90k per year, open ODbL license, new release every year), O\*NET for role descriptions | ⭐ **Recommended.** Real data, simple features, and yearly releases give a genuine data-drift story for monitoring |
| A′ | Same as A, but from a **resume upload** | Resume PDF → same output | Needs NLP (text classification or skill extraction) | Kaggle *Resume Dataset* (~2.4k resumes, ~24 categories), O\*NET skills lists | Only if we specifically want resume upload. Could be bolted onto A later as a simple keyword-matching step |
| B | Student stream/major recommender | Marks, interests, personality quiz (RIASEC) → recommended stream or degree | Tabular classifier | O\*NET interest profiles, Kaggle student-career datasets | Easy, but thin. Small or synthetic datasets, little to version, tune or monitor |
| C | Career chatbot (LLM + RAG) | Chat → advice | Pretrained LLM, no training | O\*NET, job descriptions | Impressive demo, but there's no model to train, track or retrain, so it fits Phase 1 badly |
| D | **A + a small LLM layer** | Same as A, plus a chat box that explains *why* | A is the MLOps-managed model; the LLM only rephrases its output | Same as A | Good stretch goal for later phases |

**Recommendation: A.** A′ and D are optional add-ons later. Roughly what users would see:

```
┌─ CareerCompass ─────────────────────────────────┐
│  Languages:  [x] Python [x] SQL [ ] Java [ ] Go   │
│  Tools:      [x] Git [ ] Docker [x] Excel …       │
│  Education:  [ B.E./B.Tech ▾ ]  Experience: [ 0 ] │
│  [ Analyze ]                                      │
├───────────────────────────────────────────────────┤
│  Top matches                                      │
│   1. Data Analyst ............ 78%                │
│   2. Data Scientist .......... 52%                │
│   3. Back-end Developer ...... 31%                │
│  Skill gap → Data Analyst                         │
│   ✔ Python  ✔ SQL  ✘ Power BI  ✘ Tableau          │
│  Next steps: Power BI basics, a statistics course │
└───────────────────────────────────────────────────┘
```

Behind the UI, the full system ends up like this: a Streamlit or React frontend calls a FastAPI service, which runs a Dockerized model pulled from the MLflow registry ("champion"). Predictions are logged to a monitoring stack (Prometheus, Grafana and a drift check). When drift shows up, GitHub Actions runs `dvc repro` to retrain, and the new "challenger" model is compared against the champion.

**Why the Stack Overflow survey suits MLOps:** a new release comes out every year. We can train on one year and treat the next year's respondents as "live traffic". Tech popularity really does shift between years, so the drift we detect in later phases is real, not simulated.

**Limitation to accept:** the survey covers tech/software roles only. That fits an audience of engineering students, but it isn't career guidance for every field.

---

## 2. Phase 1: Development & Reproducibility

> Dataset acquisition/approval; data exploration; feature engineering; baseline models; Git repository; DVC dataset versioning; MLflow experiment tracking; initial model evaluation.

**What "done" means for Phase 1:** a teammate can run

```bash
git clone <repo> && uv sync && dvc pull && dvc repro
```

and get the **same metrics**, with every run visible in MLflow. Every step below works toward that.

### Step 0: Decisions to lock in on day 1 (seniors' "versions!!!" advice)
- **One package manager: `uv`.** It's fast, it writes a lockfile (`uv.lock`) and it pins the Python version. Never mix in `pip install` or conda. Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **Python 3.12.**
- **Decide now whether a neural model (e.g. a Keras MLP) comes later, or whether we stay with scikit-learn/XGBoost.** It decides the serving setup. A TensorFlow model pairs with TF Serving and has built-in Prometheus metrics, which is what the seniors used, and TFLite makes quantization easy. Tree models get served through FastAPI or MLflow serving and are quantized through ONNX. If TensorFlow is coming, read its NumPy compatibility notes before adding other packages.
- **Dataset approval:** get the dataset choice (and which survey year(s)) approved by faculty before building on it.

### Step 1: Repository structure (cookiecutter-data-science style)
```
career-guidance/
├── data/
│   ├── raw/            # DVC-tracked, never edited by hand
│   ├── interim/
│   └── processed/      # outputs of the pipeline
├── notebooks/          # 01_eda.ipynb, 02_baselines.ipynb
├── src/
│   ├── data/prepare.py
│   ├── features/build.py
│   └── models/{train.py, evaluate.py}
├── models/             # DVC output
├── reports/            # metrics.json, plots, profiling HTML
├── params.yaml         # every tunable value lives here
├── dvc.yaml            # the pipeline
├── pyproject.toml + uv.lock
├── .gitignore  .dvcignore
└── README.md / journey.md
```

```bash
uv init --python 3.12
uv add pandas scikit-learn xgboost mlflow dvc ydata-profiling
uv add --dev ruff jupyter pytest
```
"Pandas Profiler" is now called **`ydata-profiling`**. It has lagged behind new NumPy releases before, so check its supported versions before adding it.

### Step 2: Git and DVC (remember: versioning ≠ storage)
- **Versioning** means small `.dvc` and `dvc.lock` files that hold content hashes. These go in Git.
- **Storage** is the DVC *remote* that holds the actual bytes.
```bash
dvc init
dvc add data/raw/survey_2024.csv      # creates data/raw/survey_2024.csv.dvc → commit it
dvc remote add -d storage <remote>
dvc push
```
- **Remote choice:** the Google Drive remote now needs your own Google Cloud OAuth client or service account, because the default DVC app is blocked. Read DVC's gdrive docs first. **DagsHub** is a simpler option for a team: it gives a free DVC remote *and* a hosted MLflow server.
- Raw data never goes into Git (the survey CSV is large). Add a `.gitignore` before the first commit (100 MB push limit).
- Adding a new survey year later is just another `dvc add` and commit. That makes it a clean demonstration of data versioning.

### Step 3: Data exploration (`notebooks/01_eda.ipynb`)
- Run a `ydata-profiling` report and save it to `reports/`.
- **Target column:** look at the job-role column (`DevType`). Check whether it is single- or multi-select in the year we pick, since that decides multi-class vs. multi-label. Drop respondents with no role, and decide what to do with students and "Other".
- **Class balance:** a few roles (e.g. full-stack) dominate. Merge or drop very rare roles, and use macro-F1 rather than accuracy.
- **Missing values:** many survey questions are optional. Measure how much is missing per column.
- **Multi-select columns** (e.g. `LanguageHaveWorkedWith`) are semicolon-separated strings. Look at how many distinct values each one has.
- **Train/serve consistency:** use only features a user could actually enter in our form (skills, tools, education, experience). Columns such as salary or company size may predict the role, but the app can't ask for them, so they don't belong in the model.

### Step 4: Feature engineering (`src/features/`)
- **Multi-hot encode** the multi-select skill columns (one 0/1 column per language/tool/framework). Group very rare skills into "other".
- **Ordinal encode** education level and years of experience (convert values like "Less than 1 year" to numbers).
- Impute or flag missing values.
- **Skill-gap profiles:** for each role, compute the share of people in that role who use each skill. The app compares a user against these profiles to show "✘ missing" skills. It's plain pandas, not a model, but it is a pipeline output, so version it with DVC too.
- Optionally use the `…WantToWorkWith` columns to drive "what to learn next" suggestions.
- Put preprocessing **inside an sklearn `Pipeline`/`ColumnTransformer`** so it is fit only on the training split (no leakage). The whole pipeline is then one artifact to serve later.

### Step 5: Baseline models
| Model | Why |
|---|---|
| `DummyClassifier` (majority class) | The floor that every other model must beat |
| Logistic Regression | Simple, interpretable baseline |
| Random Forest | Handles feature interactions without tuning |
| XGBoost / LightGBM | Usually the strongest model on tabular data |

Use a stratified split with a fixed seed, both set in `params.yaml`.
**Metrics:** macro-F1, **top‑3 accuracy** (`top_k_accuracy_score`, since the product shows three careers), a per-class report and a confusion matrix.

### Step 6: The DVC pipeline (`dvc.yaml`)
```yaml
stages:
  prepare:
    cmd: uv run python -m src.data.prepare
    deps: [src/data/prepare.py, data/raw]
    params: [prepare]
    outs: [data/processed]
  train:
    cmd: uv run python -m src.models.train
    deps: [src/models/train.py, data/processed]
    params: [train]
    outs: [models/model.pkl]
  evaluate:
    cmd: uv run python -m src.models.evaluate
    deps: [src/models/evaluate.py, models/model.pkl, data/processed]
    metrics: [reports/metrics.json: {cache: false}]
    plots: [reports/confusion_matrix.png]
```
`dvc repro` reruns only the stages whose inputs changed. `dvc exp show` gives the same experiment table the seniors showed on slide 14.

### Step 7: MLflow experiment tracking
```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```
```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("career-baselines")
with mlflow.start_run(run_name="xgboost"):
    mlflow.set_tags({"git_commit": commit, "data_md5": raw_dvc_md5})  # ← links MLflow to DVC
    mlflow.log_params(params["train"])
    pipe.fit(X_train, y_train)
    mlflow.log_metrics({"macro_f1": f1, "top3_acc": top3})
    mlflow.log_artifact("reports/confusion_matrix.png")
    mlflow.sklearn.log_model(pipe, name="model")
```
Tagging each run with the **git commit and DVC data hash** is what makes a run reproducible: we can always trace which code and which data produced it. Register the best baseline in the Model Registry now; the champion/challenger aliases are needed in later phases.

### Step 8: Initial evaluation
Compare runs in the MLflow UI and pick the best baseline. Write up its metrics, its confusion patterns (which roles get mixed up, e.g. data analyst vs. data scientist) and the imbalance and missing-data findings in `journey.md`. Feature importances from the tree models are a quick first look at what drives predictions, ahead of SHAP in a later phase.

### Suggested team split
1. Data acquisition, EDA and target/class cleanup
2. Feature engineering and skill-gap profiles
3. Baselines and evaluation
4. Repo, uv, DVC remote and MLflow server setup

The infrastructure person shouldn't do everyone's setup. The seniors said to set up the tools yourselves, so each person should do their own `uv sync` and `dvc pull` at least once.

---

## 3. Roadmap summary

| # | MLOps stage | Tools for this project | Phase 1? |
|---|---|---|---|
| 1 | Problem definition & data collection | Stack Overflow Developer Survey, O\*NET | ✅ Done |
| 2 | Data cleaning & preprocessing | pandas, ydata-profiling, scikit-learn encoders | ✅ Done |
| 3 | Data versioning & storage | Git, DVC with a GDrive or DagsHub remote | ✅ Done (dataset versioning plus a basic pipeline) |
| 4a | Model development: baselines & tracking | scikit-learn, XGBoost, MLflow | ✅ Done |
| 4b | Model development: tuning (and optional neural model) | Optuna, optional Keras MLP | ⏳ Later |
| 5 | Validation & testing | MLflow registry (champion/challenger), pytest, SHAP, CodeCarbon | 🟡 Partly (initial evaluation only) |
| 6 | Packaging & CI/CD | Docker or Podman, TF Serving (if neural), **quantization** (TFLite/ONNX), GitHub Actions | ⏳ Later |
| 7 | Deployment | FastAPI with Render or Hugging Face Spaces (or SageMaker) | ⏳ Later |
| 8 | Monitoring | Prometheus + Grafana **run as Docker images**. For drift, **not Evidently** (per seniors): use NannyML, Alibi Detect or custom chi-square/JS-distance tests, with the next survey year as "live" data | ⏳ Later |
| 9 | Continuous training & feedback | GitHub Actions → `dvc repro` → challenger vs. champion, plus user feedback ("was this helpful?") | ⏳ Later |

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
