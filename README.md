# AI-Based Career Guidance Assistant — MLOps Project

> Working plan for the project: what we're building, how we tackle Phase 1, and the full MLOps roadmap.

---

## 1. What the final product could look like

The seniors' project is a useful template: **text goes in, a trained model predicts, the prediction is shown in a UI, and the MLOps tooling wraps around it.** Our project has the same shape. The difference is that the model predicts career paths instead of emotions.

| # | Product | Input → Output | Model type | Good datasets | Verdict |
|---|---|---|---|---|---|
| **A** | **Resume / profile → career match + skill-gap report** | Resume PDF or skills text → top‑3 job roles with confidence, skills you have vs. skills you're missing, suggested next steps | Text classification (TF‑IDF/embeddings → classifier), plus skill matching | Kaggle *Resume Dataset* (~2.4k resumes, ~24 categories), Kaggle *LinkedIn Job Postings*, **O\*NET** (skills per occupation, free, CC‑BY) | ⭐ **Recommended.** Closest to the seniors' pipeline, with real NLP and a clear demo |
| B | Student stream/major recommender | Marks, interests, personality quiz (RIASEC) → recommended stream or degree | Tabular classifier (XGBoost/RF) | O\*NET interest profiles, Kaggle student-career datasets | Easy, but thin. Little to version, tune or monitor |
| C | Career chatbot (LLM + RAG) | Chat → advice | Pretrained LLM, no training | O\*NET, job descriptions | Impressive demo, but there's no model to train, track or retrain, so it fits Phase 1 badly |
| D | **A + a small LLM layer** | Same as A, plus a chat box that explains *why* | A is the MLOps-managed model; the LLM only rephrases its output | Same as A | Good stretch goal for later phases |

**Recommendation: A, with D as an optional add-on later.** Roughly what users would see:

```
┌─ CareerCompass ─────────────────────────────────┐
│  [ Upload resume.pdf ]   or   paste your skills  │
│  [ Analyze ]                                      │
├───────────────────────────────────────────────────┤
│  Top matches                                      │
│   1. Data Analyst ............ 78%                │
│   2. Business Analyst ........ 61%                │
│   3. ML Engineer ............. 34%                │
│  Skill gap → Data Analyst                         │
│   ✔ Python  ✔ SQL  ✘ Tableau  ✘ Statistics        │
│  Next steps: learn Tableau basics, stats course…  │
└───────────────────────────────────────────────────┘
```

Behind the UI, the full system ends up like this: a Streamlit or React frontend calls a FastAPI service, which runs a Dockerized model pulled from the MLflow registry ("champion"). Predictions are logged to a monitoring stack (Prometheus, Grafana and a drift check). When drift shows up, GitHub Actions runs `dvc repro` to retrain, and the new "challenger" model is compared against the champion.

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
- **Choose the deep learning framework now (TensorFlow or PyTorch), even though Phase 1 only uses scikit-learn.** The choice decides the serving setup later. TensorFlow pairs with TF Serving and has built-in Prometheus metrics, which is exactly what the seniors used. It also pins NumPy, so read its compatibility notes before adding anything else.
- **Dataset approval:** get the dataset choice approved by faculty before building on it.

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
uv add pandas scikit-learn nltk spacy mlflow dvc ydata-profiling
uv add --dev ruff jupyter pytest
```
"Pandas Profiler" is now called **`ydata-profiling`**. It has lagged behind new NumPy releases before, so check its supported versions before adding it.

### Step 2: Git and DVC (remember: versioning ≠ storage)
- **Versioning** means small `.dvc` and `dvc.lock` files that hold content hashes. These go in Git.
- **Storage** is the DVC *remote* that holds the actual bytes.
```bash
dvc init
dvc add data/raw/resumes.csv          # creates data/raw/resumes.csv.dvc → commit it
dvc remote add -d storage <remote>
dvc push
```
- **Remote choice:** the Google Drive remote now needs your own Google Cloud OAuth client or service account, because the default DVC app is blocked. Read DVC's gdrive docs first. **DagsHub** is a simpler option for a team: it gives a free DVC remote *and* a hosted MLflow server.
- Raw data never goes into Git. Add a `.gitignore` before the first commit (100 MB push limit).

### Step 3: Data exploration (`notebooks/01_eda.ipynb`)
- Run a `ydata-profiling` report and save it to `reports/`.
- **Class balance:** some job categories will be rare, so use macro-F1 rather than accuracy.
- Look at text-length distributions and duplicates (resume datasets often contain near-duplicates).
- **Leakage check:** many scraped resumes start with the job title, which is often the label itself. Strip the title line or we'll see fake 95% accuracy.
- **PII:** resumes contain names, emails and phone numbers. Scrub them during preprocessing; examiners notice this.

### Step 4: Feature engineering (`src/features/`)
- Cleaning: lowercase the text and remove URLs, emails, phone numbers and stopwords. Lemmatize with NLTK or spaCy.
- **TF‑IDF** on word 1–2 grams, optionally with character n‑grams as well.
- **Skill extraction:** match text against O\*NET's technology-skills list with spaCy's `PhraseMatcher`. This produces multi-hot skill features *and* the skill-gap feature of the product, so one step serves both.
- Optional stronger features: sentence embeddings (`all-MiniLM-L6-v2`).
- Put the vectorizer **inside an sklearn `Pipeline`** so it is fit only on the training split (no leakage). The whole pipeline is then one artifact to serve later.

### Step 5: Baseline models
| Model | Why |
|---|---|
| `DummyClassifier` (majority class) | The floor that every other model must beat |
| TF‑IDF + Logistic Regression | Strong, interpretable baseline |
| TF‑IDF + LinearSVC | Usually the best classic text model |
| TF‑IDF + MultinomialNB | Fast reference point |
| Embeddings + LogReg *(optional)* | Previews Phase 2 deep learning |

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
with mlflow.start_run(run_name="tfidf-logreg"):
    mlflow.set_tags({"git_commit": commit, "data_md5": raw_dvc_md5})  # ← links MLflow to DVC
    mlflow.log_params(params["train"])
    pipe.fit(X_train, y_train)
    mlflow.log_metrics({"macro_f1": f1, "top3_acc": top3})
    mlflow.log_artifact("reports/confusion_matrix.png")
    mlflow.sklearn.log_model(pipe, name="model")
```
Tagging each run with the **git commit and DVC data hash** is what makes a run reproducible: we can always trace which code and which data produced it. Register the best baseline in the Model Registry now; the champion/challenger aliases are needed in later phases.

### Step 8: Initial evaluation
Compare runs in the MLflow UI and pick the best baseline. Write up its metrics, its confusion patterns (which careers get mixed up) and the leakage and imbalance findings in `journey.md`.

### Suggested team split
1. Data acquisition, EDA and PII cleaning
2. Feature engineering and skill extraction
3. Baselines and evaluation
4. Repo, uv, DVC remote and MLflow server setup

The infrastructure person shouldn't do everyone's setup. The seniors said to set up the tools yourselves, so each person should do their own `uv sync` and `dvc pull` at least once.

---

## 3. Roadmap summary

| # | MLOps stage | Tools for this project | Phase 1? |
|---|---|---|---|
| 1 | Problem definition & data collection | Kaggle resumes, job postings, O\*NET | ✅ Done |
| 2 | Data cleaning & preprocessing | pandas, ydata-profiling, NLTK/spaCy | ✅ Done |
| 3 | Data versioning & storage | Git, DVC with a GDrive or DagsHub remote | ✅ Done (dataset versioning plus a basic pipeline) |
| 4a | Model development: baselines & tracking | scikit-learn, MLflow | ✅ Done |
| 4b | Model development: deep models & tuning | DistilBERT or BiLSTM, Optuna / keras-tuner | ⏳ Later |
| 5 | Validation & testing | MLflow registry (champion/challenger), pytest, SHAP/LIME, CodeCarbon | 🟡 Partly (initial evaluation only) |
| 6 | Packaging & CI/CD | Docker or Podman, TF Serving, **quantization** (TFLite/ONNX), GitHub Actions | ⏳ Later |
| 7 | Deployment | FastAPI with Render or Hugging Face Spaces (or SageMaker) | ⏳ Later |
| 8 | Monitoring | Prometheus + Grafana **run as Docker images**. For drift, **not Evidently** (per seniors): use NannyML, Alibi Detect or custom KS/JS-distance tests | ⏳ Later |
| 9 | Continuous training & feedback | GitHub Actions → `dvc repro` → challenger vs. champion, plus user feedback ("was this helpful?") | ⏳ Later |

**What Phase 1 covers:** stages 1–4a and the start of 5. That means approved and explored data, versioned with DVC, run through a reproducible pipeline, with tracked baseline experiments in MLflow and an initial evaluation. Everything after that (deep models, tuning, packaging, serving, monitoring, retraining) builds on this foundation.

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
