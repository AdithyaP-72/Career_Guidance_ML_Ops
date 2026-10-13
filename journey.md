# Phase 1 · Project Journey & Evaluation Report

This document records the architectural decisions, alternatives evaluated, baseline experiments, and evaluation findings for Phase 1.
Every baseline number here comes from `uv run dvc repro` (seed 42). The sources are `reports/model_comparison.csv`, `reports/metrics/*.json`, and `reports/eval/<model>/`, tracked under MLflow experiment `career-role-baselines`.

---

## 1. Product & Architecture Choices Considered

During initial project scoping, several problem framings and approaches were evaluated before settling on the tabular multi-class architecture:

| Option | Description | Verdict & Rationale |
|---|---|---|
| **A. Skills Profile $\to$ Role Match + Skill Gap (Selected)** | Tabular classifier over a fixed ~6k skill vocabulary + experience $\to$ top-3 of 103 canonical roles. | **Adopted.** Pure tabular classification without NLP overhead; fast, deterministic, reproducible, transparent feature contributions. |
| **A′. Resume Upload / Free-text Parsing** | Parsing PDF/Docx resumes and extracting skills using NLP/NER models. | **Rejected for Phase 1.** Introduces heavy NLP dependencies, extraction variance, and unstructured errors; potential future extension by matching extracted tokens against our skill vocabulary. |
| **B. Student Stream Recommender** | Recommending college degree/stream based on school subjects or psychometric tests. | **Rejected.** Available open-source datasets (Kaggle) were either tiny, synthetic, or disconnected from real employment outcomes. |
| **C. Pure LLM Chatbot** | Prompting an LLM (e.g. Claude, GPT-4) with skills to suggest careers. | **Rejected for MLOps.** No trainable weights, no data versioning, no drift tracking, high inference cost, non-deterministic outputs. |
| **D. Hybrid (Option A + LLM Explainer)** | Model predicts roles deterministically; LLM drafts custom learning advice. | **Saved for later phases.** Feasible as a UI polish once serving and monitoring are established. |

### Framing Constraint
The model is trained on **job postings**, not individual career histories. The model learns *"postings requiring these skills belong to this role"*, so the product phrasing must state *"your skills match what X jobs ask for"*, not *"people with your profile became X"*.

---

## 2. Dataset Evaluations & Rejections

Every dataset was audited for schema validity, real-world distribution, and licensing:

| Dataset | Why Rejected / Omitted |
|---|---|
| **Stack Overflow Developer Survey** | Software engineering only; fails the multi-industry scope (sales, finance, healthcare, etc.). |
| **Indeed India 2021 (PromptCloud, 60k)** | Contains category labels but **no skills column** — only free-text descriptions requiring full NLP. |
| **Naukri Job Listings 2020** | No structured skills column. |
| **Indian Fresher Job Market** | **Synthetic data:** exact uniform distribution across roles (~3.6%), roles completely independent of degree. |
| **Indian Tech Job Market 2026 (23k)** | Restricted strictly to Data/AI roles (only 6 categories). |
| **LinkedIn Jobs India (949 rows)** | Insufficient sample size. |
| **1.3M LinkedIn Jobs 2024** | Real and comprehensive, but restricted to US/UK/CA/AU; high storage footprint (6.2 GB). Kept as a potential future international benchmark. |
| **PLFS Microdata (MoSPI)** | Kept locally only (`.gitignore`); excluded from public DVC remote due to MoSPI redistribution restrictions. Aggregated priors will be published instead. |

---

## 3. Toolchain Choices & Senior Guidance

* **Package Management (`uv`):** Single tool replacing pip, virtualenv, and poetry. Pinned Python 3.12 via `.python-version` and locked sub-dependencies via `uv.lock`.
* **Data Profiling (`fg-data-profiling`):** Chose `fg-data-profiling` over `ydata-profiling` because `ydata-profiling 4.18` is deprecated and crashes on modern setuptools due to removed `pkg_resources`.
* **Data Versioning vs. Storage:** DVC manages hashes and pipeline DAGs committed in Git; DagsHub S3 remote acts purely as binary storage.
* **Drift Detection (Phase 8 note):** Evidently AI was explicitly avoided per senior advice. Future drift monitoring will leverage custom JS-divergence / chi-square tests or NannyML.
* **Containerization & Observability (Future):** Docker/Podman images for serving and Prometheus/Grafana instead of raw host binaries.

---

## 4. The Task & Experimental Setup

* **Input:** The skills a user possesses (5 in headline evaluation) and years of experience.
* **Output:** Top-3 of **103 specific roles**, grouped into 22 families across all sectors.
* **Training Data:** Naukri 2025 postings rule-labelled by `configs/role_rules.yaml`.
  * **Train Set:** 54,752 postings augmented with 2 cut-down copies each via **skill dropout** (164,256 training rows).
  * **Test Set:** 13,688 held-out postings, each downsampled to **5 random skills**.
* **Headline Metric:** Top-3 accuracy on the 5-skill test set (reflecting real user input patterns).

---

## 5. Baseline Evaluation Results

| Model | Top‑3 (5 skills) | Top‑1 (5 skills) | Macro‑F1 | Right Family | Top‑3 (All skills) | Train Time | Predict 1k Users | Model Size |
|---|---|---|---|---|---|---|---|---|
| **Dummy** (Prior baseline) | 19.8% | 8.5% | 0.002 | 17.5% | 19.8% | 1 s | 11 ms | 0.07 MB |
| **Profile Matching** (Cosine) | 66.3% | 43.5% | 0.455 | 62.9% | 73.9% | 2 s | 14 ms | 0.31 MB |
| **Logistic Regression** | **79.4%** | 56.5% | 0.553 | **72.1%** | 84.6% | 200 s | **27 ms** | 4.8 MB |
| **Random Forest** | 73.8% | 49.3% | 0.440 | 66.8% | 79.3% | 45 s | 144 ms | 112 MB |
| **LightGBM** | 78.2% | **56.8%** | **0.560** | 72.0% | **84.8%** | 843 s | 1026 ms | 8.1 MB |

### Key Findings
1. **Champion Model: Logistic Regression.**
   * Beats the non-learning baseline (Profile Matching) by **+13.1% top-3 accuracy**.
   * Outperforms LightGBM on top-3 accuracy by +1.2%, trains 4× faster, and predicts 38× faster (27 ms vs 1,026 ms per 1k users).
   * Random Forest performs significantly worse and produces an unwieldy model artifact (112 MB).
2. **Why Linear Models Win Here:**
   Job role prediction with sparse one-hot skill indicators is largely additive (`gst` + `tally` $\to$ Accountant; `react` + `javascript` $\to$ Frontend). Because inputs are extremely sparse (~5 active features out of 6,000), decision trees struggle to find split interactions that generalize better than additive log-odds.

---

## 6. What Moved the Numbers

Hyperparameter iterations measured on the 5-skill test set with Logistic Regression:

| Experiment / Modification | Top-3 Accuracy | $\Delta$ |
|---|---|---|
| 1,500-skill vocabulary, `class_weight: balanced` | 68.9% | Baseline |
| 1,500-skill vocabulary, unweighted (`class_weight: None`) | 74.6% | **+5.7%** |
| 6,000-skill vocabulary, unweighted | **79.4%** | **+4.8%** |

* **Class Weighting:** Enforcing `balanced` weights penalized frequent classes too aggressively, causing common postings to misclassify into niche roles.
* **Vocabulary Size:** Specific roles require specialized skill tags (e.g. `gnm` for Staff Nurse, `tds` for Accounts, `kotlin` for Android). Expanding the vocabulary from 1.5k to 6k gave immediate coverage gains without overfitting.
* **Data Ingestion Fix:** HTML entities in titles (`Sales &amp; Marketing`) previously broke title matching for 215 postings. Adding entity unescaping in `src/data/ingest.py` recovered them.
* **Label Rule Fixes:**
  * Fixed `unity` regex falsely matching *"Opportunity"* $\to$ Game Developer.
  * Fixed `hiring` pattern mislabelling BPO ads as Recruiter.
  * Fixed `secretary` catching *Company Secretary* and mapping to Executive Assistant.
  * Fixed generic *Architect* rule catching *Solution Architect* instead of Building Architect.

---

## 7. Error Analysis & Model Weaknesses

Detailed analysis from `reports/eval/logreg/`:

1. **Neighbouring Role Confusions (`reports/eval/logreg/top_confusions_5skills.csv`):**
   * *Customer Service Executive* $\leftrightarrow$ *Customer Support (Voice)* (109 / 83 confusions).
   * *Business Development Executive* $\leftrightarrow$ *Sales Executive* (99 / 55 confusions).
   * *Full Stack*, *Java*, *Frontend*, and *ML/AI Engineer* $\to$ *Software Engineer (general)*.
   * *Note:* Catch-all roles naturally absorb sparse inputs because their skill distributions overlap heavily with specialized roles.
2. **Family-Level Misclassifications (`reports/eval/logreg/family_confusion_5skills.png`):**
   * Software Development (86%), Customer Service/BPO (82%), HR (81%), and Sales (81%) have strong family precision.
   * **Banking postings often fall into Sales (26%)**, as branch/relationship management roles emphasize sales terminology.
   * **Strategy & Consulting is the weakest family (48%)**: business analysts and project managers share generic management skills across tech and analytics.
3. **Weakest Roles ($F_1 < 0.40$):**
   * Brand/Product Marketing Manager, Financial Analyst, Finance Manager, Cloud Engineer, Management Consultant, Automation Test Engineer, Warehouse/Inventory Executive, Microsoft Dynamics Consultant.
4. **Input Length Sensitivity:**
   * 5-skill inputs achieve 79.4% top-3 accuracy vs. 84.6% when all posting skills (~8) are provided. This confirms the product decision to enforce a minimum of 5 skills in the UI search box.
5. **Observed Temporal Drift (2019 $\to$ 2025):**
   * Evaluating the 2025-trained model against the 2019 reference set (100 stable classes) shows a drop to **66.8% top-3 accuracy** (-12.6%).
   * Rapidly rising skills (Kubernetes, Power BI, AWS, GenAI) were virtually absent in 2019, while declining skills (jQuery, PHP) had higher representation. This forms the foundation for Phase 8 drift monitoring.

---

## 8. Next Steps (Beyond Phase 1)

1. **Phase 4b (Tuning):** Run Optuna Bayesian optimization on Logistic Regression (regularization $C$, solver, tolerance) and LightGBM hyperparameters.
2. **Phase 5 (Testing & Registry):** Setup model cards, SHAP explainability for top skill attributions, and register the champion model under the MLflow Model Registry.
3. **Phase 6–7 (Packaging & Serving):** Quantize champion model to ONNX for sub-10ms inference and wrap in a FastAPI endpoint containerized via Docker.
4. **Phase 8 (Monitoring):** Establish baseline distribution metrics against the 2019/2025 split to detect incoming production skill drift.
