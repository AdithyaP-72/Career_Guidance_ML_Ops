# Model Card — Indian Career Guidance System

> **Standard:** Follows the [Google Model Cards](https://modelcards.withgoogle.com/about) framework
> (Mitchell et al., 2019) and the [Hugging Face model card spec](https://huggingface.co/docs/hub/model-card-annotated).

---

## Model Details

| Field | Value |
|---|---|
| **Name** | Career Guidance Classifier |
| **Version** | 1.0 (Phase 2 baseline, Nov 2026) |
| **Type** | Multi-class text classification (103 Indian job roles) |
| **Champion baseline** | Complement Naïve Bayes (`complement_nb`) |
| **Input** | Up to 6 skills (normalised strings) + years of experience (optional) |
| **Output** | Top-3 predicted job roles with confidence scores (0–1) |
| **Framework** | scikit-learn 1.5 · Python 3.12 |
| **Artefact** | `models/model.joblib` (joblib, compressed, ~1.8 MB) |
| **DVC pipeline** | `dvc repro` — fully reproducible from raw data |
| **MLflow experiment** | `career-role-baselines` |
| **Authors** | Adarsh K R (project lead) |
| **Licence** | MIT (code) · CC BY-NC-SA 4.0 (Coursera course data, see §Data) |

---

## Intended Use

### Primary use
Recommend the top-3 Indian job roles that best match a user's self-reported skills
and experience, and surface a learning path (gap skills + free NPTEL/SWAYAM courses)
to reach those roles.

### Target users
Indian job seekers — fresh graduates, career changers, and working professionals —
using a web or mobile app that calls this model's API.

### Out-of-scope uses
- Automated hiring or resume screening (the model is **not** validated for
  employer-side decisions and must not be used that way).
- Non-Indian job markets (trained only on Naukri.com postings; role taxonomy and
  skill vocabulary are India-specific).
- Real-time high-stakes decisions without human review.

---

## Training Data

| Dataset | Years | Rows (after filtering) | Source |
|---|---|---|---|
| Naukri.com job postings | 2017, 2019, 2022, 2025 | ~69 k (train+test) | Naukri scrape (project-internal) |
| Reference set | 2019 | ~20 k | Same source, held out for drift measurement |

**Pre-processing:**
- Skills normalised with `src.utils.normalise_skill()` and `configs/skill_aliases.yaml`
- Postings with fewer than 3 skills excluded (`data.min_skills = 3`)
- Role labels assigned by rule-based taxonomy (`configs/role_rules.yaml`)
- Train/test split: 80/20 stratified by role, seed 42

**Class balance:** 103 roles; largest class = Software Engineer (general) at 8.5%
of training rows; smallest = 84 rows. No class weighting applied (balanced weights
cost ~6 percentage points of top-3 accuracy).

---

## Evaluation

### Test set (2025 postings, 5-skill inputs — closest to real usage)

| Metric | Champion (complement\_nb) |
|---|---|
| **Top-3 accuracy** | **80.0%** |
| Top-1 accuracy | 56.0% |
| Top-5 accuracy | 86.6% |
| Macro F1 | 0.519 |
| Family top-1 | 72.0% |
| Latency (ms / 1k) | 11.5 ms |
| Model size | 1.8 MB |

### Temporal drift (2019 reference set)

| Metric | 2025 test | 2019 reference | Drop |
|---|---|---|---|
| Top-3 accuracy | 80.0% | 69.9% | **−10.1 pp** |
| Top-1 accuracy | 56.0% | 44.2% | −11.8 pp |

> A ~10 pp drop is expected: roles, skill names, and market demand all shift over six years.
> New roles (e.g. Data Analyst, ML Engineer) were absent in 2019 postings.
> The retraining trigger (`src.monitoring.retrain_trigger`) flags a >15% drop as REQUIRED.

### Course recommendation (Objective O3)

| Metric | Value |
|---|---|
| Skills with ≥ 1 course | 59.7% (449/752) |
| Free NPTEL/SWAYAM coverage | 33.4% (251/752) |
| Relevance (50-pair manual check) | 98.0% |

---

## Limitations & Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Role-label bias** — labels come from rule-based taxonomy, not human annotation | Medium | Taxonomy rules reviewed; `reports/labels/` shows label distribution |
| **Skill-name variability** — many synonyms ("Python", "python3", "py") | Medium | 6 000-term vocabulary + alias YAML; normalisation pipeline |
| **Temporal drift** — model trained on 2025 data may degrade for older/future patterns | Medium | Reference-2019 evaluation at every training run; retraining trigger |
| **Rare roles** — 23 roles have < 200 training rows; predictions unreliable | Medium | Per-role F1 in `reports/eval/complement_nb/per_role_5skills.csv`; shown in dashboard |
| **Indian market only** — Naukri is India-centric; global roles may be misclassified | Low | Documented; out-of-scope use case |
| **No personal data** — model inputs are skills + experience; no PII | None | Privacy-safe by design |
| **Hallucinated courses** — O3 may link a skill to a weakly-matched course | Low | TF-IDF threshold + polysemy filter; relevance 98% on 50-pair sample |

---

## Fairness Considerations

The model predicts roles from skills, not from demographic attributes (gender, caste, region).
However:

- **Training data reflects market bias**: if certain skills are historically associated with
  demographics (e.g. nursing with women), the model may replicate that association through
  role predictions. This was **not** audited in Phase 2.
- **Future work (Phase 4+):** Conduct a fairness audit by intersecting role predictions with
  synthetic demographic proxies, and report disparate impact across skill sub-groups.

---

## Data Licences & Attributions

| Dataset | Licence | Attribution required |
|---|---|---|
| Naukri.com postings | Project-internal; not redistributed | — |
| NPTEL / SWAYAM courses | Public domain / CC BY | "NPTEL is an MHRD initiative, IIT consortium" |
| Coursera course metadata | **CC BY-NC-SA 4.0** | "Course data sourced from Coursera (CC BY-NC-SA 4.0)" — footer required in app UI |

> **Important:** The Coursera CC BY-NC-SA licence means the **app UI must display**:
> *"Course data sourced from Coursera (CC BY-NC-SA 4.0)"* whenever Coursera links appear.

---

## Monitoring & Governance

| Process | Tool / File | Frequency |
|---|---|---|
| Prediction logging | `src.monitoring.logger` → `logs/predictions.jsonl` | Every request |
| Distributional monitoring | `src.monitoring.monitor` → `reports/monitoring/` | Weekly |
| Drift experiment | `src.monitoring.drift_experiment` → `reports/drift/` | Each retrain |
| Retraining trigger | `src.monitoring.retrain_trigger` → `reports/retraining_trigger.json` | Weekly CI |
| Performance dashboard | `src.monitoring.dashboard` → `reports/performance_dashboard.html` | Weekly |
| Model comparison | `src.monitoring.model_comparison` → `reports/model_comparison_extended.csv` | Each retrain |
| DVC pipeline | `dvc repro` — all stages reproducible from raw data | On demand |
| MLflow experiment tracking | `mlruns/` or DagsHub MLflow server | Each training run |

**Retraining criteria (auto-detected by `retrain_trigger.py`):**
1. Top-3 accuracy drops > 15% from 2025 test to 2019 reference → REQUIRED
2. Jensen–Shannon divergence (live vs training distribution) > 0.15 → REQUIRED
3. > 25% of live requests have confidence < 0.30 → REQUIRED
4. Any one of the above → RECOMMENDED

---

## Citation

```bibtex
@misc{career_guidance_mlops_2026,
  title  = {Indian Career Guidance System – MLOps Phase 2},
  author = {Adarsh K R},
  year   = {2026},
  note   = {Naukri.com postings (2017–2025), scikit-learn, DVC, MLflow},
  url    = {https://dagshub.com/<your-dagshub-repo>}
}
```

---

*Model card generated: November 2026 · Next review due: May 2027 (or on retraining)*

