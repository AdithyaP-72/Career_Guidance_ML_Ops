# Model governance

CareerCompass supports career exploration with a fixed role taxonomy. The model card documents weak groups, calibration, data sources and intended use. Role coverage is a model-quality check. A demographic fairness assessment would require suitable data and separately defined measures.

| Decision | Evidence and accountability |
|---|---|
| Detect and investigate | Check sample count and data origin, inspect top drifted features and confirm whether the shift is sustained |
| Compare models | Compare both on the same labelled data, enforce the real quality gate and the allowed real-score regression, then record the holdout check |
| Register | Write the version, algorithm, configuration, fit timestamp, Git revision, dataset and model hashes and run ID to isolated MLflow and JSON reports |
| Approve | An eligible local experiment needs `--promote --approved-by NAME`. The identifier records the operator's declaration |
| Roll back | Restore the previous experimental champion and record who did it and when |
| Release production | Obtain independent real-world validation and explicit approval. Registration, alias promotion and deployment each have their own records |

The lab's `career-role-classifier-phase3-synthetic-2026` registry is separate from the hosted production registry. The reference champion and candidate are stored in an isolated SQLite database under the report directory. Even an eligible synthetic candidate requires a separate operator action.

```bash
scripts/phase3_lab.sh promote "Demo reviewer"    # = uv run python -m scripts.retrain_phase3 --promote --approved-by ...
scripts/phase3_lab.sh rollback "Demo reviewer"
```

Run the first command after an **eligible** result. If the report rejects promotion, show the failed gate and use the isolated alias tests to demonstrate that a passing candidate can be promoted and rolled back. In production, an alert starts investigation. Available options include examining input validation, checking service health, obtaining labelled evaluation data, and reviewing release history.

The monitor stores normalized vocabulary identifiers and aggregate counts, with no resume contents in its observations. Limit dashboard access to reviewers and keep credential files out of Git. For production use, assign a responsible model owner, define a review and incident procedure, and agree on retention and access rules for aggregate evidence.

## Model card summary

The quality gate regenerates the full model card, with every metric and gate value, at [reports/model_card.md](../reports/model_card.md) on each `dvc repro`. Numbers live there so they cannot go stale here.

| Field | Value |
|---|---|
| Model | `career-role-classifier`, calibrated ComplementNB (`complement_nb_cal`), MLflow registry on DagsHub |
| Input | 5 to 60 skills from a fixed ~6,000-term vocabulary, plus optional years of experience |
| Output | Ranked roles out of 103, with calibrated confidence, the skill gap and course links |
| Training data | Naukri 2025 postings: 54,752 train and 13,688 test rows. 2019 reference split of 20,183 rows for drift |
| Labels | Rule-based on job titles (`configs/role_rules.yaml`), not human-annotated |
| Artefact | `model.skops` in the image bundle. skops loads only types on an allow-list, so loading cannot run arbitrary pickle code |

### Intended use
Career exploration for Indian students, graduates and career changers: "jobs that ask for skills like mine" and a learning path to close the gap.

### Out of scope
- Hiring, screening or ranking candidates. The model is not validated for employer-side decisions.
- Job markets outside India. The vocabulary and roles come from Naukri.
- Roles outside the 103 classes, and any high-stakes decision without human review.

## Why every artefact is versioned

| Artefact | Versioned by | Reason |
|---|---|---|
| Code, configs, workflows | Git commit; image tag and `/health` report the SHA | Show which logic produced a prediction; review changes before merge |
| Data and trained files | DVC hashes in `dvc.lock`, stored on DagsHub | Reproduce a model exactly; prove which postings trained it |
| Model | MLflow registry version numbers with `@champion`, `@previous-champion` and `@challenger` aliases | A version is immutable. Aliases move, so promotion and rollback are a pointer change that leaves an audit trail and needs no retrain |
| Serving image | GHCR tag per commit | Redeploy a known-good image if a release misbehaves |
| Drift reference | `drift_reference.json` pinned to the model's run ID and vocabulary hash | A reference from another model would give meaningless PSI. The API refuses to start with a mismatch |

## Data handling and privacy

- Resume files are parsed in memory to extract skills. They are not stored or logged, and a test asserts that resume text never reaches the logs.
- Each prediction log line holds counts, the predicted role, confidence, latency and the model version only. The drift monitor keeps vocabulary IDs and bins, not raw strings, and a test asserts that a submitted skill string does not appear in events, logs or metrics.
- `/metrics` and `/drift` are off unless a bearer token is configured, and the Compose stack binds to loopback only. Secrets live in GitHub Actions secrets, `.env` and `.phase3/secrets/`, all outside Git.

## Risks and fairness

| Risk | Mitigation |
|---|---|
| Rule-based labels overlap, for example "Software Engineer (general)" vs specific roles | Ranked top 3 instead of a single answer; family-level accuracy is also gated |
| Rare roles: 26 roles have fewer than 200 training rows | Per-role and per-family minimum top-3 accuracy in the quality gate |
| Temporal drift: the 2019 split loses about 10 points of top-3 accuracy, PSI about 0.26 | PSI monitoring, the 2019 gate check, and the challenger retraining path |
| Market bias: postings may encode demographic associations between skills and roles | The model has no demographic inputs. The association was not audited; that needs suitable data and defined measures |

## Licences and attribution

| Source | Licence | Obligation |
|---|---|---|
| Naukri postings | Project use only; not redistributed | Raw data stays in the DVC remote, not in Git |
| Coursera course metadata | CC BY-NC-SA 4.0 | Attribution wherever Coursera links appear. The web UI footer states it. Non-commercial use only |
| NPTEL / SWAYAM | Government of India open courseware | Attribution in the UI footer |
| NCO-2015 occupation codes | Government of India | Cited in the README |
