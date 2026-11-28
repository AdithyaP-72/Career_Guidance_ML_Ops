# Phase 3 faculty demonstration

## Set up before the presentation

Use Python 3.12, uv, and Podman or Docker with Compose. Commands below are for Linux/macOS bash; `scripts/phase3_lab.sh` wraps every step (run it with no arguments for the list). On Windows PowerShell, run the underlying `uv run python -m ...` commands shown in the script and set `$env:PHASE3_GIT_SHA` yourself. Use the project's existing DVC remote credentials. The real 2025 datasets and model must already be available to DVC. The optional `--registry-evidence` flag uses the existing MLflow connection to read the hosted aliases. Grafana credentials and the monitoring token are generated locally under `.phase3/secrets/`.

```bash
uv sync --locked
uv run dvc pull
scripts/phase3_lab.sh prepare --registry-evidence   # synthetic 2026 data, bundle, drift reference, secrets
scripts/phase3_lab.sh real-drift                    # real 2019 vs 2025 report (about 25 s)
scripts/phase3_lab.sh up                            # build + start API, Prometheus, Grafana; prints URLs
```

The script uses Podman when installed (it starts the rootless `podman.socket` that `podman compose` needs) and Docker otherwise. If the optional DagsHub evidence read fails, run `scripts/phase3_lab.sh prepare` without the flag and show the registry directly in DagsHub. If ports 8000/9090/3000 are taken, export `PHASE3_API_PORT`, `PHASE3_PROMETHEUS_PORT` or `PHASE3_GRAFANA_PORT` first; the script passes them to the demo too.

- API health is at `http://localhost:8000/health`.
- Prometheus alert status is at `http://localhost:9090/alerts`.
- Grafana is at `http://localhost:3000/d/career-phase3/`. Sign in as `admin` with `.phase3/secrets/grafana_password`.
- Open the hosted [DagsHub project](https://dagshub.com/AdithyaP-72/Career_Guidance_ML_Ops) and choose **Experiments** to view the team's MLflow runs and registry. Open the registered model `career-role-classifier` to show the hosted versions and aliases.

## Prepare in three to four hours

| Time | Practice | Explain it aloud |
|---|---|---|
| 40 min | Read the [architecture](PHASE3_ARCHITECTURE.md). Follow a resume skill list through the API and follow a model artifact from DVC to MLflow. | “Git versions the code, DVC identifies the data and model files, MLflow records the runs and gives a model version an alias.” |
| 55 min | Bring up Compose and run the labelled replay. Find the traffic, PSI, and alerts in Grafana and Prometheus. | “The API exports anonymous aggregates. Prometheus samples them and evaluates alerts. Grafana lets me see the numbers and history.” |
| 55 min | Open the challenger report and both registries. Walk through one metric, one gate, and the local audit trail. | “The shifted evaluation guides selection, real 2025 checks protect prior behaviour, and the held-out synthetic set checks the final choice.” |
| 35–60 min | Do two timed walkthroughs and practise the viva questions below. | Name the tool, the evidence it preserves, and the decision that evidence supports. |

## Run the demo

1. Show DagsHub's production champion and previous champion. Open `/health` locally and identify the serving model.
2. Show the three healthy containers and open the dashboard. Point out the live metrics and the timestamped repository evidence.
3. Run the command below. The first window sends previously unseen real 2025 records to `/predict`. The second sends synthetic 2026 records to that same model. The CLI waits for Prometheus alert evaluation.

   ```bash
   scripts/phase3_lab.sh demo     # about 3.5 minutes with the default 1,500 rows per window
   ```

4. Show the change in PSI, the feature distributions and the actual alert. Compare accepted-request accuracy and the all-submitted top-3 metric. The latter includes requests refused because the API could not form a usable prediction.
5. Show real drift too: the "Real temporal drift" table puts the same model on real 2019 postings next to the 2025 control (PSI about 0.26 critical vs 0.007, top-3 70% vs 80%). This is the evidence that the detector reacts to real market change, not just to the simulation.
6. Run `scripts/phase3_lab.sh retrain` (about 2 minutes). The first run fits and evaluates. An identical run reuses evidence. Open the stage, quality, comparison and governance panels once the report is written. Show the real 2025 retention result beside the shifted evaluation and final holdout.
7. Explain the decision. An eligible result permits an explicit local promotion and rollback. A rejected result is evidence that the controls work. See [model governance](MODEL_GOVERNANCE.md).
8. Point to the timestamped report and local MLflow run IDs. In a second terminal, open the local registry UI and select **Models**. A numbered version identifies a specific artifact. An alias such as `champion` is a movable pointer, so rollback can point it back at an earlier version.

   ```bash
   scripts/phase3_lab.sh promote "Your name"    # only after an eligible result
   scripts/phase3_lab.sh rollback "Your name"
   scripts/phase3_lab.sh registry               # MLflow UI for the isolated registry
   ```

   The local UI is at [localhost:5002](http://127.0.0.1:5002/). Compare the alias there with the hosted DagsHub alias. DagsHub stores production release evidence, and the live application is deployed separately.

If Wi-Fi is unreliable, prepare the images ahead of time and keep the generated `reports/phase3/demo/latest.json` and `reports/phase3/retraining/latest.json` open. The reports identify the experiment and the timestamps. Stop the lab with `scripts/phase3_lab.sh down` (add `-v` to also delete the Prometheus and Grafana volumes).

## Measured rehearsal on 27 November 2026

The hosted DagsHub snapshot showed calibrated ComplementNB v2 as `champion` and logistic regression v1 as `previous-champion`. Both replay windows used the same local bundle from calibrated ComplementNB. The real 2025 training reference consisted of 49,840 API-eligible rows. A separate full offline check of 12,357 eligible 2025 test rows gave **0.007 PSI** and normal status.

| Default API replay with 1,500 rows per window, seed 42 | Accepted / sent | Top-1 of accepted | Top-3 of accepted | Top-3 of all sent | HTTP p95 | Max PSI | Prometheus alert |
|---|---:|---:|---:|---:|---:|---:|---|
| Real 2025 control | 1,374 / 1,500 | 61.50% | 85.01% | 77.87% | 274 ms | 0.089 | No drift alert |
| Simulated 2026 shift | 1,495 / 1,500 | 58.93% | 83.48% | 83.20% | 250 ms | 0.627 | Critical firing |

An initial 1,000-row replay gave **0.115 PSI** for the control and **0.636** for the shift, with both alerts firing at their respective severity. The control's largest PSI came from the predicted-role distribution. A small replay sample, 103 role categories and a maximum across features make the warning threshold sensitive. The largest shift in both synthetic replays was AI skill presence. In the default replay, accepted-only top-3 fell by 1.53 percentage points, while more submissions received a prediction. Show both denominators. HTTP p95 includes simultaneous requests, server work and local transport. The quality gate measures single-row model inference separately.

The retraining experiment used full inputs for its drift check (**0.634 PSI**, 2,391 eligible synthetic evaluation rows) and the same five-skill representation for both models in the labelled comparison below.

| Labelled dataset | Top-1, baseline → challenger | Top-3, baseline → challenger | Macro F1, baseline → challenger | ECE, baseline → challenger |
|---|---:|---:|---:|---:|
| Synthetic 2026 evaluation | 53.92% → 61.46% | 79.79% → 84.13% | 0.5200 → 0.5325 | 0.0482 → 0.0802 |
| Synthetic 2026 holdout | 55.21% → 62.46% | 80.88% → 85.42% | 0.5452 → 0.5714 | 0.0599 → 0.0949 |
| Real 2025 test | 56.55% → 56.38% | 80.22% → 80.04% | 0.5436 → 0.5385 | 0.0719 → 0.0772 |

The paired synthetic evaluation gain was **4.33 percentage points** (95% bootstrap interval **3.29 to 5.46 points**). Calibration error rose, but the configured caps and real-data regression limits passed. Challenger size was **1.96 MB** and model-only p95 was **56.77 ms** in this rehearsal. All **13 existing quality gates and 8 lab comparison checks** passed.

The first eligible rehearsal recorded isolated baseline v1, run `e682834fee9342179dce63f031238718`, and challenger v3, run `8c6c8de392b74790bd3652a216029d8c`. The reviewer promoted **v1 → v3** and rolled back **v3 → v1**. The registry kept an audit entry for each change. The production run identifiable in the DagsHub snapshot was `c5d59f5f81ba4979bafafc27af11f949`. Fresh runs receive their own local versions and run IDs. Timing varies with machine load.

## Viva questions

- **Why choose PSI?** It compares binned distributions and gives a per-feature number we can explain and monitor. The thresholds require calibration on stable data.
- **Does drift prove the model became inaccurate?** A feature distribution change and a drop in correctness are separate measurements. We compare labelled rows to measure correctness.
- **Why hold the model fixed during replay?** Then changing outputs can be traced to changed inputs while the model is held constant.
- **Why reserve an untouched holdout?** It checks the result after model selection, reducing the chance that we report a result optimized for the comparison set.
- **Why do we version data, models and images?** Each can change independently. Their identifiers tell us what was evaluated and deployed and give us a rollback target.
- **What happens when the alert fires?** Prometheus records a sustained threshold crossing. A separate controlled job checks the labelled dataset and decides whether fitting is warranted.
- **Why keep the experiment in a separate registry?** A synthetic experiment can establish repeatability and expose regressions. Changing a production alias requires separate evidence and approval.
