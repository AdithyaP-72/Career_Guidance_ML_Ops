# Selected model card

Model: `complement_nb_cal`

Quality gate: PASSED

| Metric | Measured | Required | Result |
|---|---|---|---|
| top3_5skills | 0.8022 | minimum 0.75 | PASS |
| ece_5skills | 0.0719 | maximum 0.1 | PASS |
| predict_ms_per_request | 60.4905 | maximum 100.0 | PASS |

Evaluation uses the held-out 2025 test set with five-skill inputs.
ECE uses the top-ranked role's confidence in 10 uniform bins, including confidence 1.
Latency is the median of 20 warmed single-posting predictions, including feature preprocessing.

Calibration CV runs on augmented training rows; variants of the same source posting may span folds.
Held-out test postings stay separate from training and calibration.

MLflow run: `5e971f2cb5e94e3f8a317dab2977de60`
MLflow model URI: `models:/m-e88cbcd46dd04fb6b1e102138e6ad2f9`

The candidate is not registered or promoted by this stage. Registry promotion follows PR merge.
