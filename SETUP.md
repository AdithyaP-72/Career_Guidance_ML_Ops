# Setup and run (Windows, macOS, Linux)

The app runs **fully offline on your own machine**. The trained model ships inside the repo (`models/champion/`, 7 MB), so you do **not** need to download any data or train anything just to use it.

## 1. Quick start: run the app (about 2 minutes)

### Windows (PowerShell or Command Prompt)
```powershell
git clone https://github.com/AdithyaP-72/Career_Guidance_ML_Ops.git
cd Career_Guidance_ML_Ops
git checkout Adi6k-build-main
.\run.bat
```
`run.bat` installs `uv` (a small Python manager) if it is missing, installs the dependencies (including the right Python 3.12), starts the app and opens http://127.0.0.1:8000 in your browser. You can also just double-click `run.bat` in File Explorer.

### macOS / Linux (Terminal)
```bash
git clone https://github.com/AdithyaP-72/Career_Guidance_ML_Ops.git
cd Career_Guidance_ML_Ops
git checkout Adi6k-build-main
chmod +x run.sh && ./run.sh
```

Stop the app with `Ctrl + C` in the terminal.

### Prefer to do it by hand?
```bash
# 1) install uv once:   Windows:  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
#                       macOS/Linux:  curl -LsSf https://astral.sh/uv/install.sh | sh
#    then reopen the terminal
uv sync                       # installs Python 3.12 + all pinned dependencies
uv run python run.py          # starts the app and opens the browser
uv run python run.py --no-browser --port 8080    # optional: another port, no auto-open
```

## 2. Using the app
1. **Find my career**: type a skill, pick it from the list (↑/↓ + Enter works), add 5 to 10 skills, set experience and education, press *Find my matches*.
2. Click any of the three matches to see readiness, why it matched, missing skills and the learning path with course links.
3. **Explore roles**: browse what each role asks for. **Insights**: model quality, 2019→2025 drift and live usage.

## 3. Rebuilding the model from raw data (optional)
Only needed if you change the data, the labelling rules (`configs/role_rules.yaml`) or `params.yaml`. Needs about 1.5 GB of free disk and an internet connection for the one-time download.

```bash
uv run python manage.py download     # fetches the datasets from Kaggle's public API into data/raw/
uv run python manage.py pipeline     # data -> labels -> model -> metrics -> profiles -> lookups -> drift -> promote
uv run python manage.py test         # 40 tests
uv run python manage.py compare      # logistic regression vs random forest vs LightGBM (about 6 minutes)
uv run python manage.py simulate     # send sample traffic so the Insights "Live usage" panel has data
```
(`manage.py` replaces `make`, which Windows does not have. On macOS/Linux, `make serve`, `make pipeline`, etc. also work.)

The Naukri 2025 file is `indian-job-market-dataset-2025.xlsx`. If the automatic download of any dataset fails, the script prints the Kaggle page; download it manually and unzip into `data/raw/<name>/` (names: `naukri_2025`, `naukri_2019`, `naukri_2022`, `naukri_2017`, `internshala`, `nptel`, `coursera`, `plfs`). For PLFS only `perv1_2023_24.csv` is needed.

Experiment tracking UI (optional): `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db` then open http://127.0.0.1:5000.

## 4. Troubleshooting
| Problem | Fix |
|---|---|
| `uv` is not recognised right after installing | Close and reopen the terminal (or run `$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"` in PowerShell) |
| PowerShell blocks `run.bat` / scripts | Run `powershell -ExecutionPolicy Bypass -File .\run.bat`, or use the manual steps above in Command Prompt |
| Port 8000 already in use | `uv run python run.py --port 8080` |
| Page shows "Cannot reach the API" | Keep the terminal window open; the app lives in it. Reload the page |
| `models/champion` missing | You are not on the `Adi6k-build-main` branch, or the pipeline has not run: `uv run python manage.py pipeline` |
| LightGBM error on macOS about `libomp` | `brew install libomp` (only affects `manage.py compare`) |
| Antivirus / firewall prompt | The app only listens on 127.0.0.1 (your own machine); allow it |

## 5. Project layout (short)
`run.py` launcher · `app/static/index.html` the web page · `src/api/` FastAPI service · `src/models/` training, recommender, lookups · `src/monitoring/` drift · `configs/` role rules and mappings · `models/champion/` the model the app serves · `reports/` metrics. More in `README.md`.
