# Hibi (PPIA): Personal Productivity Intelligence App

A Flask habit tracker. You log daily habits (deep work, sleep, stress, caffeine,
journaling, ...) and Hibi shows a yearly heatmap, a weekly grid, a dashboard and
a history page. An **experimental** estimate of a day's productivity is shown
too. It comes from models trained on **synthetic (simulated) data**, so it is
not a statement about you and there is no personal model.

## Setup (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # optional: set SECRET_KEY in .env
```

## Train the models

The trained files in `artifacts/` are not committed (`*.joblib`), so create them once:

```powershell
python -m ml.dummydataset --rows 10000   # synthetic data -> datasets/productivity_logs.csv
python -m ml.train                       # models + artifacts/metadata.json
python -m ml.evaluate                    # hold-out metrics, figures, evaluation_report.md
```

Without the models the app still works; the estimate card is just hidden.

## Run

```powershell
python run.py
```

Open http://127.0.0.1:5000, register, choose the habits you want to track, and log.

## Tests

```powershell
pip install pytest
python -m pytest tests -q        # tests/test_ml.py needs the dataset and trained models
```

## Layout

| Path | What it is |
|---|---|
| `app/habits.py` | habit definitions and the "goal met" rules |
| `app/stats.py` | dashboard numbers (pandas) over the habits a user tracks |
| `app/main/routes.py` | home, setup, dashboard, habit log, history, CSV export, profile |
| `app/insights/` | the ML estimate pages and the helper used by the home card |
| `ml/` | synthetic data, training, evaluation, prediction |

Your data export is at `/export.csv` (logged in), and every log can be edited or
deleted from `/history`.
