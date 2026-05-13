# Beginner Setup Guide: From Download to Final Dashboard

This guide is written for beginners on Windows.

Goal: Build and run the full project end-to-end.

You will complete:
- Python pipeline setup
- Model training
- Power BI-ready export generation
- Power BI report creation with provided mapping files

## 0. What You Need

Install these first:
1. Python 3.11 (recommended)
2. VS Code
3. Power BI Desktop
4. Git (optional, only if you want clone instead of ZIP download)

## 1. Download and Install Tools

## 1.1 Install Python 3.11
- Open browser and go to https://www.python.org/downloads/
- Download Python 3.11.x for Windows (64-bit)
- During install, check Add Python to PATH
- Click Install Now

Verify in terminal:
- `py -3.11 --version`

## 1.2 Install VS Code
- Download from https://code.visualstudio.com/
- Install with default settings

## 1.3 Install Power BI Desktop
- Download from Microsoft Store or official Microsoft page
- Install and launch once

## 1.4 (Optional) Install Git
- Download from https://git-scm.com/download/win
- Install with default settings

## 2. Get Project Files

You can use either method.

## Method A: Download ZIP (beginner-friendly)
1. Download project ZIP from your source.
2. Extract into a folder, for example:
   - `C:/Users/HP/Documents/NetBeansProjects/crypto_forecasting_project`
3. Open this folder in VS Code:
   - File > Open Folder

## Method B: Clone with Git
1. Open terminal.
2. Run:
   - `git clone <your-repo-url>`
3. Open the cloned project folder in VS Code.

## 3. Create Python Environment

Inside VS Code terminal (PowerShell):

1. Move to project folder:
- `cd C:/Users/HP/Documents/NetBeansProjects/crypto_forecasting_project`

2. Create environment:
- `py -3.11 -m venv .venv311`

3. Activate environment:
- `.venv311/Scripts/Activate.ps1`

If activation is blocked, run once in PowerShell as Administrator:
- `Set-ExecutionPolicy RemoteSigned`

4. Upgrade pip:
- `python -m pip install --upgrade pip`

5. Install dependencies:
- `python -m pip install -r requirements.txt`

6. Verify key packages:
- `python -c "import pandas,statsmodels,prophet,streamlit,tensorflow; print('ok')"`

## 4. Run Training Pipeline

This step trains models and creates outputs.

Run:
- `python src/train.py`

Expected output files:
- `outputs/model_metrics.csv`
- `outputs/forecast_preview.csv`
- `outputs/model_recommendation.txt`
- `outputs/rolling_backtest.csv`

## 5. Generate Power BI Export Tables

Run:
- `python src/build_powerbi_exports.py`

Expected folder:
- `outputs/powerbi`

Key files inside:
- `master_btc_daily.csv`
- `forecast_comparison.csv`
- `model_metrics.csv`
- `sentiment_daily.csv`
- `sentiment_event_impact.csv`
- `risk_daily.csv`
- `regime_timeline.csv`
- `feature_importance.csv`
- `strategy_equity_curves.csv`
- `strategy_performance_summary.csv`
- `seasonality_monthly.csv`
- `seasonality_weekday.csv`
- `returns_daily_change.csv`

## 6. Open Streamlit Dashboard (Optional Demo)

Run:
- `python -m streamlit run src/dashboard.py --server.port 8503`

Open in browser:
- http://localhost:8503

If port is busy, change to 8504 or 8505.

## 7. Build Power BI Report (Main Deliverable)

## 7.1 Import Data
In Power BI Desktop:
1. Get Data > Text/CSV
2. Import all files from:
   - `outputs/powerbi`

## 7.2 Create Date Table
Use DAX:
- `Date = CALENDAR(MIN(master_btc_daily[Date]), MAX(master_btc_daily[Date]))`

Add Year/Month/Quarter/Weekday columns in Date table.

## 7.3 Build Relationships
Use the relationship plan from:
- [Power BI Build Guide](docs/POWERBI_BUILD_GUIDE.md)

## 7.4 Add DAX Measures
Copy-paste starter measures from:
- [Power BI Measures](docs/POWERBI_MEASURES.dax)

## 7.5 Build Visuals Page-by-Page
Follow exact drag-and-drop mappings from:
- [Power BI Visual Field Map](docs/POWERBI_VISUAL_FIELD_MAP.md)

Suggested order:
1. Executive Overview
2. Forecast & Uncertainty
3. Price Explorer & Candlesticks
4. Sentiment & News Impact
5. Remaining pages

## 8. Beginner Checklist Before Submission

- Training runs without errors.
- Power BI export generation runs without errors.
- Power BI report has all required pages.
- Slicers work (date/model/regime).
- Model comparison table is visible.
- Strategy page shows equity curves and summary table.
- Report file is saved and opens correctly.

## 9. Common Problems and Fixes

## Problem: Python command not found
Fix:
- Reinstall Python 3.11 and check Add Python to PATH.
- Use `py -3.11` command instead of `python`.

## Problem: venv activation error
Fix:
- Run PowerShell as Administrator once:
  - `Set-ExecutionPolicy RemoteSigned`

## Problem: TensorFlow import error
Fix:
- Confirm Python is 3.11, not 3.14.
- Recreate environment with 3.11 and reinstall requirements.

## Problem: Streamlit port already used
Fix:
- Use another port:
  - `python -m streamlit run src/dashboard.py --server.port 8505`

## Problem: Power BI date filter not working
Fix:
- Ensure Date table is marked as date table.
- Verify relationships from Date table to fact tables.

## 10. Final Deliverables

Prepare these for college/interview/portfolio:
1. Python code folder
2. Generated outputs folder
3. Power BI file (.pbix)
4. Short presentation (PPT/PDF)
5. Project report chapters

Use these project docs for final packaging:
- [Project Plan](docs/PROJECT_PLAN.md)
- [7-Day Roadmap](docs/ROADMAP_7_DAYS.md)
- [Report Chapter Plan](docs/REPORT_CHAPTER_PLAN.md)
