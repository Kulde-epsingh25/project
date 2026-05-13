# Big Project Plan: Cryptocurrency Time Series Analytics

## 1. Objective and Scope

Build an end-to-end cryptocurrency analytics and forecasting platform that:

- Analyzes historical price behavior and volatility
- Forecasts future price movement with ARIMA, SARIMA, Prophet, and LSTM
- Integrates conceptual sentiment as exogenous signals
- Delivers a dashboard for executive and analyst workflows

This scope is aligned to the supplied PDF brief, TXT workflow notes, image references, and dashboard references.

## 2. Available Resources

### Core Data Assets
- bitcoin_data_2014_to_2025.csv
- BTC_USD_Yahoo.csv
- cryptocurrency.csv
- prophet_prepared_data.csv

### Project Definition and Workflow Assets
- Data Analytics Project.pdf
- TIME SERIES CRYPTO_WORKFLOW.txt
- TSA with Crpto Power BI.txt
- ARIMA.txt

### Visual and Dashboard Reference Assets
- WhatsApp image set (11 files)
- Power BI archives:
  - BTC_Crypto_TimeSeries_Dashboard.pbix.zip
  - Cryptocurrency PBI Dashboard (1).pbix.zip
  - Crypto_BITCOIN_POWER_BI_2.pbix.zip
  - STOCK ANALYSIS DASHBOARD.pbix.zip

## 3. Data Quality Findings

- BTC_USD_Yahoo has an extra header-like first row and object-typed numerics.
- cryptocurrency has currency symbols/percent signs and null-prone change fields.
- Date parsing required across all datasets.
- Forecasting-ready dataset can be built from bitcoin_data_2014_to_2025 with engineered features.

## 4. Delivery Phases

### Phase A: Data Foundation
- Standardize schema and date typing
- Correct numeric parsing issues
- Remove full-null columns when required
- Preserve raw data and produce cleaned outputs

Deliverables:
- Cleaned dataframes
- Data quality report

### Phase B: EDA and Feature Engineering
- Trend, volatility, and volume analysis
- Rolling metrics and lag features
- Time decomposition features (weekday/month/year)

Deliverables:
- EDA notebook or scripts
- Feature-enhanced data output

### Phase C: Baseline Forecasting Models
- Train ARIMA, SARIMA, Prophet
- Train baseline LSTM on close-price sequences
- Evaluate RMSE/MAE and compare

Deliverables:
- Model metric table
- Forecast preview dataset

### Phase D: Sentiment Integration
- Simulate or ingest sentiment data (news/social)
- Engineer sentiment aggregates and lags
- Retrain LSTM with multivariate features

Deliverables:
- Sentiment-augmented model metrics
- Improvement assessment

### Phase E: Dashboard and Storytelling
- KPI overview
- Price explorer and candlestick view
- Forecast with uncertainty
- Volatility and risk visuals
- Model performance comparison

Deliverables:
- Streamlit dashboard
- Dashboard guide

### Phase F: Optional Deployment
- Streamlit cloud, container, or internal server deployment
- Monitoring and retraining strategy

Deliverables:
- Deployment notes
- Runtime checklist

## 5. Acceptance Criteria

- Reproducible pipeline runs from raw data to metrics
- Forecast model comparison available in exported outputs
- Dashboard runs locally and reads generated outputs
- Power BI archives acknowledged and cataloged as references/deliverables

## 6. Risk and Mitigation

- Non-stationarity: rolling retrain schedule and model refresh
- Sentiment availability: fallback to simulated placeholder
- Overfitting in deep learning: validation splits and early stopping
- Data quality drift: schema checks during each run

## 7. Immediate Next Steps

1. Run `src/train.py` to generate output artifacts.
2. Launch `src/dashboard.py` to validate visual flow.
3. Replace simulated sentiment with live source integration.
4. Add walk-forward validation and hyperparameter tuning.
