# Power BI Build Guide: AI-Powered Bitcoin Market Intelligence & Strategy Lab

## 1. Data Import
Import all CSV files from `outputs/powerbi/`:

- `master_btc_daily.csv`
- `kpi_overview.csv`
- `forecast_comparison.csv`
- `model_metrics.csv`
- `sentiment_daily.csv`
- `sentiment_event_impact.csv`
- `risk_daily.csv`
- `regime_timeline.csv`
- `market_topcoins_latest.csv`
- `market_correlation_matrix.csv`
- `feature_importance.csv`
- `strategy_equity_curves.csv`
- `strategy_performance_summary.csv`
- `seasonality_monthly.csv`
- `seasonality_weekday.csv`
- `returns_daily_change.csv`

## 2. Data Model (Relationships)
Create a Date table and relate all date-based tables to it.

### Date Table
Use `Date = CALENDAR(MIN(master_btc_daily[Date]), MAX(master_btc_daily[Date]))`
Add Year, MonthNo, MonthName, Quarter, Weekday.

### Relationships (single direction from Date -> fact)
- `Date[Date]` -> `master_btc_daily[Date]`
- `Date[Date]` -> `forecast_comparison[Date]`
- `Date[Date]` -> `sentiment_daily[Date]`
- `Date[Date]` -> `sentiment_event_impact[Date]`
- `Date[Date]` -> `risk_daily[Date]`
- `Date[Date]` -> `regime_timeline[Date]`
- `Date[Date]` -> `strategy_equity_curves[Date]`
- `Date[Date]` -> `returns_daily_change[Date]`
- `Date[Year]` + `Date[MonthNo]` -> `seasonality_monthly[year]` + `seasonality_monthly[month]` (or bridge by month index)

Disconnected helper tables:
- `model_metrics`
- `market_topcoins_latest`
- `market_correlation_matrix`
- `feature_importance`
- `strategy_performance_summary`
- `kpi_overview`

## 3. Page-by-Page Visual Mapping

## Page 1: Executive Overview
Goal: What happened + what is current state.

Visuals:
- KPI Cards: Latest Close, Best Model, Best RMSE, Latest Regime, Max Drawdown.
- Line chart: `master_btc_daily[Close]` over `Date[Date]`.
- Area chart: `risk_daily[volatility_30]` over `Date[Date]`.
- Donut: regime share from `regime_timeline[regime]` count.
- Table: top 5 rows from `model_metrics` sorted by RMSE.

## Page 2: Price Explorer & Candlesticks
Goal: Detailed market movement.

Visuals:
- Candlestick (custom visual): Open/High/Low/Close from `master_btc_daily` by `Date[Date]`.
- Column chart: `master_btc_daily[Volume]` by Date.
- Line chart: `sma_20`, `sma_50`, `Close` by Date.
- Band lines: `support_30` and `resistance_30`.

## Page 3: Forecast & Uncertainty
Goal: What will happen next.

Visuals:
- Line chart: `forecast_comparison[actual]`, `ARIMA`, `SARIMA`, `Prophet`, `NAIVE`, `ENSEMBLE_WEIGHTED`.
- Error card: absolute error = `ABS(actual - selected model)`.
- Matrix: model-by-metric from `model_metrics`.
- Optional parameter slicer: model selector.

## Page 4: Sentiment & News Impact
Goal: Why did it happen.

Visuals:
- Line chart: `sentiment_daily[sentiment_score]` and `master_btc_daily[Close]` (secondary axis).
- Column chart: `sentiment_daily[article_count]` by Date.
- Scatter: `sentiment_event_impact[sentiment_score]` vs `post_3d_return`, size by `article_count`.
- Event table: Date, headline, event_tag, sentiment_score, pre_3d_return, post_3d_return.

## Page 5: Volatility & Risk
Goal: Risk and stress.

Visuals:
- Line chart: `risk_daily[volatility_7]`, `volatility_30`.
- Drawdown line: `risk_daily[drawdown]`.
- Histogram: `returns_daily_change[returns]`.
- Stress table: worst 20 days by returns.

## Page 6: Indicators Dashboard
Goal: Signal monitoring.

Visuals:
- Line: `rsi_14` with reference lines 30 and 70.
- Combo chart: `macd_line`, `macd_signal`; columns for `macd_hist`.
- Band chart: `bb_upper`, `bb_mid`, `bb_lower`, with `Close`.
- MA crossover signal marker from `sma_20` vs `sma_50`.

## Page 7: Correlations & Market Structure
Goal: BTC in market context.

Visuals:
- Bar chart: `market_topcoins_latest[market_dominance_pct]` by name (top N).
- Treemap: market cap by coin.
- Matrix heatmap: `market_correlation_matrix` values.
- Ranking table: name, symbol, market_cap, dominance.

## Page 8: Feature Importance & Explainability
Goal: Why model chooses what it does.

Visuals:
- Horizontal bar chart: `feature_importance[importance]` by feature.
- Model ranking table from `model_metrics`.
- Decomposition by regime using `regime_timeline` and returns.

## Page 9: Strategy Backtest & Performance
Goal: What analyst should do.

Visuals:
- Line chart: `equity_ret_buy_hold`, `equity_ret_ma_cross`, `equity_ret_rsi`, `equity_ret_forecast`.
- Table: `strategy_performance_summary` (total_return, annualized_return, max_drawdown, win_rate, sharpe_like).
- Bar chart: total return by strategy.

## Page 10: Interactive Explorer & Export
Goal: Analyst self-service controls.

Visuals:
- Dynamic line chart with model/strategy selector.
- Date range slicer.
- Regime slicer.
- Bookmark buttons for quick states.
- Export-friendly table visual.

## Page 11: Bonus - Seasonality & Monthly Trend
Goal: Detect calendar effects.

Visuals:
- Heatmap matrix: `seasonality_monthly[avg_return]` by year x month_name.
- Column chart: monthly avg volatility.
- Line chart: monthly avg return trend.

## Page 12: Bonus - Returns & Daily Change Analysis
Goal: Microstructure and return behavior.

Visuals:
- Histogram/KDE proxy for daily returns.
- Box plot by weekday from `seasonality_weekday`.
- Cumulative return line from `returns_daily_change`.

## 4. Slicers to Add Globally
- Date range (`Date[Date]`)
- Regime (`regime_timeline[regime]`)
- Model (`model_metrics[model]`)
- Strategy (`strategy_performance_summary[strategy]`)
- Coin (`market_topcoins_latest[name]` for market pages)

## 5. Presentation Flow (Storytelling)
- Start at Executive Overview: headline KPI + key insight.
- Move to Price + Forecast pages for what happened/what next.
- Use Sentiment + Risk to explain why.
- Close with Strategy + Explainability for actionability.
