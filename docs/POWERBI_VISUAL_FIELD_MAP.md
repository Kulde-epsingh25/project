# Power BI Visual Field Map (Drag-and-Drop Checklist)

Use this as a build sheet while creating visuals. Every item lists where each field should go in Power BI.

## Global Setup

## Date Table Fields
- `Date[Date]`
- `Date[Year]`
- `Date[MonthNo]`
- `Date[MonthName]`
- `Date[Quarter]`
- `Date[Weekday]`

## Global Slicers (all pages)
- Date range slicer:
  - Field: `Date[Date]`
- Regime slicer:
  - Field: `regime_timeline[regime]`
- Model slicer:
  - Field: `model_metrics[model]`
- Strategy slicer:
  - Field: `strategy_performance_summary[strategy]`

---

## Page 1: Executive Overview

## Visual 1: KPI Card - Latest Close
- Visual: Card
- Field: measure `Latest Close`

## Visual 2: KPI Card - Best Model
- Visual: Card
- Field: measure `Best Model`

## Visual 3: KPI Card - Best RMSE
- Visual: Card
- Field: measure `Best RMSE`

## Visual 4: KPI Card - Current Regime
- Visual: Card
- Field: measure `Current Regime`

## Visual 5: Line - BTC Close Trend
- Visual: Line chart
- X-axis: `Date[Date]`
- Y-axis: `master_btc_daily[Close]` (Average)
- Tooltip: `master_btc_daily[Volume]`, `regime_timeline[regime]`

## Visual 6: Area - Volatility Trend
- Visual: Area chart
- X-axis: `Date[Date]`
- Y-axis: `risk_daily[volatility_30]` (Average)

## Visual 7: Donut - Regime Share
- Visual: Donut
- Legend: `regime_timeline[regime]`
- Values: `Regime Count` (measure)

## Visual 8: Table - Model Ranking
- Visual: Table
- Columns: `model_metrics[model]`, `model_metrics[rmse]`, `model_metrics[mae]`, `model_metrics[mape]`, `model_metrics[directional_accuracy]`
- Sort: `model_metrics[rmse]` ascending

---

## Page 2: Price Explorer & Candlesticks

## Visual 1: Candlestick OHLC
- Visual: Candlestick custom visual
- Date: `Date[Date]`
- Open: `master_btc_daily[Open]`
- High: `master_btc_daily[High]`
- Low: `master_btc_daily[Low]`
- Close: `master_btc_daily[Close]`

## Visual 2: Volume Columns
- Visual: Clustered column chart
- X-axis: `Date[Date]`
- Y-axis: `master_btc_daily[Volume]`

## Visual 3: Price + MAs
- Visual: Line chart
- X-axis: `Date[Date]`
- Y-axis: `master_btc_daily[Close]`, `master_btc_daily[sma_20]`, `master_btc_daily[sma_50]`

## Visual 4: Support/Resistance
- Visual: Line chart
- X-axis: `Date[Date]`
- Y-axis: `master_btc_daily[support_30]`, `master_btc_daily[resistance_30]`, `master_btc_daily[Close]`

---

## Page 3: Forecast & Uncertainty

## Visual 1: Actual vs Forecast
- Visual: Line chart
- X-axis: `Date[Date]`
- Y-axis: `forecast_comparison[actual]`, `forecast_comparison[ARIMA]`, `forecast_comparison[SARIMA]`, `forecast_comparison[Prophet]`, `forecast_comparison[LSTM_BASELINE]`, `forecast_comparison[LSTM_WITH_SENTIMENT]`, `forecast_comparison[ENSEMBLE_WEIGHTED]`

## Visual 2: Forecast Error Card
- Visual: Card
- Field: measure `Forecast Error Abs`

## Visual 3: Forecast Error % Card
- Visual: Card
- Field: measure `Forecast Error Pct`

## Visual 4: Model Metrics Matrix
- Visual: Matrix
- Rows: `model_metrics[model]`
- Values: `model_metrics[rmse]`, `model_metrics[mae]`, `model_metrics[mape]`, `model_metrics[smape]`, `model_metrics[directional_accuracy]`

---

## Page 4: Sentiment & News Impact

## Visual 1: Sentiment + Price Trend
- Visual: Line chart (dual-axis if available)
- X-axis: `Date[Date]`
- Y-axis: `sentiment_daily[sentiment_score]`
- Secondary Y-axis: `master_btc_daily[Close]`

## Visual 2: Article Count
- Visual: Column chart
- X-axis: `Date[Date]`
- Y-axis: `sentiment_daily[article_count]`

## Visual 3: Event Impact Scatter
- Visual: Scatter chart
- X-axis: `sentiment_event_impact[sentiment_score]`
- Y-axis: `sentiment_event_impact[post_3d_return]`
- Size: `sentiment_event_impact[article_count]`
- Legend: `sentiment_event_impact[event_tag]`
- Tooltip: `sentiment_event_impact[headline]`, `sentiment_event_impact[pre_3d_return]`

## Visual 4: Events Table
- Visual: Table
- Columns: `sentiment_event_impact[Date]`, `sentiment_event_impact[event_tag]`, `sentiment_event_impact[headline]`, `sentiment_event_impact[sentiment_score]`, `sentiment_event_impact[pre_3d_return]`, `sentiment_event_impact[post_3d_return]`

---

## Page 5: Volatility & Risk

## Visual 1: Rolling Volatility
- Visual: Line chart
- X-axis: `Date[Date]`
- Y-axis: `risk_daily[volatility_7]`, `risk_daily[volatility_30]`

## Visual 2: Drawdown Curve
- Visual: Line chart
- X-axis: `Date[Date]`
- Y-axis: `risk_daily[drawdown]`

## Visual 3: Return Distribution
- Visual: Histogram
- Field: `returns_daily_change[returns]`

## Visual 4: Stress Days Table
- Visual: Table
- Columns: `returns_daily_change[Date]`, `returns_daily_change[returns]`, `returns_daily_change[log_returns]`
- Filter: Top N worst returns (or sort ascending)

---

## Page 6: Indicators Dashboard

## Visual 1: RSI
- Visual: Line chart
- X-axis: `Date[Date]`
- Y-axis: `master_btc_daily[rsi_14]`
- Analytics line: constant 30, 70

## Visual 2: MACD
- Visual: Combo (line + column)
- X-axis: `Date[Date]`
- Column Y-axis: `master_btc_daily[macd_hist]`
- Line Y-axis: `master_btc_daily[macd_line]`, `master_btc_daily[macd_signal]`

## Visual 3: Bollinger Bands
- Visual: Line chart
- X-axis: `Date[Date]`
- Y-axis: `master_btc_daily[Close]`, `master_btc_daily[bb_upper]`, `master_btc_daily[bb_mid]`, `master_btc_daily[bb_lower]`

## Visual 4: MA Crossover Signal
- Visual: Line chart
- X-axis: `Date[Date]`
- Y-axis: `master_btc_daily[sma_20]`, `master_btc_daily[sma_50]`

---

## Page 7: Correlations & Market Structure

## Visual 1: Dominance Bar
- Visual: Bar chart
- X-axis: `market_topcoins_latest[name]`
- Y-axis: `market_topcoins_latest[market_dominance_pct]`

## Visual 2: Market Cap Treemap
- Visual: Treemap
- Group: `market_topcoins_latest[name]`
- Values: `market_topcoins_latest[market_cap]`

## Visual 3: Correlation Heat Matrix
- Visual: Matrix
- Rows: `market_correlation_matrix[coin]`
- Values: correlation columns from matrix table

## Visual 4: Top Coin Table
- Visual: Table
- Columns: `market_topcoins_latest[name]`, `market_topcoins_latest[symbol]`, `market_topcoins_latest[market_cap]`, `market_topcoins_latest[market_dominance_pct]`

---

## Page 8: Feature Importance & Explainability

## Visual 1: Feature Importance Bar
- Visual: Bar chart
- X-axis: `feature_importance[importance]`
- Y-axis: `feature_importance[feature]`
- Sort: descending by importance

## Visual 2: Model Comparison Table
- Visual: Table
- Columns: `model_metrics[model]`, `model_metrics[rmse]`, `model_metrics[mape]`, `model_metrics[directional_accuracy]`

## Visual 3: Regime vs Returns
- Visual: Box plot (or column average)
- Axis: `regime_timeline[regime]`
- Values: `regime_timeline[returns]`

---

## Page 9: Strategy Backtest & Performance

## Visual 1: Equity Curves
- Visual: Line chart
- X-axis: `Date[Date]`
- Y-axis: `strategy_equity_curves[equity_ret_buy_hold]`, `strategy_equity_curves[equity_ret_ma_cross]`, `strategy_equity_curves[equity_ret_rsi]`, `strategy_equity_curves[equity_ret_forecast]`

## Visual 2: Strategy Summary Table
- Visual: Table
- Columns: `strategy_performance_summary[strategy]`, `strategy_performance_summary[total_return]`, `strategy_performance_summary[annualized_return]`, `strategy_performance_summary[max_drawdown]`, `strategy_performance_summary[win_rate]`, `strategy_performance_summary[sharpe_like]`

## Visual 3: Strategy Return Bars
- Visual: Bar chart
- Axis: `strategy_performance_summary[strategy]`
- Values: `strategy_performance_summary[total_return]`

---

## Page 10: Interactive Explorer & Export

## Visual 1: Dynamic Price/Forecast Trend
- Visual: Line chart
- X-axis: `Date[Date]`
- Y-axis: at least `forecast_comparison[actual]` + selected model lines

## Visual 2: Export Table
- Visual: Table
- Columns: `Date[Date]`, selected metrics/forecast values

## Interaction Controls
- Slicers: date, model, regime
- Bookmarks: preset model views

---

## Page 11: Bonus - Seasonality & Monthly Trend

## Visual 1: Monthly Return Matrix
- Visual: Matrix
- Rows: `seasonality_monthly[year]`
- Columns: `seasonality_monthly[month_name]`
- Values: `seasonality_monthly[avg_return]`

## Visual 2: Monthly Volatility Bars
- Visual: Column chart
- Axis: `seasonality_monthly[month_name]`
- Values: `seasonality_monthly[avg_volatility]`

## Visual 3: Monthly Return Trend
- Visual: Line chart
- X-axis: `seasonality_monthly[month_name]`
- Y-axis: `seasonality_monthly[avg_return]`

---

## Page 12: Bonus - Returns & Daily Change Analysis

## Visual 1: Daily Return Histogram
- Visual: Histogram
- Field: `returns_daily_change[returns]`

## Visual 2: Weekday Pattern
- Visual: Bar chart
- Axis: `seasonality_weekday[weekday_name]`
- Values: `seasonality_weekday[avg_return]`

## Visual 3: Log Return Trend
- Visual: Line chart
- X-axis: `Date[Date]`
- Y-axis: `returns_daily_change[log_returns]`

---

## ModelSelector Helper Table (for forecast switching)
Create a disconnected table:

`ModelSelector = DATATABLE("Model", STRING, {{"NAIVE"},{"ARIMA"},{"SARIMA"},{"Prophet"},{"LSTM_BASELINE"},{"LSTM_WITH_SENTIMENT"},{"ENSEMBLE_WEIGHTED"}})`

Use it with the `Selected Model Forecast` measure from the DAX file.
