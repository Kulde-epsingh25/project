# 7-Day Implementation Roadmap

## Day 1: Data Foundation
- Validate and clean BTC historical data.
- Clean multi-coin market snapshot.
- Prepare curated news sentiment dataset.
- Produce initial master BTC table.

## Day 2: Feature Engineering
- Add returns, log returns, volatility windows.
- Add moving averages, RSI, MACD, Bollinger Bands.
- Add drawdown, lag features, and volume trend features.
- Validate feature completeness and null handling.

## Day 3: Forecasting Engine
- Run baseline and statistical models: Naive, ARIMA, SARIMA, Prophet.
- Run LSTM and LSTM with sentiment features.
- Build weighted ensemble forecast.
- Export holdout metrics and model ranking.

## Day 4: Validation and Regime Detection
- Perform walk-forward/rolling backtesting.
- Add market regime detection labels.
- Build risk diagnostics and stress window tables.
- Select champion model and rationale.

## Day 5: Strategy Lab
- Implement Buy-and-Hold baseline.
- Implement MA crossover strategy.
- Implement RSI strategy.
- Implement forecast-guided strategy.
- Export equity curves and strategy summary metrics.

## Day 6: Power BI Build
- Build pages 1-6: Overview, Candles, Forecast, Sentiment, Risk, Indicators.
- Add slicers, controls, and interactive comparisons.
- Validate table relationships and DAX measures.

## Day 7: Finalization and Storytelling
- Build pages 7-12: Market structure, Explainability, Backtest, Explorer, Seasonality, Daily Returns.
- Add final styling, bookmarks, and presentation flow.
- Prepare presentation deck and project defense narrative.
- Package final deliverable: Python pipeline + Power BI + presentation.
