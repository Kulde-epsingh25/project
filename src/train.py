"""
Model training and validation module for the CryptoForecast project.
This module handles the core training logic for various time-series models,
including classical statistical methods (ARIMA/SARIMA), modern packages (Prophet),
and deep learning architectures (LSTM).
"""

import warnings

import numpy as np
import pandas as pd

from config import (
    BITCOIN_CSV,
    BTC_YAHOO_CSV,
    CRYPTO_MARKET_CSV,
    OUTPUT_FEATURES,
    OUTPUT_FORECAST,
    OUTPUT_BACKTEST,
    OUTPUT_METRICS,
    OUTPUT_RECOMMENDATION,
    OUTPUT_WITH_SENTIMENT,
    PROPHET_PREPARED_CSV,
)
from data_pipeline import (
    build_bitcoin_features,
    clean_bitcoin,
    clean_btc_yahoo,
    clean_crypto_market,
    load_inputs,
    merge_sentiment,
    simulate_sentiment,
)
from models import (
    create_metrics_df,
    directional_accuracy,
    evaluate,
    inverse_lstm_preds,
    mape,
    naive_forecast,
    prepare_lstm_multivariate,
    prepare_lstm_univariate,
    smape,
    train_arima,
    train_lstm_model,
    train_prophet,
    train_sarima,
    weighted_ensemble,
)

warnings.filterwarnings("ignore")


def score_all(y_true, y_pred):
    """Computes a comprehensive set of error metrics for model comparison."""
    rmse, mae = evaluate(y_true, y_pred)
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


def rolling_backtest(bitcoin_features, horizon=30, folds=3):
    """
    Performs a rolling-origin backtest to validate model stability over time.
    This simulates how the model would have performed if deployed at different points in history.
    """
    out_rows = []
    series = bitcoin_features.set_index("Date")["Close"]
    n = len(series)

    # We select fold endpoints that prioritize recent history while maintaining a sufficient training window.
    fold_endpoints = []
    step = max(horizon, (n // (folds + 4)))
    cursor = n - (folds * horizon)
    for _ in range(folds):
        if cursor + horizon >= n:
            break
        fold_endpoints.append(cursor)
        cursor += step

    if not fold_endpoints:
        return pd.DataFrame(columns=["fold", "model", "rmse", "mae", "mape", "smape", "directional_accuracy"])

    for fold_idx, end_pos in enumerate(fold_endpoints, start=1):
        train = series.iloc[:end_pos]
        test = series.iloc[end_pos : end_pos + horizon]
        
        # Skip folds with insufficient data to prevent biased metrics
        if len(test) < 5 or len(train) < 200:
            continue

        # Naive baseline: The simplest 'human' prediction (price tomorrow == price today)
        naive_pred = naive_forecast(train.iloc[-1], len(test))
        naive_scores = score_all(test.values, naive_pred)
        out_rows.append((fold_idx, "NAIVE", naive_scores["rmse"], naive_scores["mae"], naive_scores["mape"], naive_scores["smape"], naive_scores["directional_accuracy"]))

        # ARIMA: Classical autoregressive integrated moving average
        try:
            arima_fit = train_arima(train)
            arima_pred = arima_fit.predict(start=len(train), end=len(train) + len(test) - 1).values
            sc = score_all(test.values, arima_pred)
            out_rows.append((fold_idx, "ARIMA", sc["rmse"], sc["mae"], sc["mape"], sc["smape"], sc["directional_accuracy"]))
        except Exception:
            out_rows.append((fold_idx, "ARIMA", np.nan, np.nan, np.nan, np.nan, np.nan))

        # SARIMA: Seasonal ARIMA (accounting for potential weekly/monthly cycles)
        try:
            sarima_fit = train_sarima(train)
            sarima_pred = sarima_fit.predict(start=len(train), end=len(train) + len(test) - 1).values
            sc = score_all(test.values, sarima_pred)
            out_rows.append((fold_idx, "SARIMA", sc["rmse"], sc["mae"], sc["mape"], sc["smape"], sc["directional_accuracy"]))
        except Exception:
            out_rows.append((fold_idx, "SARIMA", np.nan, np.nan, np.nan, np.nan, np.nan))

    return pd.DataFrame(
        out_rows,
        columns=["fold", "model", "rmse", "mae", "mape", "smape", "directional_accuracy"],
    )


def write_recommendation(metrics_df, backtest_df):
    valid = metrics_df.dropna(subset=["rmse"]).sort_values("rmse")
    if valid.empty:
        text = "No model produced valid scores in this environment."
        OUTPUT_RECOMMENDATION.write_text(text, encoding="utf-8")
        return

    winner = valid.iloc[0]
    lines = [
        "Model Recommendation",
        "====================",
        f"Champion model: {winner['model']}",
        f"Holdout RMSE: {winner['rmse']:.3f}",
        f"Holdout MAE: {winner['mae']:.3f}",
        f"Holdout MAPE: {winner['mape']:.3f}%",
        f"Holdout SMAPE: {winner['smape']:.3f}%",
        f"Directional Accuracy: {winner['directional_accuracy']:.2f}%",
        "",
    ]

    if not backtest_df.empty:
        bt_summary = (
            backtest_df.groupby("model", as_index=False)[["rmse", "mae", "mape", "smape", "directional_accuracy"]]
            .mean()
            .sort_values("rmse")
        )
        lines.append("Rolling Backtest Average (lower is better except directional accuracy):")
        for _, row in bt_summary.iterrows():
            lines.append(
                f"- {row['model']}: RMSE={row['rmse']:.3f}, MAE={row['mae']:.3f}, MAPE={row['mape']:.3f}%, "
                f"SMAPE={row['smape']:.3f}%, Directional={row['directional_accuracy']:.2f}%"
            )

    OUTPUT_RECOMMENDATION.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Main execution block for model training and evaluation."""
    # Data Ingestion
    frames = load_inputs(BITCOIN_CSV, BTC_YAHOO_CSV, CRYPTO_MARKET_CSV, PROPHET_PREPARED_CSV)

    bitcoin = clean_bitcoin(frames["bitcoin"])
    bitcoin_features = build_bitcoin_features(bitcoin)
    bitcoin_features.to_csv(OUTPUT_FEATURES, index=False)

    # Sentiment Synthesis
    sentiment = simulate_sentiment(bitcoin_features["Date"])
    with_sentiment = merge_sentiment(bitcoin_features, sentiment)
    with_sentiment.to_csv(OUTPUT_WITH_SENTIMENT, index=False)

    # Train/Test Split (80/20)
    split = int(len(bitcoin_features) * 0.8)
    train_df = bitcoin_features.iloc[:split].copy()
    test_df = bitcoin_features.iloc[split:].copy()

    train_series = train_df.set_index("Date")["Close"]
    test_series = test_df.set_index("Date")["Close"]

    metrics_rows = []
    metric_map = {}
    pred_map = {}
    forecast_preview = pd.DataFrame({"Date": test_df["Date"].values, "actual": test_df["Close"].values})

    # Benchmark: Naive
    naive_pred = naive_forecast(train_series.iloc[-1], len(test_series))
    naive_sc = score_all(test_series.values, naive_pred)
    metrics_rows.append(("NAIVE", naive_sc["rmse"], naive_sc["mae"], naive_sc["mape"], naive_sc["smape"], naive_sc["directional_accuracy"]))
    metric_map["NAIVE"] = naive_sc
    pred_map["NAIVE"] = naive_pred
    forecast_preview["NAIVE"] = naive_pred

    # Statistical Model: ARIMA
    try:
        arima_fit = train_arima(train_series)
        arima_pred = arima_fit.predict(start=len(train_series), end=len(train_series) + len(test_series) - 1)
        arima_sc = score_all(test_series.values, arima_pred.values)
        metrics_rows.append(("ARIMA", arima_sc["rmse"], arima_sc["mae"], arima_sc["mape"], arima_sc["smape"], arima_sc["directional_accuracy"]))
        metric_map["ARIMA"] = arima_sc
        pred_map["ARIMA"] = arima_pred.values
        forecast_preview["ARIMA"] = arima_pred.values
    except Exception:
        metrics_rows.append(("ARIMA", np.nan, np.nan, np.nan, np.nan, np.nan))

    # Statistical Model: SARIMA
    try:
        sarima_fit = train_sarima(train_series)
        sarima_pred = sarima_fit.predict(start=len(train_series), end=len(train_series) + len(test_series) - 1)
        sarima_sc = score_all(test_series.values, sarima_pred.values)
        metrics_rows.append(("SARIMA", sarima_sc["rmse"], sarima_sc["mae"], sarima_sc["mape"], sarima_sc["smape"], sarima_sc["directional_accuracy"]))
        metric_map["SARIMA"] = sarima_sc
        pred_map["SARIMA"] = sarima_pred.values
        forecast_preview["SARIMA"] = sarima_pred.values
    except Exception:
        metrics_rows.append(("SARIMA", np.nan, np.nan, np.nan, np.nan, np.nan))

    # ML Model: Prophet
    try:
        prophet_train = train_df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        prophet_test = test_df[["Date"]].rename(columns={"Date": "ds"})
        prophet_fit = train_prophet(prophet_train)
        prophet_pred = prophet_fit.predict(prophet_test)["yhat"].values
        prophet_sc = score_all(test_series.values, prophet_pred)
        metrics_rows.append(("Prophet", prophet_sc["rmse"], prophet_sc["mae"], prophet_sc["mape"], prophet_sc["smape"], prophet_sc["directional_accuracy"]))
        metric_map["Prophet"] = prophet_sc
        pred_map["Prophet"] = prophet_pred
        forecast_preview["Prophet"] = prophet_pred
    except Exception:
        metrics_rows.append(("Prophet", np.nan, np.nan, np.nan, np.nan, np.nan))

    # Deep Learning: LSTM Baseline
    try:
        X_train_u, X_test_u, y_train_u, y_test_u, scaler_y_u = prepare_lstm_univariate(bitcoin_features)
        lstm_u = train_lstm_model(X_train_u, y_train_u, input_features=1, epochs=5)
        pred_u_scaled = lstm_u.predict(X_test_u, verbose=0).ravel()
        actual_u, pred_u = inverse_lstm_preds(pred_u_scaled, y_test_u, scaler_y_u)
        lstm_u_sc = score_all(actual_u, pred_u)
        metrics_rows.append(("LSTM_BASELINE", lstm_u_sc["rmse"], lstm_u_sc["mae"], lstm_u_sc["mape"], lstm_u_sc["smape"], lstm_u_sc["directional_accuracy"]))
    except Exception:
        metrics_rows.append(("LSTM_BASELINE", np.nan, np.nan, np.nan, np.nan, np.nan))

    # Deep Learning: LSTM with Sentiment Features
    try:
        feature_cols = [
            "Close", "Volume", "dayofweek", "month", "year",
            "Close_lag1", "MA_7", "MA_30", "Daily_Return",
            "Volatility_7D", "sentiment_positive", "sentiment_negative",
            "sentiment_neutral", "sentiment_composite", "sentiment_volume",
        ]
        X_train_m, X_test_m, y_train_m, y_test_m, _, scaler_y_m = prepare_lstm_multivariate(
            with_sentiment,
            feature_cols=feature_cols,
            target_col="Close",
            look_back=60,
            test_ratio=0.2,
        )
        lstm_m = train_lstm_model(X_train_m, y_train_m, input_features=X_train_m.shape[2], epochs=5)
        pred_m_scaled = lstm_m.predict(X_test_m, verbose=0).ravel()
        actual_m, pred_m = inverse_lstm_preds(pred_m_scaled, y_test_m, scaler_y_m)
        lstm_m_sc = score_all(actual_m, pred_m)
        metrics_rows.append(("LSTM_WITH_SENTIMENT", lstm_m_sc["rmse"], lstm_m_sc["mae"], lstm_m_sc["mape"], lstm_m_sc["smape"], lstm_m_sc["directional_accuracy"]))
    except Exception:
        metrics_rows.append(("LSTM_WITH_SENTIMENT", np.nan, np.nan, np.nan, np.nan, np.nan))

    # Hybrid Approach: Weighted Ensemble
    ensemble_pred, ensemble_weights = weighted_ensemble(pred_map, metric_map)
    if ensemble_pred is not None:
        ensemble_sc = score_all(test_series.values, ensemble_pred)
        metrics_rows.append(("ENSEMBLE_WEIGHTED", ensemble_sc["rmse"], ensemble_sc["mae"], ensemble_sc["mape"], ensemble_sc["smape"], ensemble_sc["directional_accuracy"]))
        forecast_preview["ENSEMBLE_WEIGHTED"] = ensemble_pred

    metrics_df = create_metrics_df(metrics_rows).sort_values("rmse", na_position="last")
    metrics_df.to_csv(OUTPUT_METRICS, index=False)

    # Rolling Backtest Validation
    backtest_df = rolling_backtest(bitcoin_features, horizon=30, folds=3)
    backtest_df.to_csv(OUTPUT_BACKTEST, index=False)

    write_recommendation(metrics_df, backtest_df)
    forecast_preview.head(300).to_csv(OUTPUT_FORECAST, index=False)

    print("--- Training Pipeline Completed ---")
    if ensemble_pred is not None:
        print(f"Optimal Ensemble Weights: {ensemble_weights}")
    print("\nModel Leaderboard (by RMSE):")
    print(metrics_df[["model", "rmse", "directional_accuracy"]])


if __name__ == "__main__":
    main()
