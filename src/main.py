"""
Main entry point for the CryptoForecast pipeline.
This script orchestrates the entire workflow: data acquisition, preprocessing,
model training, evaluation, and report generation.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from charts import generate_all
from config import (
    MODELS_DIR,
    OUTPUT_BACKTEST,
    OUTPUT_DIR,
    OUTPUT_FEATURES,
    OUTPUT_FORECAST,
    OUTPUT_HTML,
    OUTPUT_METRICS,
    OUTPUT_RECOMMENDATION,
    OUTPUT_WITH_SENTIMENT,
    PROCESSED_BTC_PARQUET,
    PROCESSED_DIR,
    PROCESSED_FULL_PARQUET,
)
from data_loader import download_all
from evaluator import backtest, compute_all, feature_importance
from models import forecast_all, train_all
from preprocessor import FEATURE_COLS, add_features, build_sentiment_features, clean
from report import build


def _save_parquet_with_fallback(df: pd.DataFrame, target: Path):
    """Saves a DataFrame to Parquet format, falling back to CSV if Parquet fails."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(target, index=False)
        return str(target)
    except Exception:
        csv_path = target.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return str(csv_path)


def _fmt_money(v):
    """Helper to format currency values."""
    return f"${float(v):,.2f}"


def main():
    """Executes the full forecasting pipeline."""
    t0 = time.time()
    
    # Initialize directory structure
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 1: Data Acquisition...")
    load_result = download_all(start="2015-01-01", end="2025-01-01")
    full_raw = load_result.frame
    print(f"  > Source: {load_result.source} ({len(full_raw):,} total rows)")
    
    cov = full_raw.groupby("Ticker").size().sort_values(ascending=False)
    for t in sorted(full_raw["Ticker"].unique()):
        print(f"  > {t}: {len(full_raw[full_raw['Ticker'] == t]):,} rows")
    
    if not cov.empty and int(cov.min()) < 365:
        print(f"  [!] Note: Minimum ticker history is low ({int(cov.min())} rows).")

    print("Phase 2: Feature Engineering & Preprocessing...")
    full_clean = clean(full_raw)
    full_feat = add_features(full_clean)
    missing = [c for c in FEATURE_COLS if c not in full_feat.columns]
    if missing:
        raise RuntimeError(f"Missing engineered columns: {missing}")

    btc = full_feat[full_feat["Ticker"] == "BTC"].copy().sort_values("Date").reset_index(drop=True)
    sentiment = build_sentiment_features(btc["Date"], seed=42)
    btc_sent = btc.merge(sentiment, on="Date", how="left")

    full_out = _save_parquet_with_fallback(full_feat, PROCESSED_FULL_PARQUET)
    btc_out = _save_parquet_with_fallback(btc_sent, PROCESSED_BTC_PARQUET)
    
    btc_sent.to_csv(OUTPUT_WITH_SENTIMENT, index=False)
    btc.to_csv(OUTPUT_FEATURES, index=False)
    
    print(f"  > Feature set complete: {full_feat.shape[1]} columns generated.")

    print("Phase 3: Model Training...")
    split = int(len(btc_sent) * 0.80)
    train_df = btc_sent.iloc[:split].copy()
    test_df = btc_sent.iloc[split:].copy()

    model_map = train_all(train_df)
    for name in model_map:
        print(f"  > Training {name}...")

    preds = forecast_all(model_map, train_df, test_df)
    for k, v in preds.items():
        preds[k] = np.clip(np.asarray(v, dtype=float), 0.01, train_df["Close"].max() * 50)

    print("Phase 4: Performance Evaluation & Backtesting...")
    y_true = np.asarray(test_df["Close"], dtype=float)
    metrics_df = compute_all(y_true, preds)
    metrics_df.to_csv(OUTPUT_METRICS, index=False)

    fi_df = feature_importance(btc_sent)
    curves, bt_summary = backtest(y_true, preds)
    bt_rows = []
    for name, eq in curves.items():
        bt_rows.append(pd.DataFrame({"step": np.arange(len(eq)), "model": name, "equity": eq}))
    
    rolling_backtest = pd.concat(bt_rows, ignore_index=True)
    rolling_backtest.to_csv(OUTPUT_BACKTEST, index=False)

    for _, row in metrics_df.iterrows():
        print(
            f"  > {row['model']:<12} | RMSE: {row['rmse']:.2f} | DirAcc: {row['directional_accuracy']:.2f}%"
        )

    print("Phase 5: Generating Visual Analytics...")
    lstm_future_90 = model_map["LSTM"].predict(np.asarray(btc_sent["Close"], dtype=float), 90)
    charts = generate_all(
        full_df=full_feat,
        btc_df=btc_sent,
        test_df=test_df,
        forecast_map=preds,
        metrics_df=metrics_df,
        btc_sent_df=btc_sent,
        fi_df=fi_df,
        curves=curves,
        summary_df=bt_summary,
        lstm_future_90=lstm_future_90,
    )

    print("Phase 6: Finalizing HTML Report...")
    forecast_preview = test_df[["Date", "Close"]].rename(columns={"Close": "actual"}).copy()
    for name, p in preds.items():
        forecast_preview[name] = p
    forecast_preview.to_csv(OUTPUT_FORECAST, index=False)

    best_rmse_row = metrics_df.sort_values("rmse").iloc[0]
    best_dir_row = metrics_df.sort_values("directional_accuracy", ascending=False).iloc[0]

    stats = {
        "latest_btc": _fmt_money(btc_sent["Close"].iloc[-1]),
        "ath": _fmt_money(btc_sent["Close"].max()),
        "avg_vol_30": _fmt_money(btc_sent["Volume"].tail(30).mean()),
        "ret_10y": f"{(btc_sent['Close'].iloc[-1] / btc_sent['Close'].iloc[0] - 1) * 100:.2f}%",
        "best_rmse": f"{best_rmse_row['model']} ({best_rmse_row['rmse']:.2f})",
        "best_dir": f"{best_dir_row['model']} ({best_dir_row['directional_accuracy']:.2f}%)",
        "assets": int(full_feat["Ticker"].nunique()),
        "rows": f"{len(full_feat):,}",
        "models": ", ".join(metrics_df["model"].tolist()),
        "source": load_result.source,
    }
    html = build(charts, stats)
    OUTPUT_HTML.write_text(html, encoding="utf-8")

    print("\n--- Pipeline Execution Summary ---")
    size_mb = OUTPUT_HTML.stat().st_size / (1024 * 1024)
    rmse_line = " | ".join([f"{r['model']}={r['rmse']:.2f}" for _, r in metrics_df.iterrows()])
    summary = (
        f"Report Location: {OUTPUT_HTML}\n"
        f"Report Size: {size_mb:.2f} MB\n"
        f"Model Performance: {rmse_line}\n"
        f"Champion Model (RMSE): {best_rmse_row['model']} ({best_rmse_row['rmse']:.2f})\n"
        f"Best Trend Predictor: {best_dir_row['model']} ({best_dir_row['directional_accuracy']:.2f}%)\n"
        "Pipeline successful. Open the HTML report to view insights."
    )
    OUTPUT_RECOMMENDATION.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Total Runtime: {time.time() - t0:.1f} seconds")


if __name__ == "__main__":
    main()
