from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    den = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / den) * 100.0)


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - (ss_res / ss_tot))


def directional_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return float("nan")
    return float((np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))).mean() * 100.0)


def compute_all(y_true, forecast_map: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, pred in forecast_map.items():
        p = np.asarray(pred, dtype=float)
        rows.append(
            {
                "model": name,
                "rmse": rmse(y_true, p),
                "mae": mae(y_true, p),
                "smape": smape(y_true, p),
                "r2": r2(y_true, p),
                "directional_accuracy": directional_accuracy(y_true, p),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def feature_importance(btc_df: pd.DataFrame) -> pd.DataFrame:
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])

    feats = [
        "MA_7",
        "MA_30",
        "MA_90",
        "MA_200",
        "BB_mid",
        "BB_upper",
        "BB_lower",
        "RSI",
        "MACD",
        "MACD_signal",
        "MACD_hist",
        "Lag_1",
        "Lag_7",
        "Lag_30",
        "Momentum_14",
        "Vol_7D",
        "Vol_30D",
        "Market_Cap",
        "Sentiment_7MA",
        "Fear_Greed",
    ]

    x = btc_df[feats].copy()
    y = btc_df["Close"].shift(-1)
    z = pd.concat([x, y.rename("target")], axis=1).dropna()
    if z.empty:
        return pd.DataFrame(columns=["feature", "importance"])

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(z[feats], z["target"])
    out = pd.DataFrame({"feature": feats, "importance": model.feature_importances_})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def backtest(y_true, forecast_map: Dict[str, np.ndarray], initial_capital=100000.0):
    y = np.asarray(y_true, dtype=float)
    true_ret = pd.Series(y).pct_change().fillna(0.0).values

    curves = {}
    summary_rows = []

    for name, pred in forecast_map.items():
        p = np.asarray(pred, dtype=float)
        signal = np.sign(np.diff(p, prepend=p[0]))
        strat_ret = np.clip(signal * true_ret, -0.20, 0.20)
        eq = initial_capital * np.cumprod(1.0 + strat_ret)
        curves[name] = eq
        summary_rows.append(
            {
                "model": name,
                "final_portfolio": float(eq[-1]),
                "pct_gain_loss": float((eq[-1] / initial_capital - 1) * 100.0),
                "directional_accuracy": directional_accuracy(y, p),
            }
        )

    return curves, pd.DataFrame(summary_rows).sort_values("final_portfolio", ascending=False)
