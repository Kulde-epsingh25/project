import numpy as np
import pandas as pd

from config import (
    BITCOIN_CSV,
    CRYPTO_MARKET_CSV,
    NEWS_SENTIMENT_CSV,
    OUTPUT_FORECAST,
    OUTPUT_METRICS,
    OUTPUT_POWERBI_DIR,
)
from data_pipeline import clean_bitcoin, clean_crypto_market


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def detect_regime(df):
    out = df.copy()
    trend_30 = out["Close"].pct_change(30)
    vol_30 = out["log_returns"].rolling(30).std() * np.sqrt(365)

    out["regime"] = "Range-Bound"
    out.loc[(trend_30 > 0.05) & (vol_30 < vol_30.quantile(0.75)), "regime"] = "Bull-Trend"
    out.loc[(trend_30 < -0.05) & (vol_30 < vol_30.quantile(0.75)), "regime"] = "Bear-Trend"
    out.loc[vol_30 >= vol_30.quantile(0.75), "regime"] = "High-Volatility"
    return out


def add_indicators(df):
    out = df.copy().sort_values("Date").reset_index(drop=True)

    out["returns"] = out["Close"].pct_change()
    out["log_returns"] = np.log(out["Close"]).diff()

    out["volatility_7"] = out["log_returns"].rolling(7).std() * np.sqrt(365)
    out["volatility_30"] = out["log_returns"].rolling(30).std() * np.sqrt(365)

    out["sma_7"] = out["Close"].rolling(7).mean()
    out["sma_20"] = out["Close"].rolling(20).mean()
    out["sma_50"] = out["Close"].rolling(50).mean()
    out["ema_20"] = out["Close"].ewm(span=20, adjust=False).mean()

    out["rsi_14"] = rsi(out["Close"], 14)
    m_line, s_line, m_hist = macd(out["Close"])
    out["macd_line"] = m_line
    out["macd_signal"] = s_line
    out["macd_hist"] = m_hist

    std_20 = out["Close"].rolling(20).std()
    out["bb_mid"] = out["sma_20"]
    out["bb_upper"] = out["sma_20"] + (2 * std_20)
    out["bb_lower"] = out["sma_20"] - (2 * std_20)
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"].replace(0, np.nan)

    out["cummax_close"] = out["Close"].cummax()
    out["drawdown"] = out["Close"] / out["cummax_close"] - 1

    out["lag_close_1"] = out["Close"].shift(1)
    out["lag_close_3"] = out["Close"].shift(3)
    out["lag_close_7"] = out["Close"].shift(7)

    out["volume_ma_7"] = out["Volume"].rolling(7).mean()
    out["volume_ma_30"] = out["Volume"].rolling(30).mean()
    out["volume_trend_ratio"] = out["volume_ma_7"] / out["volume_ma_30"].replace(0, np.nan)

    out["support_30"] = out["Low"].rolling(30).min()
    out["resistance_30"] = out["High"].rolling(30).max()

    out = detect_regime(out)
    return out


def strategy_returns(df, signal_col):
    signal = df[signal_col].fillna(0)
    return signal.shift(1).fillna(0) * df["returns"].fillna(0)


def build_strategy_tables(df, forecast_df):
    out = df[["Date", "Close", "returns", "rsi_14", "sma_20", "sma_50"]].copy()

    out["signal_buy_hold"] = 1
    out["ret_buy_hold"] = strategy_returns(out, "signal_buy_hold")

    out["signal_ma_cross"] = (out["sma_20"] > out["sma_50"]).astype(int)
    out["ret_ma_cross"] = strategy_returns(out, "signal_ma_cross")

    out["signal_rsi"] = 0
    out.loc[out["rsi_14"] < 30, "signal_rsi"] = 1
    out.loc[out["rsi_14"] > 70, "signal_rsi"] = -1
    out["ret_rsi"] = strategy_returns(out, "signal_rsi")

    out["signal_forecast"] = 0
    if forecast_df is not None and not forecast_df.empty:
        fc = forecast_df.copy()
        fc_cols = [c for c in fc.columns if c not in ["Date", "actual"]]
        target_col = "ENSEMBLE_WEIGHTED" if "ENSEMBLE_WEIGHTED" in fc_cols else (fc_cols[0] if fc_cols else None)
        if target_col is not None:
            signal_df = fc[["Date", "actual", target_col]].copy()
            signal_df["signal_forecast"] = (signal_df[target_col] > signal_df["actual"]).astype(int)
            out = out.merge(signal_df[["Date", "signal_forecast"]], on="Date", how="left", suffixes=("", "_fc"))
            out["signal_forecast"] = out["signal_forecast_fc"].fillna(0)
            out = out.drop(columns=["signal_forecast_fc"])

    out["ret_forecast"] = strategy_returns(out, "signal_forecast")

    for col in ["ret_buy_hold", "ret_ma_cross", "ret_rsi", "ret_forecast"]:
        out[f"equity_{col}"] = (1 + out[col].fillna(0)).cumprod()

    return out


def summarize_strategy(name, series):
    s = series.fillna(0)
    eq = (1 + s).cumprod()
    total_return = eq.iloc[-1] - 1
    annual_ret = (1 + total_return) ** (365 / max(len(s), 1)) - 1
    vol = s.std() * np.sqrt(365)
    downside = s[s < 0].std() * np.sqrt(365) if (s < 0).any() else 0.0
    sharpe_like = annual_ret / vol if vol and vol > 0 else np.nan
    max_dd = (eq / eq.cummax() - 1).min()
    win_rate = (s > 0).mean()
    return {
        "strategy": name,
        "total_return": float(total_return),
        "annualized_return": float(annual_ret),
        "annualized_volatility": float(vol),
        "downside_risk": float(downside),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "sharpe_like": float(sharpe_like) if not np.isnan(sharpe_like) else np.nan,
    }


def build_sentiment_tables(master_df):
    s = pd.read_csv(NEWS_SENTIMENT_CSV)
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s = s.dropna(subset=["date"]).rename(columns={"date": "Date"})

    sentiment_daily = (
        s.groupby("Date", as_index=False)
        .agg(
            sentiment_score=("sentiment_score", "mean"),
            article_count=("article_count", "sum"),
            event_count=("event_tag", "count"),
        )
    )

    merged = master_df[["Date", "Close", "returns"]].merge(sentiment_daily, on="Date", how="left")
    merged[["sentiment_score", "article_count", "event_count"]] = merged[["sentiment_score", "article_count", "event_count"]].fillna(0)

    event_impact_rows = []
    idx_by_date = {d: i for i, d in enumerate(master_df["Date"]) }
    for _, row in s.iterrows():
        dt = row["Date"]
        if dt not in idx_by_date:
            continue
        i = idx_by_date[dt]
        pre_i = max(0, i - 3)
        post_i = min(len(master_df) - 1, i + 3)
        pre_return = master_df.iloc[i]["Close"] / master_df.iloc[pre_i]["Close"] - 1
        post_return = master_df.iloc[post_i]["Close"] / master_df.iloc[i]["Close"] - 1
        event_impact_rows.append(
            {
                "Date": dt,
                "headline": row["headline"],
                "event_tag": row["event_tag"],
                "sentiment_score": row["sentiment_score"],
                "article_count": row["article_count"],
                "pre_3d_return": float(pre_return),
                "post_3d_return": float(post_return),
            }
        )

    event_impact = pd.DataFrame(event_impact_rows)
    return merged, event_impact


def build_market_structure_tables():
    market_raw = pd.read_csv(CRYPTO_MARKET_CSV)
    market = clean_crypto_market(market_raw)

    latest_ts = market["timestamp"].max()
    latest = market[market["timestamp"] == latest_ts].copy()
    latest = latest.sort_values("market_cap", ascending=False)
    total_cap = latest["market_cap"].sum()
    latest["market_dominance_pct"] = (latest["market_cap"] / total_cap) * 100
    top_latest = latest.head(20).reset_index(drop=True)

    market["date"] = market["timestamp"].dt.floor("D")
    daily = (
        market.groupby(["date", "name"], as_index=False)
        .agg(price_usd=("price_usd", "mean"), market_cap=("market_cap", "mean"))
    )

    cap_rank = daily.groupby("name", as_index=False).agg(avg_market_cap=("market_cap", "mean"))
    top_names = cap_rank.sort_values(by="avg_market_cap", ascending=False).head(10)["name"].tolist()
    piv = daily[daily["name"].isin(top_names)].pivot(index="date", columns="name", values="price_usd")
    corr = piv.pct_change().corr().reset_index().rename(columns={"name": "coin"})

    return top_latest, corr


def build_explainability(master_df):
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])

    out = master_df.copy()
    out["target_next_return"] = out["returns"].shift(-1)
    feats = [
        "returns",
        "log_returns",
        "volatility_7",
        "volatility_30",
        "sma_7",
        "sma_20",
        "sma_50",
        "rsi_14",
        "macd_line",
        "macd_signal",
        "bb_width",
        "drawdown",
        "volume_trend_ratio",
        "lag_close_1",
        "lag_close_3",
        "lag_close_7",
    ]
    model_df = out[feats + ["target_next_return"]].dropna()
    if model_df.empty:
        return pd.DataFrame(columns=["feature", "importance"])

    X = model_df[feats]
    y = model_df["target_next_return"]
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X, y)

    fi = pd.DataFrame({"feature": feats, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    return fi


def seasonality_tables(master_df):
    out = master_df.copy()
    out["year"] = out["Date"].dt.year
    out["month"] = out["Date"].dt.month
    out["month_name"] = out["Date"].dt.month_name()
    out["weekday_name"] = out["Date"].dt.day_name()

    monthly = out.groupby(["year", "month", "month_name"], as_index=False).agg(
        avg_return=("returns", "mean"),
        avg_volatility=("volatility_30", "mean"),
        avg_volume=("Volume", "mean"),
    )

    weekday = out.groupby(["weekday_name"], as_index=False).agg(
        avg_return=("returns", "mean"),
        avg_volatility=("volatility_30", "mean"),
        observations=("Date", "count"),
    )
    return monthly, weekday


def main():
    OUTPUT_POWERBI_DIR.mkdir(parents=True, exist_ok=True)

    btc_raw = pd.read_csv(BITCOIN_CSV)
    btc = clean_bitcoin(btc_raw)
    master = add_indicators(btc)

    forecast = pd.read_csv(OUTPUT_FORECAST, parse_dates=["Date"]) if OUTPUT_FORECAST.exists() else None
    metrics = pd.read_csv(OUTPUT_METRICS) if OUTPUT_METRICS.exists() else pd.DataFrame()

    sentiment_daily, event_impact = build_sentiment_tables(master)
    top_latest, corr = build_market_structure_tables()
    strategy_curve = build_strategy_tables(master, forecast)
    fi = build_explainability(master)
    monthly, weekday = seasonality_tables(master)

    strategy_summary = pd.DataFrame(
        [
            summarize_strategy("buy_hold", strategy_curve["ret_buy_hold"]),
            summarize_strategy("ma_crossover", strategy_curve["ret_ma_cross"]),
            summarize_strategy("rsi_strategy", strategy_curve["ret_rsi"]),
            summarize_strategy("forecast_guided", strategy_curve["ret_forecast"]),
        ]
    ).sort_values("total_return", ascending=False)

    kpi = pd.DataFrame(
        [
            {
                "latest_close": master["Close"].iloc[-1],
                "latest_regime": master["regime"].iloc[-1],
                "latest_volatility_30": master["volatility_30"].iloc[-1],
                "max_drawdown": master["drawdown"].min(),
                "best_model": metrics.sort_values("rmse").iloc[0]["model"] if not metrics.empty else "N/A",
                "best_model_rmse": metrics.sort_values("rmse").iloc[0]["rmse"] if not metrics.empty else np.nan,
            }
        ]
    )

    risk_daily = master[["Date", "Close", "returns", "volatility_7", "volatility_30", "drawdown"]].copy()
    regime_timeline = master[["Date", "regime", "returns", "volatility_30"]].copy()
    returns_analysis = master[["Date", "returns", "log_returns"]].copy()

    master.to_csv(OUTPUT_POWERBI_DIR / "master_btc_daily.csv", index=False)
    kpi.to_csv(OUTPUT_POWERBI_DIR / "kpi_overview.csv", index=False)
    if forecast is not None:
        forecast.to_csv(OUTPUT_POWERBI_DIR / "forecast_comparison.csv", index=False)
    metrics.to_csv(OUTPUT_POWERBI_DIR / "model_metrics.csv", index=False)
    sentiment_daily.to_csv(OUTPUT_POWERBI_DIR / "sentiment_daily.csv", index=False)
    event_impact.to_csv(OUTPUT_POWERBI_DIR / "sentiment_event_impact.csv", index=False)
    risk_daily.to_csv(OUTPUT_POWERBI_DIR / "risk_daily.csv", index=False)
    regime_timeline.to_csv(OUTPUT_POWERBI_DIR / "regime_timeline.csv", index=False)
    strategy_curve.to_csv(OUTPUT_POWERBI_DIR / "strategy_equity_curves.csv", index=False)
    strategy_summary.to_csv(OUTPUT_POWERBI_DIR / "strategy_performance_summary.csv", index=False)
    top_latest.to_csv(OUTPUT_POWERBI_DIR / "market_topcoins_latest.csv", index=False)
    corr.to_csv(OUTPUT_POWERBI_DIR / "market_correlation_matrix.csv", index=False)
    fi.to_csv(OUTPUT_POWERBI_DIR / "feature_importance.csv", index=False)
    monthly.to_csv(OUTPUT_POWERBI_DIR / "seasonality_monthly.csv", index=False)
    weekday.to_csv(OUTPUT_POWERBI_DIR / "seasonality_weekday.csv", index=False)
    returns_analysis.to_csv(OUTPUT_POWERBI_DIR / "returns_daily_change.csv", index=False)

    print("Power BI export pipeline completed.")
    print(f"Exported files in: {OUTPUT_POWERBI_DIR}")


if __name__ == "__main__":
    main()
