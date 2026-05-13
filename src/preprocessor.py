from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
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
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    out = out.drop_duplicates(subset=["Date", "Ticker"])
    out = out.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    before_nulls = int(out.isna().sum().sum())
    out = out.dropna(subset=["Date"])
    out[["Open", "High", "Low", "Close", "Volume"]] = (
        out.groupby("Ticker")[["Open", "High", "Low", "Close", "Volume"]].ffill().bfill()
    )
    out = out.dropna(subset=["Close"]).reset_index(drop=True)
    after_nulls = int(out.isna().sum().sum())
    print(f"  [INFO] Null count before={before_nulls}, after={after_nulls}")

    out["Daily_Return"] = out.groupby("Ticker")["Close"].pct_change()
    return out


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    def _one_ticker(g: pd.DataFrame) -> pd.DataFrame:
        x = g.sort_values("Date").copy()
        x["Ticker"] = g.name
        x["MA_7"] = x["Close"].rolling(7, min_periods=1).mean()
        x["MA_30"] = x["Close"].rolling(30, min_periods=1).mean()
        x["MA_90"] = x["Close"].rolling(90, min_periods=1).mean()
        x["MA_200"] = x["Close"].rolling(200, min_periods=1).mean()

        x["BB_mid"] = x["Close"].rolling(20, min_periods=1).mean()
        bb_std = x["Close"].rolling(20, min_periods=1).std().fillna(0)
        x["BB_upper"] = x["BB_mid"] + 2 * bb_std
        x["BB_lower"] = x["BB_mid"] - 2 * bb_std

        x["RSI"] = _rsi(x["Close"], 14)

        ema12 = x["Close"].ewm(span=12, adjust=False).mean()
        ema26 = x["Close"].ewm(span=26, adjust=False).mean()
        x["MACD"] = ema12 - ema26
        x["MACD_signal"] = x["MACD"].ewm(span=9, adjust=False).mean()
        x["MACD_hist"] = x["MACD"] - x["MACD_signal"]

        x["Lag_1"] = x["Close"].shift(1)
        x["Lag_7"] = x["Close"].shift(7)
        x["Lag_30"] = x["Close"].shift(30)
        x["Momentum_14"] = x["Close"] / x["Close"].shift(14) - 1

        x["Vol_7D"] = x["Daily_Return"].rolling(7, min_periods=1).std()
        x["Vol_30D"] = x["Daily_Return"].rolling(30, min_periods=1).std()
        x["Market_Cap"] = x["Close"] * x["Volume"] * 0.01
        return x

    out = df.groupby("Ticker", group_keys=False).apply(_one_ticker).reset_index(drop=True)
    return out


def build_sentiment_features(date_series: pd.Series, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    n = len(date_series)

    walk = np.cumsum(np.random.normal(0, 0.03, size=n))
    min_w, max_w = walk.min(), walk.max()
    sentiment_raw = (walk - min_w) / (max_w - min_w + 1e-12)
    sentiment_7ma = pd.Series(sentiment_raw).rolling(7, min_periods=1).mean().values

    news_count = np.random.poisson(lam=8, size=n)
    fear_greed = np.clip(sentiment_raw * 100, 0, 100).astype(int)
    pos_share = np.clip(0.2 + sentiment_raw * 0.7, 0.1, 0.9)
    neg_share = np.clip(0.65 - sentiment_raw * 0.6, 0.05, 0.6)
    neu_share = np.clip(1.0 - pos_share - neg_share, 0.05, 0.5)

    return pd.DataFrame(
        {
            "Date": pd.to_datetime(date_series),
            "Sentiment_Raw": sentiment_raw,
            "Sentiment_7MA": sentiment_7ma,
            "News_Count": news_count,
            "Fear_Greed": fear_greed,
            "Pos_Share": pos_share,
            "Neg_Share": neg_share,
            "Neu_Share": neu_share,
        }
    )
