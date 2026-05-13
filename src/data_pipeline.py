import numpy as np
import pandas as pd


def load_inputs(bitcoin_csv, btc_yahoo_csv, crypto_market_csv, prophet_csv):
    return {
        "bitcoin": pd.read_csv(bitcoin_csv),
        "btc_yahoo": pd.read_csv(btc_yahoo_csv),
        "crypto_market": pd.read_csv(crypto_market_csv),
        "prophet_prepared": pd.read_csv(prophet_csv),
    }


def clean_bitcoin(df):
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").drop_duplicates()
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["Close"]).reset_index(drop=True)


def clean_btc_yahoo(df):
    out = df.copy()
    if str(out.iloc[0].get("Date", "")).lower() in ["nan", "date", ""]:
        out = out.iloc[1:].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates()
    return out.reset_index(drop=True)


def clean_crypto_market(df):
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    money_cols = ["price_usd", "vol_24h", "market_cap"]
    pct_cols = ["total_vol", "chg_24h", "chg_7d"]

    for col in money_cols:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace("+", "", regex=False)
            )
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].astype(str).str.replace("%", "", regex=False)
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["chg_24h", "chg_7d"]:
        if col in out.columns and out[col].isna().all():
            out = out.drop(columns=[col])

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates()
    return out.reset_index(drop=True)


def build_bitcoin_features(df):
    out = df.copy().sort_values("Date").reset_index(drop=True)
    out["dayofweek"] = out["Date"].dt.dayofweek
    out["month"] = out["Date"].dt.month
    out["year"] = out["Date"].dt.year

    out["Close_lag1"] = out["Close"].shift(1)
    out["MA_7"] = out["Close"].rolling(7).mean()
    out["MA_30"] = out["Close"].rolling(30).mean()

    out["Daily_Return"] = out["Close"].pct_change()
    out["Volatility_7D"] = out["Daily_Return"].rolling(7).std()
    return out


def simulate_sentiment(date_series, seed=42):
    rng = np.random.default_rng(seed)
    n = len(date_series)

    positive = rng.uniform(0.25, 0.9, n)
    negative = rng.uniform(0.05, 0.55, n)
    neutral = rng.uniform(0.02, 0.35, n)
    raw_composite = positive - negative

    # Normalize composite sentiment roughly to [-1, 1].
    comp_min = raw_composite.min()
    comp_max = raw_composite.max()
    composite = ((raw_composite - comp_min) / (comp_max - comp_min)) * 2 - 1

    volume = rng.integers(50, 2000, n)

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(date_series),
            "sentiment_positive": positive,
            "sentiment_negative": negative,
            "sentiment_neutral": neutral,
            "sentiment_composite": composite,
            "sentiment_volume": volume,
        }
    )
    return out.sort_values("Date").reset_index(drop=True)


def merge_sentiment(feature_df, sentiment_df):
    out = feature_df.merge(sentiment_df, on="Date", how="left")
    # Forward fill is acceptable for conceptual sentiment placeholders.
    sent_cols = [
        "sentiment_positive",
        "sentiment_negative",
        "sentiment_neutral",
        "sentiment_composite",
        "sentiment_volume",
    ]
    out[sent_cols] = out[sent_cols].ffill().bfill()
    return out
