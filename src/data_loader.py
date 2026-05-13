"""
Data loading and ingestion layer.
Provides a resilient multi-source data acquisition pipeline with automatic
fallback to local datasets if APIs are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import BTC_YAHOO_CSV, CRYPTO_MARKET_CSV, PROCESSED_FULL_PARQUET, PROJECT_ROOT, RAW_DIR, RAW_OHLCV_CSV

TICKER_MAP: Dict[str, str] = {
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
    "BNB-USD": "BNB",
    "SOL-USD": "SOL",
    "ADA-USD": "ADA",
    "XRP-USD": "XRP",
}

CG_IDS: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "SOL": "solana",
    "ADA": "cardano",
    "XRP": "ripple",
}

REQ_COLUMNS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
TARGET_TICKERS = {"BTC", "ETH", "BNB", "SOL", "ADA", "XRP"}


@dataclass
class LoadResult:
    frame: pd.DataFrame
    source: str


def _standardize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = df.copy()
    out = out[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    out["Ticker"] = ticker
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Date", "Close"])
    return out[REQ_COLUMNS].sort_values("Date").reset_index(drop=True)


def _download_from_yfinance(start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
    except Exception:
        return None

    frames: List[pd.DataFrame] = []
    for y_ticker, ticker in TICKER_MAP.items():
        raw = yf.download(y_ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            continue
        raw = raw.reset_index()
        raw = raw.rename(columns={"Adj Close": "Close"})
        if not {"Date", "Open", "High", "Low", "Close", "Volume"}.issubset(raw.columns):
            continue
        frames.append(_standardize(raw, ticker))

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _download_from_stooq(start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        import pandas_datareader.data as web
    except Exception:
        return None

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    frames: List[pd.DataFrame] = []

    for y_ticker, ticker in TICKER_MAP.items():
        try:
            raw = web.DataReader(y_ticker, "stooq", start_dt, end_dt)
        except Exception:
            continue
        if raw is None or raw.empty:
            continue
        raw = raw.reset_index().rename(columns={"Date": "Date"})
        if not {"Date", "Open", "High", "Low", "Close", "Volume"}.issubset(raw.columns):
            continue
        # Stooq can return descending dates.
        raw = raw.sort_values("Date")
        frames.append(_standardize(raw, ticker))

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _download_from_coingecko(days: int = 3650) -> Optional[pd.DataFrame]:
    try:
        import requests
    except Exception:
        return None

    frames: List[pd.DataFrame] = []
    for ticker, cg_id in CG_IDS.items():
        ohlc_url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/ohlc"
        vol_url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"

        ohlc_resp = requests.get(ohlc_url, params={"vs_currency": "usd", "days": days}, timeout=30)
        vol_resp = requests.get(vol_url, params={"vs_currency": "usd", "days": days, "interval": "daily"}, timeout=30)
        if ohlc_resp.status_code != 200 or vol_resp.status_code != 200:
            continue

        ohlc_data = ohlc_resp.json()
        vol_data = vol_resp.json()
        if not isinstance(ohlc_data, list) or not ohlc_data:
            continue

        ohlc = pd.DataFrame(ohlc_data, columns=["ts", "Open", "High", "Low", "Close"])
        ohlc["Date"] = pd.to_datetime(ohlc["ts"], unit="ms", errors="coerce").dt.floor("D")

        vols = pd.DataFrame(vol_data.get("total_volumes", []), columns=["ts", "Volume"])
        vols["Date"] = pd.to_datetime(vols["ts"], unit="ms", errors="coerce").dt.floor("D")

        merged = ohlc.merge(vols[["Date", "Volume"]], on="Date", how="left")
        merged["Volume"] = pd.to_numeric(merged["Volume"], errors="coerce").fillna(0.0)
        frames.append(_standardize(merged, ticker))

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _csv_fallback() -> pd.DataFrame:
    # Fallback keeps the project runnable when networks/APIs are blocked.
    btc = pd.read_csv(BTC_YAHOO_CSV)
    if str(btc.iloc[0].get("Date", "")).strip().lower() == "date":
        btc = btc.iloc[1:].copy()
    btc["Date"] = pd.to_datetime(btc["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        btc[c] = pd.to_numeric(btc[c], errors="coerce")
    btc = btc.dropna(subset=["Date", "Close"])
    btc = _standardize(btc, "BTC")

    market = pd.read_csv(CRYPTO_MARKET_CSV)
    if "timestamp" in market.columns and "name" in market.columns and "price_usd" in market.columns:
        market["timestamp"] = pd.to_datetime(market["timestamp"], errors="coerce")
        market["price_usd"] = pd.to_numeric(
            market["price_usd"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce",
        )
        market["vol_24h"] = pd.to_numeric(
            market.get("vol_24h", 0).astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce",
        )
        name_to_ticker = {
            "Bitcoin": "BTC",
            "Ethereum": "ETH",
            "Binance Coin": "BNB",
            "Solana": "SOL",
            "Cardano": "ADA",
            "Ripple": "XRP",
        }
        market["Ticker"] = market["name"].map(name_to_ticker)
        market = market.dropna(subset=["Ticker", "timestamp", "price_usd"]).copy()
        market = market.sort_values(["Ticker", "timestamp"]).drop_duplicates(["Ticker", "timestamp"])
        market["Date"] = market["timestamp"].dt.floor("D")
        market["Open"] = market["price_usd"]
        market["High"] = market["price_usd"]
        market["Low"] = market["price_usd"]
        market["Close"] = market["price_usd"]
        market["Volume"] = market["vol_24h"].fillna(0.0)
        alt = market[REQ_COLUMNS]
        out = pd.concat([btc, alt], ignore_index=True)
    else:
        out = btc

    return out.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def _cached_fallback() -> Optional[pd.DataFrame]:
    frames: List[pd.DataFrame] = []

    if RAW_OHLCV_CSV.exists():
        try:
            raw = pd.read_csv(RAW_OHLCV_CSV)
            if set(REQ_COLUMNS).issubset(raw.columns):
                frames.append(raw[REQ_COLUMNS].copy())
        except Exception:
            pass

    if PROCESSED_FULL_PARQUET.exists():
        try:
            p = pd.read_parquet(PROCESSED_FULL_PARQUET)
            if set(REQ_COLUMNS).issubset(p.columns):
                frames.append(p[REQ_COLUMNS].copy())
        except Exception:
            pass

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Date", "Ticker", "Close"])
    out = out.drop_duplicates(subset=["Date", "Ticker"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return out


def _ticker_from_text(text: str) -> Optional[str]:
    s = (text or "").upper()
    s = s.replace("-", "").replace("_", "").replace("/", "").replace(" ", "")
    for t in TARGET_TICKERS:
        if s == t or s.startswith(t):
            return t
    return None


def _read_table(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
    except Exception:
        return None
    return None


def _find_col(cols_lower: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols_lower:
            return cols_lower[c]
    return None


def _coerce_ohlcv_from_any(df: pd.DataFrame, file_hint: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    out = df.copy()
    cols_lower = {c.lower().strip(): c for c in out.columns}

    date_col = _find_col(cols_lower, ["date", "timestamp", "time", "datetime", "open_time"])
    close_col = _find_col(cols_lower, ["close", "close_price", "price", "price_usd"])
    open_col = _find_col(cols_lower, ["open", "open_price"])
    high_col = _find_col(cols_lower, ["high", "high_price"])
    low_col = _find_col(cols_lower, ["low", "low_price"])
    vol_col = _find_col(cols_lower, ["volume", "vol", "quote_volume", "base_volume", "vol_24h", "amount"])
    ticker_col = _find_col(cols_lower, ["ticker", "symbol", "pair", "asset", "coin", "name"])

    if date_col is None or close_col is None:
        return None

    work = pd.DataFrame()
    work["Date"] = pd.to_datetime(out[date_col], errors="coerce")
    work["Close"] = pd.to_numeric(out[close_col], errors="coerce")

    if open_col is not None:
        work["Open"] = pd.to_numeric(out[open_col], errors="coerce")
    else:
        work["Open"] = work["Close"]

    if high_col is not None:
        work["High"] = pd.to_numeric(out[high_col], errors="coerce")
    else:
        work["High"] = work[["Open", "Close"]].max(axis=1)

    if low_col is not None:
        work["Low"] = pd.to_numeric(out[low_col], errors="coerce")
    else:
        work["Low"] = work[["Open", "Close"]].min(axis=1)

    if vol_col is not None:
        work["Volume"] = pd.to_numeric(out[vol_col], errors="coerce").fillna(0.0)
    else:
        work["Volume"] = 0.0

    if ticker_col is not None:
        work["Ticker"] = out[ticker_col].astype(str).map(_ticker_from_text)
    else:
        guessed = _ticker_from_text(file_hint)
        work["Ticker"] = guessed

    work = work.dropna(subset=["Date", "Close", "Ticker"]).copy()
    work = work[work["Ticker"].isin(TARGET_TICKERS)]
    if work.empty:
        return None

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=["Open", "High", "Low", "Close"])
    if work.empty:
        return None

    return work[REQ_COLUMNS].sort_values(["Ticker", "Date"]).reset_index(drop=True)


def _local_exchange_fallback() -> Optional[pd.DataFrame]:
    roots = [
        RAW_DIR,
        RAW_DIR.parent,
        PROJECT_ROOT.parent / "data",
    ]
    files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(list(root.rglob("*.csv")))
        files.extend(list(root.rglob("*.parquet")))

    # Keep runtime bounded and avoid huge artifacts.
    selected: List[Path] = []
    for p in files:
        name = p.name.lower()
        if "crypto_ohlcv_full" in name:
            continue
        if any(x in name for x in ["btc", "eth", "bnb", "sol", "ada", "xrp", "ohlcv", "usdt", "usd", "market"]):
            try:
                if p.stat().st_size <= 50 * 1024 * 1024:
                    selected.append(p)
            except Exception:
                continue
    selected = sorted(set(selected), key=lambda x: x.name)[:25]

    frames: List[pd.DataFrame] = []
    for p in selected:
        raw = _read_table(p)
        coerced = _coerce_ohlcv_from_any(raw, p.stem)
        if coerced is not None and not coerced.empty:
            frames.append(coerced)

    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["Date", "Ticker"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return out


def _score_coverage(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    counts = df.groupby("Ticker").size()
    return int(counts.sum() + counts.min() * 100)


def _merge_prefer_deeper(primary: pd.DataFrame, secondary: Optional[pd.DataFrame]) -> pd.DataFrame:
    if secondary is None or secondary.empty:
        return primary
    a = primary.copy()
    b = secondary.copy()
    a["Date"] = pd.to_datetime(a["Date"], errors="coerce")
    b["Date"] = pd.to_datetime(b["Date"], errors="coerce")
    out = pd.concat([a, b], ignore_index=True)
    out = out.dropna(subset=["Date", "Ticker", "Close"])
    out = out.drop_duplicates(subset=["Date", "Ticker"], keep="last")
    return out.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def download_all(start: str = "2015-01-01", end: str = "2025-01-01") -> LoadResult:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    cached = _cached_fallback()

    yfin = _download_from_yfinance(start=start, end=end)
    if yfin is not None and not yfin.empty:
        out = _merge_prefer_deeper(yfin, cached)
        source = "yfinance+cache" if cached is not None else "yfinance"
    else:
        cg = _download_from_coingecko(days=3650)
        if cg is not None and not cg.empty:
            out = _merge_prefer_deeper(cg, cached)
            source = "coingecko+cache" if cached is not None else "coingecko"
        else:
            stq = _download_from_stooq(start=start, end=end)
            if stq is not None and not stq.empty:
                out = _merge_prefer_deeper(stq, cached)
                source = "stooq+cache" if cached is not None else "stooq"
            else:
                local_raw = _local_exchange_fallback()
                csv_fb = _csv_fallback()
                if local_raw is not None and _score_coverage(local_raw) > _score_coverage(csv_fb):
                    csv_fb = _merge_prefer_deeper(local_raw, csv_fb)
                # Keep the richer offline dataset when available.
                if _score_coverage(cached) > _score_coverage(csv_fb):
                    out = _merge_prefer_deeper(cached, csv_fb)
                    source = "cached_fallback"
                else:
                    out = _merge_prefer_deeper(csv_fb, cached)
                    if local_raw is not None:
                        source = "local_raw+csv_fallback+cache" if cached is not None else "local_raw+csv_fallback"
                    else:
                        source = "csv_fallback+cache" if cached is not None else "csv_fallback"

    out = out.drop_duplicates(subset=["Date", "Ticker"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    out.to_csv(RAW_OHLCV_CSV, index=False)
    return LoadResult(frame=out, source=source)
