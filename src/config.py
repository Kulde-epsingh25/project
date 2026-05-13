from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

BITCOIN_CSV = DATA_DIR / "bitcoin_data_2014_to_2025.csv"
BTC_YAHOO_CSV = DATA_DIR / "BTC_USD_Yahoo.csv"
CRYPTO_MARKET_CSV = DATA_DIR / "cryptocurrency.csv"
PROPHET_PREPARED_CSV = DATA_DIR / "prophet_prepared_data.csv"
NEWS_SENTIMENT_CSV = DATA_DIR / "news_sentiment.csv"

RAW_OHLCV_CSV = RAW_DIR / "crypto_ohlcv_full.csv"
PROCESSED_BTC_PARQUET = PROCESSED_DIR / "btc_featured.parquet"
PROCESSED_FULL_PARQUET = PROCESSED_DIR / "full_featured.parquet"
OUTPUT_HTML = OUTPUT_DIR / "crypto_dashboard.html"

OUTPUT_FEATURES = OUTPUT_DIR / "bitcoin_features.csv"
OUTPUT_WITH_SENTIMENT = OUTPUT_DIR / "bitcoin_with_sentiment.csv"
OUTPUT_METRICS = OUTPUT_DIR / "model_metrics.csv"
OUTPUT_FORECAST = OUTPUT_DIR / "forecast_preview.csv"
OUTPUT_BACKTEST = OUTPUT_DIR / "rolling_backtest.csv"
OUTPUT_RECOMMENDATION = OUTPUT_DIR / "model_recommendation.txt"
OUTPUT_POWERBI_DIR = OUTPUT_DIR / "powerbi"
