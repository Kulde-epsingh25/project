import pandas as pd
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
except Exception as exc:
    raise SystemExit(
        "Missing dashboard dependencies. Install with: pip install streamlit plotly\n"
        f"Details: {exc}"
    )

from config import (
    OUTPUT_BACKTEST,
    OUTPUT_FEATURES,
    OUTPUT_FORECAST,
    OUTPUT_METRICS,
    OUTPUT_RECOMMENDATION,
    OUTPUT_WITH_SENTIMENT,
)

st.set_page_config(page_title="Crypto Forecast Dashboard", layout="wide")
st.title("BTC Time Series Intelligence Hub")

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at 15% 20%, #13224d 0%, #090c1d 40%, #05060f 100%);
        }
        .kpi-card {
            background: linear-gradient(140deg, rgba(21, 34, 76, 0.95), rgba(8, 10, 30, 0.95));
            border: 1px solid rgba(98, 124, 255, 0.45);
            border-radius: 10px;
            padding: 10px;
        }
        .panel-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #d5dbff;
            margin-bottom: 0.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToAdd": [
        "zoom2d",
        "pan2d",
        "select2d",
        "lasso2d",
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "resetScale2d",
        "toImage",
    ],
}


def show_chart(fig, key):
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=key)


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

if not OUTPUT_FEATURES.exists() or not OUTPUT_METRICS.exists():
    st.warning("Run 'python src/train.py' first to generate outputs.")
    st.stop()

features = pd.read_csv(OUTPUT_FEATURES, parse_dates=["Date"])
metrics = pd.read_csv(OUTPUT_METRICS)
features = features.sort_values("Date").reset_index(drop=True)

st.sidebar.header("Explorer Controls")
date_min = features["Date"].min().date()
date_max = features["Date"].max().date()
date_window = st.sidebar.date_input(
    "Date range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)
if isinstance(date_window, tuple) and len(date_window) == 2:
    d0, d1 = pd.to_datetime(date_window[0]), pd.to_datetime(date_window[1])
    features_view = features[(features["Date"] >= d0) & (features["Date"] <= d1)].copy()
else:
    features_view = features.copy()

ret_col = pick_col(features_view, ["Daily_Return", "returns", "Return"])
vol7_col = pick_col(features_view, ["Volatility_7D", "Vol_7D", "volatility_7"])

show_markers = st.sidebar.toggle("Show markers on line charts", value=False)
show_rangeslider = st.sidebar.toggle("Show range slider", value=True)

st.sidebar.caption("All charts support zoom-in, zoom-out, pan, reset scale, selection, and image export from the chart toolbar.")

st.download_button(
    "Download filtered feature data",
    data=features_view.to_csv(index=False).encode("utf-8"),
    file_name="filtered_btc_features.csv",
    mime="text/csv",
)

if not metrics.dropna(subset=["rmse"]).empty:
    best = metrics.dropna(subset=["rmse"]).sort_values("rmse").iloc[0]
    mape_val = best["mape"] if "mape" in metrics.columns else (best["smape"] if "smape" in metrics.columns else float("nan"))
    dir_val = best["directional_accuracy"] if "directional_accuracy" in metrics.columns else float("nan")
    mae_val = best["mae"] if "mae" in metrics.columns else float("nan")
    st.success(
        f"Best holdout model: {best['model']} | RMSE={best['rmse']:.2f} | MAE={mae_val:.2f} | "
        f"MAPE/SMAPE={mape_val:.2f}% | Directional Accuracy={dir_val:.2f}%"
    )

top_a, top_b, top_c, top_d = st.columns(4)
with top_a:
    st.metric("Latest Close", f"${features_view['Close'].iloc[-1]:,.2f}")
with top_b:
    st.metric("Max Close", f"${features_view['Close'].max():,.2f}")
with top_c:
    if ret_col is not None:
        st.metric("Avg Daily Return", f"{features_view[ret_col].dropna().mean() * 100:.2f}%")
    else:
        st.metric("Avg Daily Return", "N/A")
with top_d:
    if vol7_col is not None:
        st.metric("Avg Volatility 7D", f"{features_view[vol7_col].dropna().mean() * 100:.2f}%")
    else:
        st.metric("Avg Volatility 7D", "N/A")

metric_plot = metrics.dropna(subset=["rmse"]).copy()
if metric_plot.empty:
    st.warning("No valid model metrics found.")

tab1, tab2, tab3 = st.tabs([
    "Price Explorer & Candlesticks",
    "Sentiment & News Impact",
    "Interactive Explorer",
])

with tab1:
    left, right = st.columns([3, 2])
    with left:
        st.markdown("<div class='panel-title'>OHLC Candlestick</div>", unsafe_allow_html=True)
        fig_candle = go.Figure(
            data=[
                go.Candlestick(
                    x=features_view["Date"],
                    open=features_view["Open"],
                    high=features_view["High"],
                    low=features_view["Low"],
                    close=features_view["Close"],
                    increasing_line_color="#28e091",
                    decreasing_line_color="#ff5b7f",
                )
            ]
        )
        fig_candle.update_layout(
            template="plotly_dark",
            height=450,
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis_rangeslider_visible=show_rangeslider,
            dragmode="zoom",
        )
        show_chart(fig_candle, key="candle")

    with right:
        st.markdown("<div class='panel-title'>Model Error Comparison</div>", unsafe_allow_html=True)
        fig_metrics = px.bar(
            metric_plot,
            x="model",
            y="rmse",
            color="rmse",
            template="plotly_dark",
            color_continuous_scale="tealgrn",
            height=220,
        )
        show_chart(fig_metrics, key="metrics")

        st.markdown("<div class='panel-title'>Volume Trend</div>", unsafe_allow_html=True)
        fig_volume = px.bar(
            features.tail(180),
            x="Date",
            y="Volume",
            template="plotly_dark",
            height=220,
        )
        if show_rangeslider:
            fig_volume.update_xaxes(rangeslider=dict(visible=True))
        show_chart(fig_volume, key="volume")

    row2_left, row2_right = st.columns(2)
    with row2_left:
        fig_close = px.line(
            features_view,
            x="Date",
            y=["Close", "MA_7", "MA_30"],
            template="plotly_dark",
            title="Price with Moving Averages",
            markers=show_markers,
        )
        if show_rangeslider:
            fig_close.update_xaxes(rangeslider=dict(visible=True))
        show_chart(fig_close, key="ma")
    with row2_right:
        if ret_col is not None and vol7_col is not None:
            fig_scatter = px.scatter(
                features_view.dropna(subset=[ret_col, vol7_col]),
                x=ret_col,
                y=vol7_col,
                template="plotly_dark",
                title="Risk Map: Return vs Volatility",
                opacity=0.7,
            )
            show_chart(fig_scatter, key="riskmap")
        else:
            st.info("Risk map unavailable: return/volatility columns not found in current dataset.")

if OUTPUT_WITH_SENTIMENT.exists():
    with_sentiment = pd.read_csv(OUTPUT_WITH_SENTIMENT, parse_dates=["Date"])
else:
    with_sentiment = None

with tab2:
    if with_sentiment is None:
        st.info("No sentiment file found. Run training first.")
    else:
        sent = with_sentiment.copy()
        sent_col_composite = pick_col(sent, ["sentiment_composite", "Sentiment_Raw", "Sentiment_7MA"])
        sent_col_pos = pick_col(sent, ["sentiment_positive", "Pos_Share"])
        sent_col_neg = pick_col(sent, ["sentiment_negative", "Neg_Share"])
        sent_col_neu = pick_col(sent, ["sentiment_neutral", "Neu_Share"])
        sent_col_volume = pick_col(sent, ["sentiment_volume", "News_Count"])

        sent = sent[(sent["Date"] >= features_view["Date"].min()) & (sent["Date"] <= features_view["Date"].max())].copy()

        s1, s2 = st.columns([3, 2])
        with s1:
            y_cols = [c for c in [sent_col_composite, sent_col_pos, sent_col_neg] if c is not None]
            if y_cols:
                fig_sent = px.line(
                    sent,
                    x="Date",
                    y=y_cols,
                    template="plotly_dark",
                    title="Sentiment and Market Mood Trend",
                    markers=show_markers,
                )
                if show_rangeslider:
                    fig_sent.update_xaxes(rangeslider=dict(visible=True))
                show_chart(fig_sent, key="senttrend")
            else:
                st.info("Sentiment trend unavailable: sentiment columns not found.")

        with s2:
            if sent_col_pos is not None and sent_col_neg is not None and sent_col_neu is not None:
                denom = sent[sent_col_pos].mean() + sent[sent_col_neg].mean() + sent[sent_col_neu].mean()
                pos_share = (sent[sent_col_pos].mean() / denom) * 100 if denom != 0 else float("nan")
                st.metric("Positive Sentiment Share", f"{pos_share:.2f}%")
            else:
                st.metric("Positive Sentiment Share", "N/A")

            if sent_col_composite is not None:
                st.metric("Avg Composite", f"{sent[sent_col_composite].mean():.3f}")
            else:
                st.metric("Avg Composite", "N/A")

            if sent_col_volume is not None:
                st.metric("Avg Mention Volume", f"{sent[sent_col_volume].mean():.0f}")
            else:
                st.metric("Avg Mention Volume", "N/A")

        s3, s4 = st.columns(2)
        with s3:
            if sent_col_composite is not None:
                fig_hist = px.histogram(
                    sent,
                    x=sent_col_composite,
                    nbins=40,
                    template="plotly_dark",
                    title="Composite Sentiment Distribution",
                )
                show_chart(fig_hist, key="senthist")
            else:
                st.info("Sentiment histogram unavailable: composite sentiment column not found.")

        with s4:
            if sent_col_composite is not None and sent_col_volume is not None:
                fig_impact = px.scatter(
                    sent,
                    x=sent_col_composite,
                    y="Close",
                    size=sent_col_volume,
                    template="plotly_dark",
                    title="Sentiment vs Price Impact",
                    opacity=0.65,
                )
                show_chart(fig_impact, key="sentimpact")
            else:
                st.info("Sentiment impact chart unavailable: required columns not found.")

if OUTPUT_FORECAST.exists():
    forecast = pd.read_csv(OUTPUT_FORECAST, parse_dates=["Date"])
else:
    forecast = None

with tab3:
    if forecast is None:
        st.info("No forecast output found. Run training first.")
    else:
        model_options = [col for col in forecast.columns if col not in ["Date", "actual"]]
        c1, c2 = st.columns([2, 1])
        with c1:
            selected_models = st.multiselect(
                "Forecast model overlay",
                model_options,
                default=model_options[:3] if len(model_options) >= 3 else model_options,
            )
        with c2:
            max_rows = st.slider("Rows to display", min_value=60, max_value=len(forecast), value=min(240, len(forecast)))

        compare_mode = st.radio("Compare view", ["Lines", "Normalized to 100", "Error vs Actual"], horizontal=True)

        view_df = forecast.tail(max_rows).copy()
        if compare_mode == "Normalized to 100":
            for c in ["actual"] + selected_models:
                base = view_df[c].iloc[0] if view_df[c].iloc[0] != 0 else 1.0
                view_df[c] = view_df[c] / base * 100.0
        elif compare_mode == "Error vs Actual":
            for c in selected_models:
                view_df[c] = view_df[c] - view_df["actual"]
            view_df["actual"] = 0.0

        plot_cols = ["actual"] + selected_models
        fig_forecast = px.line(
            view_df,
            x="Date",
            y=plot_cols,
            template="plotly_dark",
            title="Forecast vs Actual (Interactive)",
            markers=show_markers,
        )
        if show_rangeslider:
            fig_forecast.update_xaxes(rangeslider=dict(visible=True))
        show_chart(fig_forecast, key="forecast")

        st.markdown("<div class='panel-title'>Model Performance Table</div>", unsafe_allow_html=True)
        st.dataframe(metrics, width="stretch")

if OUTPUT_BACKTEST.exists():
    backtest = pd.read_csv(OUTPUT_BACKTEST)
    st.subheader("Rolling Backtest")
    st.dataframe(backtest, width="stretch")
    if "rmse" in backtest.columns and "model" in backtest.columns and not backtest.dropna(subset=["rmse"]).empty:
        fig_bt = px.box(
            backtest,
            x="model",
            y="rmse",
            points="all",
            title="Rolling Backtest RMSE Distribution",
            template="plotly_dark",
        )
        show_chart(fig_bt, key="backtest")
    elif "equity" in backtest.columns and "model" in backtest.columns:
        fig_eq = px.line(
            backtest,
            x="step" if "step" in backtest.columns else backtest.index,
            y="equity",
            color="model",
            template="plotly_dark",
            title="Backtest Equity Curves",
            markers=show_markers,
        )
        show_chart(fig_eq, key="backtest_equity")
    elif "pct_gain_loss" in backtest.columns and "model" in backtest.columns:
        fig_ret = px.bar(
            backtest,
            x="model",
            y="pct_gain_loss",
            template="plotly_dark",
            title="Backtest Return by Model",
            color="pct_gain_loss",
        )
        show_chart(fig_ret, key="backtest_return")
    else:
        st.info("Backtest chart unavailable: expected columns like rmse/equity/pct_gain_loss were not found.")

if OUTPUT_RECOMMENDATION.exists():
    st.subheader("Recommendation")
    st.code(OUTPUT_RECOMMENDATION.read_text(encoding="utf-8"))
