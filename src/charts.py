from __future__ import annotations

import base64
import io
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BG = "#050a1a"
PANEL = "#101d3f"
TEXT = "#e2e8f0"
DIM = "#94a3b8"
CYAN = "#00d4ff"
PURPLE = "#7c3aed"
AMBER = "#f59e0b"
GREEN = "#10b981"
RED = "#ef4444"
PINK = "#ec4899"


def _style_axes(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=DIM)
    for s in ax.spines.values():
        s.set_color(CYAN)
        s.set_alpha(0.25)
    ax.grid(True, alpha=0.15, color=DIM)


def fig_to_b64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _make_figure(rows, cols, size=(18, 10)):
    fig, axes = plt.subplots(rows, cols, figsize=size)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(hspace=0.28, wspace=0.25)
    if isinstance(axes, np.ndarray):
        for ax in axes.ravel():
            _style_axes(ax)
    else:
        _style_axes(axes)
    return fig, axes


def page1_overview(btc_df, metrics_df):
    fig, axes = _make_figure(2, 2)
    ax1, ax2, ax3, ax4 = axes.ravel()

    x = pd.to_datetime(btc_df["Date"])
    ax1.plot(x, btc_df["Close"], color=CYAN, lw=1.7, label="Close")
    ax1.plot(x, btc_df["MA_30"], color=AMBER, lw=1.2, ls="--", label="MA30")
    ax1.plot(x, btc_df["MA_200"], color=PINK, lw=1.2, ls=":", label="MA200")
    above = btc_df["Close"] >= btc_df["MA_30"]
    ax1.fill_between(x, btc_df["Close"], btc_df["MA_30"], where=above, color=GREEN, alpha=0.12)
    ax1.fill_between(x, btc_df["Close"], btc_df["MA_30"], where=~above, color=RED, alpha=0.12)
    ax1.set_title("BTC Price with MA30/MA200", color=TEXT)
    ax1.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT)

    annual = btc_df.copy()
    annual["Year"] = pd.to_datetime(annual["Date"]).dt.year
    annual_ret = annual.groupby("Year")["Close"].apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    ax2.bar(annual_ret.index.astype(str), annual_ret.values * 100, color=[GREEN if v >= 0 else RED for v in annual_ret.values])
    ax2.set_title("Annual Returns (%)", color=TEXT)
    ax2.tick_params(axis="x", rotation=45)

    tailv = btc_df.tail(365)
    ax3.bar(pd.to_datetime(tailv["Date"]), tailv["Volume"], color=PURPLE, alpha=0.65)
    ax3.set_title("Volume (Last 365 Days)", color=TEXT)

    ax4.axis("off")
    tbl = metrics_df[["model", "rmse", "mae", "smape"]].copy().round(3)
    cell_text = tbl.values.tolist()
    t = ax4.table(cellText=cell_text, colLabels=tbl.columns.tolist(), loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.35)
    for (_, _), cell in t.get_celld().items():
        cell.set_facecolor(PANEL)
        cell.set_edgecolor(DIM)
        cell.get_text().set_color(TEXT)
    ax4.set_title("Model Performance", color=TEXT)

    fig.suptitle("Page 1 - Executive Overview", color=TEXT, fontsize=15)
    return fig_to_b64(fig)


def page2_price_explorer(full_df, btc_df):
    fig, axes = _make_figure(3, 1, size=(18, 12))
    ax1, ax2, ax3 = axes.ravel()

    cdf = btc_df.tail(180).reset_index(drop=True)
    idx = np.arange(len(cdf))
    up = cdf["Close"] >= cdf["Open"]
    dn = ~up
    ax1.vlines(idx, cdf["Low"], cdf["High"], color=np.where(up, GREEN, RED), linewidth=1)
    ax1.bar(idx[up], (cdf["Close"] - cdf["Open"])[up], bottom=cdf["Open"][up], color=GREEN, width=0.75)
    ax1.bar(idx[dn], (cdf["Close"] - cdf["Open"])[dn], bottom=cdf["Open"][dn], color=RED, width=0.75)
    ax1.set_title("BTC Candlestick (Last 180 Days)", color=TEXT)

    table_df = cdf[["Date", "Open", "High", "Low", "Close", "Volume", "Daily_Return", "Vol_7D"]].tail(8).copy()
    table_df["Date"] = pd.to_datetime(table_df["Date"]).dt.strftime("%Y-%m-%d")
    table_df = table_df.round(3)
    ax2.axis("off")
    t = ax2.table(cellText=table_df.values.tolist(), colLabels=table_df.columns.tolist(), loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.2)
    for (_, _), cell in t.get_celld().items():
        cell.set_facecolor(PANEL)
        cell.set_edgecolor(DIM)
        cell.get_text().set_color(TEXT)

    pivot = full_df.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last").sort_index().ffill().dropna(how="all")
    norm = pivot / pivot.iloc[0] * 100
    for col in norm.columns:
        ax3.plot(pd.to_datetime(norm.index), norm[col], lw=1.25, label=col)
    ax3.set_yscale("log")
    ax3.set_title("Normalized Multi-Crypto Index (log scale)", color=TEXT)
    ax3.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT, ncol=6)

    fig.suptitle("Page 2 - Price Explorer & Candlesticks", color=TEXT, fontsize=15)
    return fig_to_b64(fig)


def page3_forecasting(test_df, forecast_map, metrics_df):
    fig, axes = _make_figure(2, 2)
    ax1, ax2, ax3, ax4 = axes.ravel()

    x = pd.to_datetime(test_df["Date"])
    actual = np.asarray(test_df["Close"], dtype=float)
    ax1.plot(x, actual, color=DIM, lw=2, label="Actual")
    color_map = {"ARIMA": AMBER, "SARIMA": CYAN, "Prophet-Like": PURPLE, "LSTM": GREEN}
    for name, pred in forecast_map.items():
        p = np.asarray(pred, dtype=float)
        c = color_map.get(name, PINK)
        ax1.plot(x, p, lw=1.35, ls="--", color=c, label=name)
        unc = np.std(actual - p)
        ax1.fill_between(x, p - 1.96 * unc, p + 1.96 * unc, color=c, alpha=0.08)
    ax1.set_title("Forecast vs Actual with 95% Bands", color=TEXT)
    ax1.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT)

    md = metrics_df[["model", "rmse", "mae"]].copy().set_index("model")
    ind = np.arange(len(md))
    ax2.bar(ind - 0.2, md["rmse"], 0.4, label="RMSE", color=CYAN)
    ax2.bar(ind + 0.2, md["mae"], 0.4, label="MAE", color=AMBER)
    ax2.set_xticks(ind)
    ax2.set_xticklabels(md.index, rotation=25)
    ax2.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT)
    ax2.set_title("RMSE / MAE by Model", color=TEXT)

    model_names = list(forecast_map.keys())[:2]
    for a, name in zip([ax3, ax4], model_names):
        resid = actual - np.asarray(forecast_map[name], dtype=float)
        a.axhline(0, color=DIM, lw=1)
        a.plot(x, resid, color=PURPLE, lw=1.2)
        a.fill_between(x, 0, resid, where=resid >= 0, color=GREEN, alpha=0.15)
        a.fill_between(x, 0, resid, where=resid < 0, color=RED, alpha=0.15)
        a.set_title(f"Residuals - {name}", color=TEXT)

    fig.suptitle("Page 3 - Forecasting & Uncertainty", color=TEXT, fontsize=15)
    return fig_to_b64(fig)


def page4_sentiment(btc_sent):
    fig, axes = _make_figure(2, 2)
    ax1, ax2, ax3, ax4 = axes.ravel()

    x = pd.to_datetime(btc_sent["Date"])
    ax1.plot(x, btc_sent["Close"], color=CYAN, lw=1.5, label="Close")
    ax1_t = ax1.twinx()
    ax1_t.plot(x, btc_sent["Sentiment_7MA"], color=AMBER, lw=1.1, label="Sentiment_7MA")
    ax1_t.set_ylim(0, 1)
    ax1_t.tick_params(colors=DIM)
    ax1.set_title("Price vs Sentiment", color=TEXT)

    last = btc_sent.iloc[-1]
    ax2.pie(
        [last["Pos_Share"], last["Neg_Share"], last["Neu_Share"]],
        labels=["Pos", "Neg", "Neu"],
        colors=[GREEN, RED, DIM],
        autopct="%1.1f%%",
        textprops={"color": TEXT},
    )
    ax2.set_title("Sentiment Mix (Latest)", color=TEXT)

    ax3.bar(x, btc_sent["News_Count"], color=PURPLE, alpha=0.7)
    ax3.plot(x, btc_sent["Fear_Greed"], color=CYAN, lw=1.2)
    ax3.set_title("News Count + Fear/Greed", color=TEXT)

    nxt_ret = btc_sent["Close"].pct_change().shift(-1) * 100
    sc = ax4.scatter(btc_sent["Sentiment_7MA"], nxt_ret, c=btc_sent["Fear_Greed"], cmap="RdYlGn", alpha=0.65)
    ax4.set_title("Sentiment vs Next-Day Return", color=TEXT)
    cb = fig.colorbar(sc, ax=ax4)
    cb.ax.tick_params(colors=DIM)

    fig.suptitle("Page 4 - Sentiment & News Impact", color=TEXT, fontsize=15)
    return fig_to_b64(fig)


def page5_risk(btc_df):
    fig, axes = _make_figure(2, 2)
    ax1, ax2, ax3, ax4 = axes.ravel()

    ret = btc_df["Daily_Return"].fillna(0)
    vol7 = ret.rolling(7).std() * 100
    vol30 = ret.rolling(30).std() * 100
    vol90 = ret.rolling(90).std() * 100
    x = pd.to_datetime(btc_df["Date"])

    ax1.plot(x, vol7, color=CYAN, label="Vol 7D")
    ax1.plot(x, vol30, color=AMBER, label="Vol 30D")
    ax1.plot(x, vol90, color=PURPLE, label="Vol 90D")
    ax1.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT)
    ax1.set_title("Rolling Volatility (%)", color=TEXT)

    var95 = np.percentile(ret, 5)
    var99 = np.percentile(ret, 1)
    cvar95 = ret[ret <= var95].mean()
    sharpe = ret.mean() / (ret.std() + 1e-12) * np.sqrt(252)
    dd = btc_df["Close"] / btc_df["Close"].cummax() - 1

    ax2.axis("off")
    lines = [
        f"VaR 95%: {var95:.3%}",
        f"VaR 99%: {var99:.3%}",
        f"CVaR 95%: {cvar95:.3%}",
        f"Sharpe: {sharpe:.3f}",
        f"Max Drawdown: {dd.min():.3%}",
    ]
    ax2.text(0.02, 0.95, "\n".join(lines), va="top", color=TEXT, fontsize=11)
    ax2.set_title("Risk Panel", color=TEXT)

    ax3.fill_between(x, dd * 100, 0, color=RED, alpha=0.35)
    ax3.plot(x, dd * 100, color=RED, lw=1)
    ax3.set_title("Drawdown (%)", color=TEXT)

    ax4.hist(ret * 100, bins=40, color=CYAN, alpha=0.65)
    ax4.axvline(var95 * 100, color=AMBER, ls="--")
    ax4.axvline(var99 * 100, color=RED, ls="--")
    ax4.set_title("Return Distribution + VaR", color=TEXT)

    fig.suptitle("Page 5 - Volatility & Risk", color=TEXT, fontsize=15)
    return fig_to_b64(fig)


def page6_indicators(btc_df):
    d = btc_df.tail(365).copy()
    fig, axes = _make_figure(4, 1, size=(18, 14))
    ax1, ax2, ax3, ax4 = axes.ravel()
    x = pd.to_datetime(d["Date"])

    ax1.plot(x, d["Close"], color=CYAN, lw=1.5, label="Close")
    ax1.plot(x, d["BB_upper"], color=AMBER, lw=1, ls="--", label="BB_upper")
    ax1.plot(x, d["BB_mid"], color=DIM, lw=1, ls="-", label="BB_mid")
    ax1.plot(x, d["BB_lower"], color=AMBER, lw=1, ls="--", label="BB_lower")
    ax1.fill_between(x, d["BB_lower"], d["BB_upper"], color=AMBER, alpha=0.10)
    ax1.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT)
    ax1.set_title("Bollinger Bands", color=TEXT)

    ax2.plot(x, d["MA_7"], color=GREEN, label="MA7")
    ax2.plot(x, d["MA_30"], color=AMBER, label="MA30")
    ax2.plot(x, d["MA_90"], color=PURPLE, label="MA90")
    cross_up = d["MA_7"] > d["MA_30"]
    ax2.fill_between(x, d["MA_7"], d["MA_30"], where=cross_up, color=GREEN, alpha=0.12)
    ax2.fill_between(x, d["MA_7"], d["MA_30"], where=~cross_up, color=RED, alpha=0.12)
    ax2.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT)
    ax2.set_title("Moving Averages", color=TEXT)

    ax3.plot(x, d["RSI"], color=CYAN)
    ax3.axhline(70, color=RED, ls="--")
    ax3.axhline(30, color=GREEN, ls="--")
    ax3.fill_between(x, 70, 100, color=RED, alpha=0.10)
    ax3.fill_between(x, 0, 30, color=GREEN, alpha=0.10)
    ax3.set_ylim(0, 100)
    ax3.set_title("RSI(14)", color=TEXT)

    ax4.bar(x, d["MACD_hist"], color=[GREEN if v >= 0 else RED for v in d["MACD_hist"]], alpha=0.6)
    ax4.plot(x, d["MACD"], color=CYAN, label="MACD")
    ax4.plot(x, d["MACD_signal"], color=AMBER, label="Signal")
    ax4.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT)
    ax4.set_title("MACD", color=TEXT)

    fig.suptitle("Page 6 - Technical Indicators", color=TEXT, fontsize=15)
    return fig_to_b64(fig)


def page7_market_structure(full_df):
    fig, axes = _make_figure(2, 2)
    ax1, ax2, ax3, ax4 = axes.ravel()

    piv = full_df.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last").sort_index().ffill().dropna(how="all")
    corr = piv.pct_change().corr()
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax1, annot_kws={"color": TEXT})
    ax1.set_title("Return Correlation Heatmap", color=TEXT)

    latest = full_df.sort_values("Date").groupby("Ticker").tail(1)
    cap = (latest["Close"] * latest["Volume"] * 0.01).sort_values()
    ax2.barh(cap.index, cap.values, color=PURPLE, alpha=0.75)
    ax2.set_title("Market Cap Proxy", color=TEXT)

    ret = piv.pct_change()
    if "BTC" in ret.columns:
        for col in [c for c in ret.columns if c != "BTC"]:
            rc = ret["BTC"].rolling(90).corr(ret[col])
            ax3.plot(pd.to_datetime(rc.index), rc.values, lw=1.1, label=col)
        ax3.axhline(0, color=DIM, lw=1)
        ax3.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT)
    ax3.set_title("Rolling 90D Correlation vs BTC", color=TEXT)

    if "BTC" in ret.columns and "ETH" in ret.columns:
        xy = ret[["BTC", "ETH"]].dropna().tail(1000)
        colors = np.linspace(0, 1, len(xy))
        ax4.scatter(xy["BTC"], xy["ETH"], c=colors, cmap="plasma", alpha=0.55)
    ax4.set_title("BTC vs ETH Daily Returns", color=TEXT)

    fig.suptitle("Page 7 - Correlations & Market Structure", color=TEXT, fontsize=15)
    return fig_to_b64(fig)


def page8_explainability(fi_df, metrics_df, btc_df):
    fig, axes = _make_figure(2, 2)
    ax1, ax2, ax3, ax4 = axes.ravel()

    top = fi_df.head(12).copy()
    colors = [GREEN if i < 5 else PURPLE for i in range(len(top))]
    ax1.barh(top["feature"], top["importance"], color=colors)
    ax1.invert_yaxis()
    ax1.set_title("Feature Importance", color=TEXT)

    s = fi_df.sort_values("importance", ascending=False)
    csum = s["importance"].cumsum()
    ax2.plot(np.arange(len(csum)), csum.values, color=CYAN)
    ax2.axhline(0.8, color=AMBER, ls="--")
    ax2.set_title("Cumulative Importance", color=TEXT)

    ax3.axis("off")
    m = metrics_df[["model", "rmse", "mae"]].round(3)
    t = ax3.table(cellText=m.values.tolist(), colLabels=m.columns.tolist(), loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.35)
    for (_, _), cell in t.get_celld().items():
        cell.set_facecolor(PANEL)
        cell.set_edgecolor(DIM)
        cell.get_text().set_color(TEXT)
    ax3.set_title("Model Contribution Table", color=TEXT)

    z = btc_df[["Lag_1", "Close"]].dropna()
    ax4.scatter(z["Lag_1"], z["Close"], alpha=0.3, color=CYAN)
    if len(z) > 5:
        m1, b1 = np.polyfit(z["Lag_1"], z["Close"], 1)
        xx = np.linspace(z["Lag_1"].min(), z["Lag_1"].max(), 100)
        ax4.plot(xx, m1 * xx + b1, color=AMBER)
        ax4.set_title(f"Lag_1 vs Close (y={m1:.3f}x+{b1:.3f})", color=TEXT)

    fig.suptitle("Page 8 - Feature Importance & Explainability", color=TEXT, fontsize=15)
    return fig_to_b64(fig)


def page9_backtest(curves, summary_df, metrics_df):
    fig, axes = _make_figure(2, 2)
    ax1, ax2, ax3, ax4 = axes.ravel()

    for name, eq in curves.items():
        ax1.plot(eq, lw=1.5, label=name)
    ax1.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT)
    ax1.set_title("Equity Curves", color=TEXT)

    ax2.axis("off")
    tdf = summary_df[["model", "final_portfolio", "pct_gain_loss"]].copy().round(2)
    t = ax2.table(cellText=tdf.values.tolist(), colLabels=tdf.columns.tolist(), loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.35)
    for (_, _), cell in t.get_celld().items():
        cell.set_facecolor(PANEL)
        cell.set_edgecolor(DIM)
        cell.get_text().set_color(TEXT)
    ax2.set_title("Final Returns Table", color=TEXT)

    ax3.bar(summary_df["model"], summary_df["pct_gain_loss"], color=[GREEN if v >= 0 else RED for v in summary_df["pct_gain_loss"]])
    ax3.tick_params(axis="x", rotation=25)
    ax3.set_title("Strategy Return (%)", color=TEXT)

    ax4.bar(metrics_df["model"], metrics_df["directional_accuracy"], color=CYAN)
    ax4.axhline(50, color=AMBER, ls="--")
    ax4.tick_params(axis="x", rotation=25)
    ax4.set_title("Directional Accuracy", color=TEXT)

    fig.suptitle("Page 9 - Strategy Backtest & Performance", color=TEXT, fontsize=15)
    return fig_to_b64(fig)


def page10_explorer(test_df, forecast_map, lstm_future_90, btc_df):
    fig, axes = _make_figure(2, 2)
    ax1, ax2, ax3, ax4 = axes.ravel()

    x = pd.to_datetime(test_df["Date"])
    ax1.plot(x, test_df["Close"], color=DIM, lw=2, label="Actual")
    for name, p in forecast_map.items():
        ax1.plot(x, p, lw=1.2, ls="--", label=name)
    ax1.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT)
    ax1.set_title("All Model Overlays", color=TEXT)

    ax2.axis("off")
    lines = []
    for name, p in forecast_map.items():
        lines.append(f"{name}: ${float(np.asarray(p)[-1]):,.2f}")
    ax2.text(0.02, 0.95, "\n".join(lines), va="top", color=TEXT, fontsize=11)
    ax2.set_title("Latest Model KPI", color=TEXT)

    hist_x = pd.to_datetime(btc_df["Date"]).tail(240)
    hist_y = btc_df["Close"].tail(240)
    ax3.plot(hist_x, hist_y, color=CYAN, label="History")
    if lstm_future_90 is not None and len(lstm_future_90) > 0:
        last_date = pd.to_datetime(btc_df["Date"]).iloc[-1]
        fut_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=len(lstm_future_90), freq="D")
        f = np.asarray(lstm_future_90, dtype=float)
        sd = np.std(np.diff(hist_y)) if len(hist_y) > 3 else np.std(f) * 0.1
        ax3.plot(fut_dates, f, color=GREEN, label="LSTM 90D")
        ax3.fill_between(fut_dates, f - sd, f + sd, color=GREEN, alpha=0.15)
    ax3.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT)
    ax3.set_title("LSTM 90-Day Forward Forecast", color=TEXT)

    ax4.axis("off")
    feats = [c for c in btc_df.columns if c not in ["Date", "Ticker"]]
    info = [
        f"Date range: {pd.to_datetime(btc_df['Date']).min().date()} to {pd.to_datetime(btc_df['Date']).max().date()}",
        f"Rows: {len(btc_df):,}",
        f"Models: {', '.join(forecast_map.keys())}",
        f"Feature count: {len(feats)}",
        "Feature names:",
        ", ".join(feats[:20]),
    ]
    ax4.text(0.02, 0.95, "\n".join(info), va="top", color=TEXT, fontsize=9)
    ax4.set_title("Config Panel", color=TEXT)

    fig.suptitle("Page 10 - Interactive Explorer", color=TEXT, fontsize=15)
    return fig_to_b64(fig)


def generate_all(full_df, btc_df, test_df, forecast_map, metrics_df, btc_sent_df, fi_df, curves, summary_df, lstm_future_90):
    pages: Dict[str, str] = {}
    pages["Page 1 - Executive Overview"] = page1_overview(btc_df, metrics_df)
    pages["Page 2 - Price Explorer"] = page2_price_explorer(full_df, btc_df)
    pages["Page 3 - Forecasting"] = page3_forecasting(test_df, forecast_map, metrics_df)
    pages["Page 4 - Sentiment"] = page4_sentiment(btc_sent_df)
    pages["Page 5 - Risk"] = page5_risk(btc_df)
    pages["Page 6 - Indicators"] = page6_indicators(btc_df)
    pages["Page 7 - Market Structure"] = page7_market_structure(full_df)
    pages["Page 8 - Explainability"] = page8_explainability(fi_df, metrics_df, btc_df)
    pages["Page 9 - Backtest"] = page9_backtest(curves, summary_df, metrics_df)
    pages["Page 10 - Explorer"] = page10_explorer(test_df, forecast_map, lstm_future_90, btc_df)
    return pages
