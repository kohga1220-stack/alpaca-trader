import os
import json
import schedule
import time
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()
os.makedirs("portfolio", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("portfolio/report.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
trade = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)
data  = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))

ALL_SYMBOLS = {
    "NVDA":"NVIDIA","TSLA":"Tesla","META":"Meta","AMZN":"Amazon",
    "GOOGL":"Alphabet","MSFT":"Microsoft","AMD":"AMD","NFLX":"Netflix",
    "CRM":"Salesforce","PLTR":"Palantir",
    "JNJ":"J&J","PG":"P&G","KO":"Coca-Cola","WMT":"Walmart",
    "MCD":"McDonald's","PEP":"PepsiCo","ABBV":"AbbVie",
    "XOM":"Exxon","VZ":"Verizon","SPY":"S&P500 ETF",
}
COLORS = {"close":"#4fc3f7","ma20":"#ffb74d","ma50":"#81c784","ma200":"#ce93d8","vol":"#ef5350"}

def get_bars(symbol, days=200):
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=days),
    )
    df = data.get_stock_bars(req).df
    if hasattr(df.index, "levels"):
        df = df.loc[symbol]
    df = df.sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["ma20"]  = df["close"].rolling(20).mean()
    df["ma50"]  = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))
    return df

def plot_chart(symbol, df, trades_df=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    facecolor="#0a0a14")
    fig.suptitle(f"{symbol} — {ALL_SYMBOLS.get(symbol,'')} 移動平均チャート",
                 color="white", fontsize=14, fontweight="bold")

    ax1.set_facecolor("#0d0d1a")
    ax1.plot(df.index, df["close"],  color=COLORS["close"], linewidth=1.5, label="終値")
    ax1.plot(df.index, df["ma20"],   color=COLORS["ma20"],  linewidth=1.2, linestyle="--", label="MA20")
    ax1.plot(df.index, df["ma50"],   color=COLORS["ma50"],  linewidth=1.2, linestyle="--", label="MA50")
    if df["ma200"].notna().sum() > 10:
        ax1.plot(df.index, df["ma200"], color=COLORS["ma200"], linewidth=1.0, linestyle=":", label="MA200")

    if trades_df is not None and not trades_df.empty:
        sym_trades = trades_df[trades_df["symbol"] == symbol]
        buys  = sym_trades[sym_trades["action"] == "BUY"]
        sells = sym_trades[sym_trades["action"] == "SELL"]
        if not buys.empty:
            ax1.scatter(pd.to_datetime(buys["ts"]).dt.tz_localize(None),
                        buys["price"], marker="^", color="#81c784", s=80, zorder=5, label="BUY")
        if not sells.empty:
            ax1.scatter(pd.to_datetime(sells["ts"]).dt.tz_localize(None),
                        sells["price"], marker="v", color="#ef5350", s=80, zorder=5, label="SELL")

    ax1.set_ylabel("価格 ($)", color="white")
    ax1.tick_params(colors="gray")
    ax1.spines[:].set_color("#333")
    ax1.grid(color="#1a1a2e", linewidth=0.5)
    ax1.legend(loc="upper left", facecolor="#0d0d1a", edgecolor="#333",
               labelcolor="white", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

    ax2.set_facecolor("#0d0d1a")
    ax2.plot(df.index, df["rsi"], color=COLORS["vol"], linewidth=1.2)
    ax2.axhline(70, color="#ef5350", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.axhline(30, color="#81c784", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.fill_between(df.index, df["rsi"], 70,
                     where=(df["rsi"] >= 70), alpha=0.2, color="#ef5350")
    ax2.fill_between(df.index, df["rsi"], 30,
                     where=(df["rsi"] <= 30), alpha=0.2, color="#81c784")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", color="white")
    ax2.tick_params(colors="gray")
    ax2.spines[:].set_color("#333")
    ax2.grid(color="#1a1a2e", linewidth=0.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

    plt.tight_layout()
    today = datetime.now().strftime("%Y%m%d")
    path  = f"portfolio/{symbol}_{today}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"グラフ保存: {path}")
    return path

def load_trades():
    path = "portfolio/trades.jsonl"
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except:
                pass
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def save_portfolio_snapshot(account, positions, today):
    snapshot = {
        "date":            today,
        "cash":            float(account.cash),
        "portfolio_value": float(account.portfolio_value),
        "positions": [
            {
                "symbol":          p.symbol,
                "qty":             float(p.qty),
                "avg_entry":       float(p.avg_entry_price),
                "current_price":   float(p.current_price),
                "market_value":    float(p.market_value),
                "unrealized_pl":   float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
            for p in positions
        ]
    }
    path = f"portfolio/snapshot_{today}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    log.info(f"ポートフォリオ保存: {path}")
    return snapshot

def plot_portfolio_history():
    snapshots = []
    for fname in sorted(os.listdir("portfolio")):
        if fname.startswith("snapshot_") and fname.endswith(".json"):
            with open(f"portfolio/{fname}", encoding="utf-8") as f:
                snapshots.append(json.load(f))
    if len(snapshots) < 2:
        log.info("ポートフォリオ履歴が少ないためグラフスキップ")
        return

    dates  = [s["date"] for s in snapshots]
    values = [s["portfolio_value"] for s in snapshots]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0a0a14")
    ax.set_facecolor("#0d0d1a")
    ax.plot(dates, values, color="#4fc3f7", linewidth=2)
    ax.fill_between(dates, values, min(values), alpha=0.15, color="#4fc3f7")
    ax.axhline(100000, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="初期資金 $100,000")
    ax.set_title("ポートフォリオ推移", color="white", fontsize=13, fontweight="bold")
    ax.set_ylabel("総資産 ($)", color="white")
    ax.tick_params(colors="gray", rotation=30)
    ax.spines[:].set_color("#333")
    ax.grid(color="#1a1a2e", linewidth=0.5)
    ax.legend(facecolor="#0d0d1a", edgecolor="#333", labelcolor="white")
    plt.tight_layout()
    path = "portfolio/portfolio_history.png"
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"ポートフォリオ推移グラフ: {path}")

def nightly_report():
    now   = datetime.now(ET)
    today = now.strftime("%Y-%m-%d")
    log.info(f"\n{'='*50}")
    log.info(f" 夜間レポート生成開始 {today}")
    log.info(f"{'='*50}")

    account   = trade.get_account()
    positions = trade.get_all_positions()
    trades_df = load_trades()
    snapshot  = save_portfolio_snapshot(account, positions, today)

    report_lines = [
        f"{'='*50}",
        f" 夜間レポート {today}",
        f"{'='*50}",
        f"現金残高  : ${snapshot['cash']:>12,.2f}",
        f"総資産    : ${snapshot['portfolio_value']:>12,.2f}",
        f"損益      : ${snapshot['portfolio_value'] - 100000:>+12,.2f}",
        f"損益率    : {(snapshot['portfolio_value'] / 100000 - 1) * 100:>+8.2f}%",
        f"\n保有ポジション ({len(snapshot['positions'])}銘柄):",
        f"{'銘柄':<8}{'株数':>6}{'取得価格':>10}{'現在価格':>10}{'評価額':>12}{'含み損益':>12}{'損益率':>8}",
        "-" * 66,
    ]
    for p in sorted(snapshot["positions"], key=lambda x: -x["market_value"]):
        pnl_str = f"{p['unrealized_pl']:>+,.2f}"
        pct_str = f"{p['unrealized_plpc']*100:>+.2f}%"
        report_lines.append(
            f"{p['symbol']:<8}{p['qty']:>6.0f}{p['avg_entry']:>10.2f}"
            f"{p['current_price']:>10.2f}{p['market_value']:>12,.2f}"
            f"{pnl_str:>12}{pct_str:>8}"
        )

    if not trades_df.empty:
        today_trades = trades_df[trades_df["ts"].str.startswith(today)] if "ts" in trades_df.columns else pd.DataFrame()
        if not today_trades.empty:
            report_lines += [f"\n本日の取引 ({len(today_trades)}件):"]
            for _, t in today_trades.iterrows():
                icon = "▲" if t["action"] == "BUY" else "▼"
                report_lines.append(
                    f"  {icon} {t['action']:4} {t['symbol']:6} {t['qty']}株 "
                    f"@ ${t['price']:.2f}  確信度{t['confidence']}%"
                )

    report_text = "\n".join(report_lines)
    print(report_text)

    rpath = f"portfolio/report_{today}.txt"
    with open(rpath, "w", encoding="utf-8") as f:
        f.write(report_text)
    log.info(f"レポート保存: {rpath}")

    chart_targets = list({p["symbol"] for p in snapshot["positions"]} |
                         set(["SPY", "NVDA", "TSLA"]))
    for symbol in chart_targets:
        try:
            df = get_bars(symbol)
            plot_chart(symbol, df, trades_df)
            time.sleep(0.3)
        except Exception as e:
            log.error(f"[{symbol}] グラフ生成失敗: {e}")

    plot_portfolio_history()
    log.info("夜間レポート完了\n")

if __name__ == "__main__":
    log.info("Alpaca Report Scheduler 起動")
    nightly_report()
    schedule.every().day.at("16:30").do(nightly_report)
    while True:
        schedule.run_pending()
        time.sleep(60)
