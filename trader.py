import os
import json
import time
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from prompt_builder import build_analysis_prompt, load_skills
from scheduler import fetch_fundamentals
from llm_client import decide, PROVIDER
from datetime import datetime, timedelta

load_dotenv()

# ── 銘柄リスト ────────────────────────────────
HIGH_RISK = {
    "NVDA": "NVIDIA（AI半導体）",
    "TSLA": "Tesla（EV・AI）",
    "META": "Meta（SNS・AI）",
    "AMZN": "Amazon（EC・クラウド）",
    "GOOGL": "Alphabet（広告・AI）",
    "MSFT": "Microsoft（クラウド・AI）",
    "AMD":  "AMD（AI・GPU）",
    "NFLX": "Netflix（動画配信）",
    "CRM":  "Salesforce（SaaS）",
    "PLTR": "Palantir（AIデータ）",
}
LOW_RISK = {
    "JNJ":  "J&J（医薬品）",
    "PG":   "P&G（消費財）",
    "KO":   "Coca-Cola（飲料）",
    "WMT":  "Walmart（小売）",
    "MCD":  "McDonald's（外食）",
    "PEP":  "PepsiCo（飲料食品）",
    "ABBV": "AbbVie（製薬）",
    "XOM":  "Exxon Mobil（エネルギー）",
    "VZ":   "Verizon（通信）",
    "SPY":  "S&P500 ETF（市場平均）",
}
ALL_SYMBOLS = {**HIGH_RISK, **LOW_RISK}
SKILLS = load_skills()

# ── クライアント初期化 ────────────────────────
trade  = TradingClient(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    paper=True
)
data   = StockHistoricalDataClient(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
)
# ── 株価・指標取得 ────────────────────────────
def get_bars(symbol, days=120):
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=days),
    )
    df = data.get_stock_bars(req).df
    if hasattr(df.index, 'levels'):
        df = df.loc[symbol]
    df = df.sort_index()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    delta = df['close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    return df

# ── Claude分析 ────────────────────────────────
def analyze(symbol, df, account, positions):
    last  = df.iloc[-1]
    hist  = df.tail(5)[['close','rsi']].to_string()
    pos   = next((p for p in positions if p.symbol == symbol), None)
    pos_info = f"{pos.qty}株保有 (取得${float(pos.avg_entry_price):.2f}, PnL${float(pos.unrealized_pl):.2f})" if pos else "なし"

    per, eps, net_margin, roe = fetch_fundamentals(symbol)

    prompt = build_analysis_prompt(
        symbol        = symbol,
        risk_type     = "HIGH_RISK" if symbol in HIGH_RISK else "LOW_RISK",
        last          = last,
        hist          = hist,
        account_cash  = float(account.cash),
        account_value = float(account.portfolio_value),
        pos_info      = pos_info,
        skills        = SKILLS,
        symbol_desc   = ALL_SYMBOLS.get(symbol, ""),
        per           = per,
        eps           = eps,
        net_margin    = net_margin,
        roe           = roe,
        df            = df,
    )

    # LLMバックエンド（Claude/PLaMo）はllm_client.decide()が吸収する
    return decide(prompt)

# ── 注文実行 ──────────────────────────────────
def execute(symbol, action, qty):
    if action in ("BUY", "SELL") and qty > 0:
        order = trade.submit_order(MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if action == "BUY" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC
        ))
        print(f"  → 注文送信完了 ID: {order.id}")

# ── メイン ────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" Alpaca AI Trader — 全銘柄一括分析")
    print("="*50)

    # モード選択
    print("\n分析モードを選択:")
    print("  1. 全20銘柄")
    print("  2. ハイリスク10銘柄のみ")
    print("  3. ローリスク10銘柄のみ")
    mode = input("番号: ").strip()

    if mode == "2":
        targets = list(HIGH_RISK.keys())
    elif mode == "3":
        targets = list(LOW_RISK.keys())
    else:
        targets = list(ALL_SYMBOLS.keys())

    account   = trade.get_account()
    positions = trade.get_all_positions()
    print(f"\n口座残高: ${float(account.cash):,.2f} / 総資産: ${float(account.portfolio_value):,.2f}")
    print(f"分析対象: {len(targets)}銘柄\n")

    results = []

    for symbol in targets:
        name = ALL_SYMBOLS[symbol]
        risk = "🔴高" if symbol in HIGH_RISK else "🟢低"
        print(f"[{risk}] {symbol} ({name}) 分析中...", end=" ", flush=True)
        try:
            df     = get_bars(symbol)
            result = analyze(symbol, df, account, positions)
            result["symbol"] = symbol
            result["name"]   = name
            result["price"]  = df.iloc[-1]['close']
            result["rsi"]    = df.iloc[-1]['rsi']
            results.append(result)
            action_icon = "▲" if result["action"] == "BUY" else ("▼" if result["action"] == "SELL" else "—")
            print(f"{action_icon} {result['action']} (確信度{result.get('confidence',0)}%) | {result['reason'][:40]}")
            time.sleep(0.5)  # API制限対策
        except Exception as e:
            print(f"エラー: {e}")

    # 結果サマリー
    print("\n" + "="*50)
    print(" 分析結果サマリー")
    print("="*50)
    buys  = [r for r in results if r["action"] == "BUY"]
    sells = [r for r in results if r["action"] == "SELL"]
    holds = [r for r in results if r["action"] == "HOLD"]

    if buys:
        print(f"\n▲ BUY候補 ({len(buys)}銘柄):")
        for r in sorted(buys, key=lambda x: -x.get("confidence", 0)):
            print(f"  {r['symbol']:6} ${r['price']:.2f}  確信度{r.get('confidence',0)}%  RSI:{r['rsi']:.0f}  {r['reason'][:50]}")

    if sells:
        print(f"\n▼ SELL候補 ({len(sells)}銘柄):")
        for r in sorted(sells, key=lambda x: -x.get("confidence", 0)):
            print(f"  {r['symbol']:6} ${r['price']:.2f}  確信度{r.get('confidence',0)}%  RSI:{r['rsi']:.0f}  {r['reason'][:50]}")

    if holds:
        print(f"\n— HOLD ({len(holds)}銘柄): {', '.join(r['symbol'] for r in holds)}")

    # 注文確認
    if buys or sells:
        print("\n上記の注文を全て執行しますか？ (y/n/s=個別選択): ", end="")
        ans = input().strip().lower()
        if ans == "y":
            for r in buys + sells:
                execute(r["symbol"], r["action"], r.get("qty", 1))
        elif ans == "s":
            for r in buys + sells:
                print(f"{r['action']} {r['symbol']} {r.get('qty',1)}株 執行? (y/n): ", end="")
                if input().strip().lower() == "y":
                    execute(r["symbol"], r["action"], r.get("qty", 1))
        else:
            print("全てキャンセル")
    else:
        print("\n全銘柄HOLD — 注文なし")

    print("\n完了")
