import os, json, time, logging, schedule, requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, StopLossRequest, TakeProfitRequest, GetOrdersRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from prompt_builder import build_analysis_prompt, load_skills
from llm_client import decide, DECISION_TOOL, PROVIDER

load_dotenv()
os.makedirs("portfolio", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("portfolio/trading.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

STOP_LOSS_PCT    = 0.06
TAKE_PROFIT_PCT  = 0.12
RISK_PER_TRADE   = 0.02
MIN_CONFIDENCE   = 75

# DECISION_TOOL / 判断スキーマは llm_client.py が正本（Claude/PLaMo共通）。
# 後方互換のため上でimportして再公開している。

HIGH_RISK = {
    "NVDA":"NVIDIA","TSLA":"Tesla","META":"Meta","AMZN":"Amazon",
    "GOOGL":"Alphabet","MSFT":"Microsoft","AMD":"AMD","NFLX":"Netflix",
    "CRM":"Salesforce","PLTR":"Palantir",
}
LOW_RISK = {
    "JNJ":"J&J","PG":"P&G","KO":"Coca-Cola","WMT":"Walmart",
    "MCD":"McDonald's","PEP":"PepsiCo","ABBV":"AbbVie",
    "XOM":"Exxon","VZ":"Verizon","SPY":"S&P500 ETF",
}
ALL_SYMBOLS = {**HIGH_RISK, **LOW_RISK}
SKILLS = load_skills()

trade  = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)
data   = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))

def slack(msg, emoji="📊"):
    url = os.getenv("SLACK_WEBHOOK_URL", "")
    if not url:
        return
    try:
        requests.post(url, json={"text": f"{emoji} {msg}"}, timeout=5)
    except:
        pass

def calc_position_size(cash, price, stop_loss_price):
    if price <= stop_loss_price:
        return 0
    risk_amount    = cash * RISK_PER_TRADE
    loss_per_share = price - stop_loss_price
    raw_qty        = risk_amount / loss_per_share
    max_qty        = int((cash * 0.20) / price)
    qty            = min(int(raw_qty), max_qty)
    return max(qty, 1)

def execute_buy(symbol, price, cash):
    stop_price   = round(price * (1 - STOP_LOSS_PCT), 2)
    target_price = round(price * (1 + TAKE_PROFIT_PCT), 2)
    qty          = calc_position_size(cash, price, stop_price)
    if qty <= 0:
        return None
    try:
        order = trade.submit_order(MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=stop_price),
            take_profit=TakeProfitRequest(limit_price=target_price),
        ))
        log.info(f"  [{symbol}] ブラケット注文完了 ID:{order.id} SL=${stop_price} TP=${target_price}")
        return {"order": order, "qty": qty, "stop": stop_price, "target": target_price}
    except Exception as e:
        # SL/TPなしにフォールバックせず、エラーをログに残してスキップ
        log.error(f"  [{symbol}] ブラケット注文失敗（SL/TPなし注文は行いません）: {e}")
        return None

def cancel_open_orders(symbol) -> int:
    """指定銘柄のオープン注文（ブラケット含む）を全てキャンセルする。

    Returns:
        キャンセルした注文件数
    """
    try:
        open_orders = trade.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol],
        ))
        if not open_orders:
            return 0
        for o in open_orders:
            try:
                trade.cancel_order_by_id(o.id)
                log.info(f"  [{symbol}] 既存注文キャンセル ID:{o.id} ({o.order_class}/{o.side})")
            except Exception as e:
                log.warning(f"  [{symbol}] 注文キャンセル失敗 ID:{o.id}: {e}")
        return len(open_orders)
    except Exception as e:
        log.error(f"  [{symbol}] オープン注文取得失敗: {e}")
        return 0


def get_live_qty(symbol):
    """注文直前に最新の保有株数を取得する。

    Returns:
        int: 保有株数（ロング>0 / ショート<0）。未保有・取得失敗時は0。
    """
    try:
        pos = trade.get_open_position(symbol)
        return int(float(pos.qty))
    except Exception:
        # ポジションが存在しない場合もここに来る（=0株）
        return 0


def execute_sell(symbol, qty):
    # ブラケット注文が残っているとheld_for_ordersになりSELLできないため先にキャンセル
    cancelled = cancel_open_orders(symbol)
    if cancelled > 0:
        log.info(f"  [{symbol}] {cancelled}件の注文をキャンセル後にSELL実行")
        time.sleep(0.5)  # キャンセル反映待ち

    # 【空売り防止】注文直前にライブの保有数を再取得し、ロング保有数を上限にクランプする。
    # スナップショットが古い/多重起動で重複SELLが走っても、保有を超えて売らない。
    live_qty = get_live_qty(symbol)
    if live_qty <= 0:
        log.warning(f"  [{symbol}] 保有なし(live={live_qty}) — SELLをスキップ（空売り防止）")
        return None
    sell_qty = min(int(qty), live_qty)
    if sell_qty < int(qty):
        log.warning(f"  [{symbol}] 売却数を {qty}→{sell_qty} に調整（実保有数に合わせて空売り防止）")

    try:
        order = trade.submit_order(MarketOrderRequest(
            symbol=symbol, qty=sell_qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
        ))
        log.info(f"  [{symbol}] 売り注文完了 {sell_qty}株 ID:{order.id}")
        return {"order": order, "qty": sell_qty}
    except Exception as e:
        log.error(f"  [{symbol}] 売り注文失敗: {e}")
        return None

PRE_FILTER_THRESHOLD = 75  # これ未満はAPIを呼ばずルールベースHOLD
AUTO_BUY_THRESHOLD   = 85  # これ以上はLLMを呼ばずルールベースで直接BUY
COOLDOWN_HOURS = 6         # 一度売買した銘柄はこの時間内は再取引しない（往復売買防止）


def recently_traded(symbol: str, hours: int = COOLDOWN_HOURS) -> tuple[bool, datetime | None]:
    """直近 hours 時間以内に当該銘柄を売買したかを trades.jsonl から判定する。

    売買直後の銘柄を反対売買して往復（ホイップソー）するのを防ぐためのガード。
    SL/TPブラケットは別途生きているので、急落時の自動損切りは引き続き機能する。

    Returns:
        (cooled, last_ts): クールダウン中か / 直近取引時刻
    """
    path = "portfolio/trades.jsonl"
    if not os.path.exists(path):
        return False, None
    cutoff = datetime.now(ET) - timedelta(hours=hours)
    latest = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("symbol") != symbol:
                continue
            try:
                ts = datetime.fromisoformat(rec["ts"])
            except Exception:
                continue
            if latest is None or ts > latest:
                latest = ts
    if latest is not None and latest >= cutoff:
        return True, latest
    return False, None


def sell_signal_check(df) -> tuple[bool, str]:
    """保有銘柄に売り検討シグナルが出ているかをルールベースで判定する。

    買い向けの pre_filter_score は下落局面でスコアが下がり API がスキップされるため、
    保有銘柄の出口判断はこちらの売りシグナルで別途検知して Claude に回す。

    Returns:
        (signal, reason): シグナル有無と理由
    """
    last = df.iloc[-1]

    # デッドクロス発生（直近2日でMA20がMA50を下抜け）
    if len(df) >= 2:
        prev = df.iloc[-2]
        if prev["ma20"] >= prev["ma50"] and last["ma20"] < last["ma50"]:
            return True, "デッドクロス発生"

    # 過熱（利確検討ゾーン）
    if last["rsi"] > 70:
        return True, f"RSI{last['rsi']:.0f}>70過熱"

    # 下降トレンド入り（価格がMA20・MA50の下）。
    # ただし売られすぎ(RSI<50)は買い側の逆張り条件と衝突し往復売買(ホイップソー)を生むため、
    # RSIが50以上＝明確に売られすぎでない場合のみ売りシグナルとする。
    if last["close"] < last["ma20"] < last["ma50"] and last["rsi"] >= 50:
        return True, "下降トレンド(価格<MA20<MA50)"

    return False, ""

def pre_filter_score(df, net_margin: float | None = None, roe: float | None = None) -> tuple[int, list[str]]:
    """definitions.mdの加点ルールをPythonで事前計算する。

    Returns:
        (score, reasons): 推定確信度スコアと加点理由のリスト
    """
    last   = df.iloc[-1]
    score  = 50
    reasons: list[str] = []

    # 上昇トレンド
    if last["close"] > last["ma20"] > last["ma50"]:
        score += 20
        reasons.append("上昇トレンド+20")

    # RSI
    rsi = last["rsi"]
    if rsi < 30:
        score += 20
        reasons.append(f"RSI{rsi:.0f}<30+20")
    elif rsi < 45:
        score += 10
        reasons.append(f"RSI{rsi:.0f}<45+10")
    elif rsi < 65:
        score += 5
        reasons.append(f"RSI{rsi:.0f}<65+5")

    # ゴールデンクロス（直近2日でMA20がMA50を上抜け）
    if len(df) >= 2:
        prev = df.iloc[-2]
        if prev["ma20"] <= prev["ma50"] and last["ma20"] > last["ma50"]:
            score += 15
            reasons.append("GC+15")

    # ファンダメンタルズ
    if net_margin is not None and roe is not None:
        if net_margin > 20 and roe > 20:
            score += 20
            reasons.append("ファンダ優良+20")
        elif net_margin > 10 or roe > 10:
            score += 10
            reasons.append("ファンダ良好+10")

    # MA50サポート
    if last["close"] >= last["ma50"] * 0.97:
        score += 10
        reasons.append("MA50サポート+10")

    # 直近5日で3日以上上昇
    closes = df["close"].tail(5)
    if (closes.diff().dropna() > 0).sum() >= 3:
        score += 5
        reasons.append("直近5日+5")

    return score, reasons


def get_bars(symbol, days=120):
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=days),
    )
    df = data.get_stock_bars(req).df
    if hasattr(df.index, "levels"):
        df = df.loc[symbol]
    df = df.sort_index()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))
    return df

def fetch_fundamentals(symbol: str) -> tuple[float | None, float | None, float | None, float | None]:
    """yfinanceでファンダメンタルズを取得する。失敗時はすべてNoneを返す。

    Returns:
        (per, eps, net_margin, roe)
    """
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        per = info.get('trailingPE')
        eps = info.get('trailingEps')
        pm  = round(info.get('profitMargins', 0) * 100, 1) if info.get('profitMargins') else None
        roe = round(info.get('returnOnEquity', 0) * 100, 1) if info.get('returnOnEquity') else None
        per = float(per) if per is not None else None
        eps = float(eps) if eps is not None else None
        return per, eps, pm, roe
    except Exception:
        return None, None, None, None


def analyze(symbol, df, account, positions,
            per=None, eps=None, net_margin=None, roe=None,
            pre_score=None, pre_reasons=None):
    last     = df.iloc[-1]
    pos      = next((p for p in positions if p.symbol == symbol), None)
    pos_info = (f"{pos.qty}株保有 取得${float(pos.avg_entry_price):.2f} "
                f"PnL${float(pos.unrealized_pl):.2f}") if pos else "なし"

    hist = df.tail(5)[['close','rsi']].to_string()
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
        pre_score     = pre_score,
        pre_reasons   = pre_reasons,
    )

    # LLMバックエンド（Claude/PLaMo）はllm_client.decide()が吸収する
    return decide(prompt)

def save_trade_log(record):
    with open("portfolio/trades.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def hourly_run():
    now = datetime.now(ET)
    log.info(f"\n{'='*50}")
    log.info(f" 定期実行 {now.strftime('%Y-%m-%d %H:%M ET')}")
    log.info(f"{'='*50}")

    if not trade.get_clock().is_open:
        log.info("市場クローズ中 — スキップ")
        return

    try:
        account   = trade.get_account()
        positions = trade.get_all_positions()
        cash      = float(account.cash)
        pv        = float(account.portfolio_value)
        log.info(f"残高: ${cash:,.2f} / 総資産: ${pv:,.2f}")
    except Exception as e:
        log.error(f"口座情報取得失敗: {e}")
        return

    buy_results  = []
    sell_results = []

    for symbol, name in ALL_SYMBOLS.items():
        try:
            # ── クールダウン: 直近に売買した銘柄は反対売買せず見送る（往復売買防止）──
            cooled, last_ts = recently_traded(symbol)
            if cooled:
                log.info(f"[{symbol}] クールダウン中（前回取引 {last_ts:%m-%d %H:%M} から{COOLDOWN_HOURS}h以内）— スキップ")
                time.sleep(0.1)
                continue

            df  = get_bars(symbol)
            per, eps, net_margin, roe = fetch_fundamentals(symbol)
            pos = next((p for p in positions if p.symbol == symbol), None)

            # ── 事前フィルタリング ─────────────────────────────────────
            # 未保有: 買いスコアが閾値以上のときだけAPIへ
            # 保有中: 買いスコアでは出口を判断できないため、売りシグナルでもAPIへ回す。
            #         既保有でBUYしか出ない高スコア銘柄はAPIを呼んでも注文不可なので、
            #         売りシグナルがなければスキップしてAPI消費を抑える。
            pre_score, pre_reasons = pre_filter_score(df, net_margin=net_margin, roe=roe)
            price = float(df.iloc[-1]["close"])

            if pos is None:
                if pre_score < PRE_FILTER_THRESHOLD:
                    log.info(f"[{symbol}] HOLD(ルールベース スコア{pre_score}<{PRE_FILTER_THRESHOLD}) — API呼び出しなし")
                    time.sleep(0.1)
                    continue

                if pre_score >= AUTO_BUY_THRESHOLD:
                    # スコアが十分高い場合はLLMを呼ばずルールベースで直接BUYする
                    # （LLMが同じ根拠から再判断すると保守的にHOLD/低確信度へ寄る傾向があるため）
                    action, conf, reason = "BUY", pre_score, ", ".join(pre_reasons)
                    log.info(f"[{symbol}] スコア{pre_score}≥{AUTO_BUY_THRESHOLD} → ルールベース直接BUY（LLM呼び出しなし）")
                else:
                    log.info(f"[{symbol}] スコア{pre_score}({', '.join(pre_reasons)}) → LLM最終判断へ")
                    result = analyze(symbol, df, account, positions,
                                      per=per, eps=eps, net_margin=net_margin, roe=roe,
                                      pre_score=pre_score, pre_reasons=pre_reasons)
                    action = result.get("action", "HOLD")
                    conf   = result.get("confidence", 0)
                    reason = result.get("reason", "")
                    log.info(f"[{symbol}] {action} conf={conf}% | {reason}")
            else:
                sell_sig, sell_reason = sell_signal_check(df)
                if not sell_sig:
                    log.info(f"[{symbol}] 保有中・売りシグナルなし — API呼び出しなし")
                    time.sleep(0.1)
                    continue
                log.info(f"[{symbol}] 保有中・売りシグナル({sell_reason}) → LLM最終判断へ")
                result = analyze(symbol, df, account, positions,
                                  per=per, eps=eps, net_margin=net_margin, roe=roe)
                action = result.get("action", "HOLD")
                conf   = result.get("confidence", 0)
                reason = result.get("reason", "")
                log.info(f"[{symbol}] {action} conf={conf}% | {reason}")
            # ──────────────────────────────────────────────────────────

            if conf < MIN_CONFIDENCE:
                log.info(f"  → 確信度不足({conf}%) — スキップ")
                time.sleep(1.5)
                continue

            if action == "BUY":
                if pos is not None:
                    log.info(f"  → 既に{pos.qty}株保有のため買い増しせずスキップ")
                else:
                    res = execute_buy(symbol, price, cash)
                    if res:
                        qty = res["qty"]
                        buy_results.append({"symbol": symbol, "qty": qty, "price": price,
                                            "confidence": conf, "reason": reason})
                        save_trade_log({"ts": now.isoformat(), "symbol": symbol, "action": "BUY",
                                        "qty": qty, "price": price, "confidence": conf, "reason": reason})
                        cash -= qty * price

            elif action == "SELL":
                if pos is None:
                    log.info("  → 未保有のためSELL不可 — スキップ")
                else:
                    pnl = float(pos.unrealized_pl)
                    res = execute_sell(symbol, int(float(pos.qty)))
                    if res:
                        qty = res["qty"]  # クランプ後の実売却数を記録する
                        sell_results.append({"symbol": symbol, "qty": qty, "price": price,
                                             "confidence": conf, "reason": reason, "pnl": pnl})
                        save_trade_log({"ts": now.isoformat(), "symbol": symbol, "action": "SELL",
                                        "qty": qty, "price": price, "confidence": conf,
                                        "reason": reason, "pnl": pnl})

            time.sleep(1.5)

        except Exception as e:
            log.error(f"[{symbol}] エラー: {e}")

    # 売買があったときだけSlackにサマリーを通知する
    if buy_results or sell_results:
        lines = []
        for r in buy_results:
            lines.append(f"BUY {r['symbol']} {r['qty']}株 @${r['price']:.2f} "
                         f"conf={r['confidence']}% {r['reason']}")
        for r in sell_results:
            lines.append(f"SELL {r['symbol']} {r['qty']}株 @${r['price']:.2f} "
                         f"PnL${r['pnl']:+,.2f} conf={r['confidence']}% {r['reason']}")
        slack("\n".join(lines), emoji="💰")

    log.info("定期実行完了\n")

if __name__ == "__main__":
    log.info("Alpaca AI Scheduler 起動")
    log.info(f"設定: SL={STOP_LOSS_PCT*100:.0f}% TP={TAKE_PROFIT_PCT*100:.0f}% "
             f"リスク/トレード={RISK_PER_TRADE*100:.0f}% 最低確信度={MIN_CONFIDENCE}% "
             f"LLM={PROVIDER}")

    hourly_run()
    schedule.every(3).hours.do(hourly_run)

    while True:
        schedule.run_pending()
        time.sleep(30)
