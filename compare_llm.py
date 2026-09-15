"""Claude と PLaMo の判断をA/B比較する検証ツール。

同一の分析プロンプト（実際の株価・指標・ファンダ・スキル注入込み）を
両プロバイダに投げ、判断・確信度・理由・応答速度を並べて表示する。
scheduler.py を本番切替する前の精度確認に使う。

使い方:
    .venv/bin/python compare_llm.py                 # 代表5銘柄
    .venv/bin/python compare_llm.py NVDA JNJ KO     # 指定銘柄
"""
import sys
import time

from prompt_builder import build_analysis_prompt
from llm_client import decide
from scheduler import (
    trade, get_bars, fetch_fundamentals, pre_filter_score,
    ALL_SYMBOLS, HIGH_RISK, SKILLS,
)

DEFAULT_SYMBOLS = ["NVDA", "GOOGL", "JNJ", "KO", "VZ"]


def build_prompt(symbol, df, account, positions, per, eps, net_margin, roe,
                  pre_score=None, pre_reasons=None):
    last = df.iloc[-1]
    pos = next((p for p in positions if p.symbol == symbol), None)
    pos_info = (f"{pos.qty}株保有 取得${float(pos.avg_entry_price):.2f} "
                f"PnL${float(pos.unrealized_pl):.2f}") if pos else "なし"
    hist = df.tail(5)[["close", "rsi"]].to_string()
    return build_analysis_prompt(
        symbol=symbol,
        risk_type="HIGH_RISK" if symbol in HIGH_RISK else "LOW_RISK",
        last=last, hist=hist,
        account_cash=float(account.cash),
        account_value=float(account.portfolio_value),
        pos_info=pos_info, skills=SKILLS,
        symbol_desc=ALL_SYMBOLS.get(symbol, ""),
        per=per, eps=eps, net_margin=net_margin, roe=roe, df=df,
        pre_score=pre_score, pre_reasons=pre_reasons,
    )


def run_one(prompt, provider):
    t0 = time.time()
    try:
        out = decide(prompt, provider=provider)
        dt = time.time() - t0
        return out, dt, None
    except Exception as e:
        return None, time.time() - t0, str(e)[:120]


def fmt(out, dt, err):
    if err:
        return f"ERROR({dt:.1f}s): {err}"
    return (f"{out.get('action'):4} conf={out.get('confidence'):3}% "
            f"qty={out.get('qty'):>3} {dt:4.1f}s | {out.get('reason','')}")


def main():
    symbols = [s.upper() for s in sys.argv[1:]] or DEFAULT_SYMBOLS

    account = trade.get_account()
    positions = trade.get_all_positions()
    print(f"\n口座: 現金${float(account.cash):,.0f} / 総資産${float(account.portfolio_value):,.0f}")
    print(f"比較対象: {len(symbols)}銘柄  (Claude vs PLaMo)\n")
    print("=" * 78)

    agree = 0
    total = 0
    for symbol in symbols:
        try:
            df = get_bars(symbol)
            per, eps, nm, roe = fetch_fundamentals(symbol)
            score, score_reasons = pre_filter_score(df, net_margin=nm, roe=roe)
            prompt = build_prompt(symbol, df, account, positions, per, eps, nm, roe,
                                   pre_score=score, pre_reasons=score_reasons)
        except Exception as e:
            print(f"[{symbol}] データ取得失敗: {e}")
            continue

        c_out, c_dt, c_err = run_one(prompt, "claude")
        p_out, p_dt, p_err = run_one(prompt, "plamo")

        risk = "🔴" if symbol in HIGH_RISK else "🟢"
        print(f"{risk} {symbol:6} (事前スコア{score})")
        print(f"   Claude : {fmt(c_out, c_dt, c_err)}")
        print(f"   PLaMo  : {fmt(p_out, p_dt, p_err)}")

        if c_out and p_out:
            total += 1
            same = c_out.get("action") == p_out.get("action")
            agree += same
            diff = abs(c_out.get("confidence", 0) - p_out.get("confidence", 0))
            mark = "✅一致" if same else "⚠️不一致"
            print(f"   判定   : {mark}  (確信度差 {diff}pt)")
        print("-" * 78)

    if total:
        print(f"\nアクション一致率: {agree}/{total} ({agree/total*100:.0f}%)")
    print("※ 一致率が高く理由が妥当なら、scheduler.py を LLM_PROVIDER=plamo に切替可能")


if __name__ == "__main__":
    main()
