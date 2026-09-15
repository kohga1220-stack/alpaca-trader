"""指定銘柄のポジションをワンショットでクローズするユーティリティ。

ショート（空売り）ポジションを買い戻して閉じる、または
ロングポジションを売って閉じる。数量・方向はAlpaca側の現在保有から
自動算出するため、手入力ミスで空売り/過剰買いになることはない。

使い方:
    .venv/bin/python cover_position.py PG          # 確認プロンプトあり
    .venv/bin/python cover_position.py PG --yes     # 確認スキップ（自動実行）
"""
import os
import sys

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

load_dotenv()

trade = TradingClient(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    paper=True,
)


def main() -> None:
    args = [a for a in sys.argv[1:]]
    auto_yes = "--yes" in args or "-y" in args
    symbols = [a for a in args if not a.startswith("-")]

    if not symbols:
        print("使い方: python cover_position.py <SYMBOL> [--yes]")
        sys.exit(1)

    symbol = symbols[0].upper()

    # 現在のポジションを取得
    try:
        pos = trade.get_open_position(symbol)
    except Exception:
        print(f"[{symbol}] 保有ポジションがありません。クローズ不要です。")
        return

    qty = int(float(pos.qty))
    side = "ショート(空売り)" if qty < 0 else "ロング"
    action = "買い戻し(BUY)" if qty < 0 else "売却(SELL)"
    pnl = float(pos.unrealized_pl)
    market_value = float(pos.market_value)

    print("=" * 50)
    print(f"  銘柄        : {symbol}")
    print(f"  方向        : {side}")
    print(f"  保有数      : {qty}株")
    print(f"  評価額      : ${market_value:,.2f}")
    print(f"  含み損益    : ${pnl:+,.2f}")
    print(f"  → 実行内容  : {abs(qty)}株を {action} してポジションを完全クローズ")
    print("=" * 50)

    if not auto_yes:
        ans = input("このポジションをクローズしますか？ [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("中止しました。")
            return

    # 残っている未約定注文を先にキャンセル（held_for_orders回避）
    try:
        open_orders = trade.get_orders()
        cancelled = 0
        for o in open_orders:
            if o.symbol == symbol:
                trade.cancel_order_by_id(o.id)
                cancelled += 1
        if cancelled:
            print(f"  未約定注文 {cancelled}件をキャンセルしました")
    except Exception as e:
        print(f"  注文キャンセル時の警告: {e}")

    # close_position は方向・数量を自動判定して反対売買を出す（空売り化しない）
    try:
        order = trade.close_position(symbol)
        print(f"✅ クローズ注文を送信しました ID:{order.id}  ({abs(qty)}株 {action})")
        print("   約定後、Alpaca画面でポジションが消えていることを確認してください。")
    except Exception as e:
        print(f"❌ クローズ失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
