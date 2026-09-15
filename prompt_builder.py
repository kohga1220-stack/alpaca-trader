"""
prompt_builder.py
スキルファイルを読み込んで構造化プロンプトを生成するモジュール。
"""

from pathlib import Path
import pandas as pd

SKILLS_DIR = Path(__file__).parent / "skills"


def load_skills() -> dict:
    files = {
        "definitions": SKILLS_DIR / "definitions.md",
        "high_risk":   SKILLS_DIR / "high_risk.md",
        "low_risk":    SKILLS_DIR / "low_risk.md",
    }
    skills = {}
    for key, path in files.items():
        if path.exists():
            skills[key] = path.read_text(encoding="utf-8")
        else:
            print(f"[WARNING] スキルファイルが見つかりません: {path}")
            skills[key] = ""
    return skills


def _cross_status(df: "pd.DataFrame") -> str:
    """直近のゴールデンクロス・デッドクロス状態を文字列で返す。

    直近2日のMA20/MA50の上下関係を比較して判定する。
    """
    if len(df) < 2:
        return "判定不可（データ不足）"
    prev = df.iloc[-2]
    last = df.iloc[-1]
    prev_above = prev["ma20"] > prev["ma50"]
    last_above = last["ma20"] > last["ma50"]

    if not prev_above and last_above:
        return "ゴールデンクロス発生（直近1日）"
    if prev_above and not last_above:
        return "デッドクロス発生（直近1日）"
    if last_above:
        return "MA20 > MA50（ゴールデンクロス後の上昇継続中）"
    return "MA20 < MA50（デッドクロス後の下降継続中）"


def build_analysis_prompt(
    symbol, risk_type, last, hist,
    account_cash, account_value, pos_info, skills,
    symbol_desc="",
    per=None, eps=None, net_margin=None, roe=None,
    df=None,
) -> str:
    domain_skill = skills.get("high_risk" if risk_type == "HIGH_RISK" else "low_risk", "")
    definitions  = skills.get("definitions", "")

    fundamentals_block = ""
    if any(v is not None for v in [per, eps, net_margin, roe]):
        lines = []
        if per        is not None: lines.append(f"PER: {per:.1f}")
        if eps        is not None: lines.append(f"EPS: {eps:.2f}")
        if net_margin is not None: lines.append(f"純利益率: {net_margin:.1f}%")
        if roe        is not None: lines.append(f"ROE: {roe:.1f}%")
        fundamentals_block = "ファンダメンタルズ:\n" + " / ".join(lines)

    cross_line = f"クロス状態: {_cross_status(df)}" if df is not None else ""

    return f"""あなたはプロの株式アナリストです。
以下の【判断基準】と【銘柄ドメイン知識】を厳守したうえで、投資判断をJSONのみで返してください。

---
【判断基準】
{definitions}

---
【銘柄ドメイン知識 ({risk_type})】
{domain_skill}

---
【分析対象データ】
銘柄: {symbol}（{symbol_desc}）/ {risk_type}
現在価格: ${last['close']:.2f}
MA20: ${last['ma20']:.2f} / MA50: ${last['ma50']:.2f}
{cross_line}
RSI(14): {last['rsi']:.1f}
口座残高: ${account_cash:.2f} / 総資産: ${account_value:.2f}
保有: {pos_info}
{fundamentals_block}

直近5日:
{hist}

---
【回答形式】
submit_decision ツールを使って判断を返すこと。
reason は20字以内の日本語で簡潔に（例: "RSI28+ファンダ優良+GC継続"）。

重要: confidence < 75 のときは必ず action を "HOLD" にすること。"""
