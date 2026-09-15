"""LLMバックエンド切替層。

環境変数 LLM_PROVIDER で分析エンジンを切り替える。
    - "claude" (デフォルト): Anthropic claude-sonnet-4-6 + Tool Use
    - "plamo"             : PLaMo 3.0 Prime（OpenAI互換API）+ function calling

どちらも同じ判断スキーマ submit_decision を強制し、
{action, qty, reason, confidence} の dict を返す。

呼び出し側（scheduler.py / trader.py）は decide(prompt) を呼ぶだけでよく、
プロバイダの差異はこのモジュールに閉じ込める。
"""
import os
import json

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "claude").lower()

CLAUDE_MODEL = "claude-sonnet-4-6"
PLAMO_MODEL  = "plamo-3.0-prime"
PLAMO_BASE_URL = "https://api.platform.preferredai.jp/v1"
MAX_TOKENS = 200

# 判断スキーマ（プロバイダ非依存の正本）。
# Anthropic形式の input_schema をそのまま OpenAI parameters にも流用する。
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action":     {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
        "qty":        {"type": "integer", "description": "推奨株数（整数）"},
        "reason":     {"type": "string", "description": "判断理由。20字以内の日本語"},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["action", "qty", "reason", "confidence"],
}
DECISION_DESC = "株式の投資判断を構造化して返す。"

# Anthropic Tool Use形式（既存コードとの後方互換のため公開）
DECISION_TOOL = {
    "name": "submit_decision",
    "description": DECISION_DESC,
    "input_schema": DECISION_SCHEMA,
}

# OpenAI function calling形式（PLaMo用）
_OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_decision",
        "description": DECISION_DESC,
        "parameters": DECISION_SCHEMA,
    },
}

# ── クライアントは遅延初期化（使うプロバイダのキーだけ要求する）─────────────
_claude = None
_plamo = None


def _get_claude():
    global _claude
    if _claude is None:
        import anthropic
        _claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _claude


def _get_plamo():
    global _plamo
    if _plamo is None:
        from openai import OpenAI
        _plamo = OpenAI(
            api_key=os.getenv("PLAMO_API_KEY"),
            base_url=PLAMO_BASE_URL,
        )
    return _plamo


def _decide_claude(prompt: str) -> dict:
    res = _get_claude().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        tools=[DECISION_TOOL],
        tool_choice={"type": "tool", "name": "submit_decision"},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in res.content:
        if block.type == "tool_use":
            return block.input
    raise ValueError(f"tool_useが見つかりません: {res.content}")


def _decide_plamo(prompt: str) -> dict:
    # reasoning_effort は "none"（高速・低コスト）/ "medium"（熟考）のみ有効
    effort = os.getenv("PLAMO_REASONING", "none").lower()
    res = _get_plamo().chat.completions.create(
        model=PLAMO_MODEL,
        max_tokens=MAX_TOKENS,
        tools=[_OPENAI_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_decision"}},
        messages=[{"role": "user", "content": prompt}],
        extra_body={"reasoning_effort": effort},
    )
    msg = res.choices[0].message
    if msg.tool_calls:
        return json.loads(msg.tool_calls[0].function.arguments)
    # フォールバック: 本文がJSONで返るケース
    if msg.content:
        return json.loads(msg.content)
    raise ValueError(f"tool_callsが見つかりません: {msg}")


def decide(prompt: str, provider: str | None = None) -> dict:
    """プロンプトを投げて投資判断 dict を返す。

    Args:
        prompt: build_analysis_prompt() が生成した分析プロンプト
        provider: "claude" / "plamo"。省略時は LLM_PROVIDER 環境変数（既定 claude）

    Returns:
        {"action", "qty", "reason", "confidence"} の dict
    """
    p = (provider or PROVIDER).lower()
    if p == "plamo":
        return _decide_plamo(prompt)
    return _decide_claude(prompt)
