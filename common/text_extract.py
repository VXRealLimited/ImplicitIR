"""Parsing helpers for model completions of the form
`<think>...</think>\n{answer text or {"answer": ...} JSON}`.

Used by the GRPO reward functions (reward_funcs.py) to pull out the final
answer and the reasoning trace from a sampled completion.
"""

import json


def extract_answer(text: str) -> str:
    """Return the part after </think>, or the whole text if no thinking block.

    If that tail parses as JSON with an "answer" key (e.g. {"answer": "25"}),
    return the stringified value instead -- otherwise a correct prediction like
    that would only match a ground truth of "25" via lucky substring overlap
    rather than a real comparison.
    """
    tail = text.split("</think>")[-1].strip() if "</think>" in text else text.strip()
    try:
        parsed = json.loads(tail)
        if isinstance(parsed, dict) and "answer" in parsed:
            return str(parsed["answer"]).strip()
    except (json.JSONDecodeError, ValueError):
        pass
    return tail


def extract_thinking(text: str) -> str:
    """Return the content between <think> and </think>, or up to </think> if the
    opening tag is missing (it can be server/template-injected rather than
    present in raw text), or "" if there's no </think> at all.
    """
    if "</think>" not in text:
        return ""
    before = text.split("</think>", 1)[0]
    if "<think>" in before:
        before = before.split("<think>", 1)[1]
    return before.strip()


def completion_text(completion) -> str:
    """TRL passes completions as [{"role": ..., "content": ...}]; other callers
    may already have a plain string."""
    return completion[0]["content"] if isinstance(completion, list) else str(completion)
