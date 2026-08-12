"""Reward functions for GRPO training.

Each function scores a batch of completions against the reference fields
carried through by data_prep.py's GRPO rows (solution, reference_raw_data,
reference_derivation) and returns one float per completion, per the TRL
RewardFunc contract.
"""

import re
from difflib import SequenceMatcher

from common.text_extract import completion_text, extract_answer, extract_thinking

_RAW_TAG_RE = re.compile(r"<raw>(.*?)</raw>", re.DOTALL)
_NUMBER_RE = re.compile(r"\d{2,}")
_QUOTE_RE = re.compile(r'["\']([^"\']{2,40})["\']')
_EQUATION_RE = re.compile(r'(-?\d[\d,]*\.?\d*)\s*([+\-])\s*(-?\d[\d,]*\.?\d*)\s*=\s*(-?\d[\d,]*\.?\d*)')
_MONEY_RE = re.compile(r'-?\d[\d,]*\.\d+')


def _extract_facts(text: str) -> set[str]:
    """Concrete, checkable facts: multi-digit numbers (dates, amounts, codes) and
    quoted field values/names -- used to check whether a completion's <raw> or
    derivation text actually engages with the same concrete data as the reference,
    without requiring exact wording."""
    facts = set(_NUMBER_RE.findall(text))
    facts |= {m.strip() for m in _QUOTE_RE.findall(text)}
    return facts


def _to_float(token: str) -> float:
    return float(token.replace(",", ""))


def _extract_equations(text: str) -> list[tuple[float, str, float, float]]:
    """All "A + B = C" / "A - B = C" style arithmetic steps shown in the text,
    e.g. the "25.00 + 50.00 = 75.00, then 75.00 + 50.00 = 125.00" running-total
    chains seen in the training data's own derivation_reasoning examples."""
    equations = []
    for a, op, b, c in _EQUATION_RE.findall(text):
        try:
            equations.append((_to_float(a), op, _to_float(b), _to_float(c)))
        except ValueError:
            continue
    return equations


def _equation_holds(a: float, op: str, b: float, c: float, tol: float = 0.01) -> bool:
    expected = a + b if op == "+" else a - b
    return abs(expected - c) <= tol


def answer_accuracy(prompts, completions, solution, **kwargs) -> list[float]:
    """1.0 for exact/substring match (same rule as the eval repo's
    score_answer()). Otherwise, partial credit from string similarity so a
    near-miss on ANY answer type -- dates, labels, amounts, weekday names, not
    just numbers -- still gets a gradient toward correct instead of the same
    flat 0.0 as a wildly wrong answer."""
    rewards = []
    for completion, sol in zip(completions, solution):
        pred = extract_answer(completion_text(completion)).strip().lower()
        exp = str(sol).strip().lower()
        if exp == pred or exp in pred:
            rewards.append(1.0)
            continue
        similarity = SequenceMatcher(None, pred, exp).ratio()
        rewards.append(similarity if similarity > 0.5 else 0.0)
    return rewards


def grounding_quality(prompts, completions, reference_raw_data, reference_derivation, **kwargs) -> list[float]:
    """Single composite score for 'did it actually use the document correctly',
    averaging four checks:
    - structure: non-empty <raw> block with reasoning both before and after it
      (the intended relevance_reasoning -> raw_data -> derivation_reasoning shape)
    - recall: did it retrieve the facts the reference raw_data actually needed
    - precision: did it avoid stating facts absent from the reference (i.e. not
      fabricating/hallucinating)
    - usage: did the retrieved facts actually get used in the derivation, rather
      than sitting unused in the <raw> block just to farm the other checks
    0.0 if there's no <raw> block at all -- everything else is moot without it."""
    rewards = []
    for completion, ref_raw, ref_deriv in zip(completions, reference_raw_data, reference_derivation):
        thinking = extract_thinking(completion_text(completion))
        match = _RAW_TAG_RE.search(thinking)
        if not match or len(match.group(1).strip()) < 3:
            rewards.append(0.0)
            continue
        raw_block = match.group(1)
        derivation = thinking[match.end():]

        before, after = thinking[: match.start()].strip(), derivation.strip()
        structure = 1.0 if (before and after) else 0.5

        ref_facts = _extract_facts(ref_raw or "")
        recall = (sum(1 for f in ref_facts if f in raw_block) / len(ref_facts)) if ref_facts else 0.5

        raw_facts = _extract_facts(raw_block)
        precision = (sum(1 for f in raw_facts if f in ref_facts) / len(raw_facts)) if (raw_facts and ref_facts) else 0.5

        deriv_facts = _extract_facts(ref_deriv or "")
        usage = (sum(1 for f in deriv_facts if f in derivation) / len(deriv_facts)) if deriv_facts else 0.5

        rewards.append((structure + recall + precision + usage) / 4)
    return rewards


def format_coherence(prompts, completions, **kwargs) -> list[float]:
    """Single composite score for 'is this a well-formed response', averaging:
    - length: substantive but not rambling <think> content (capped growth to 1.0)
    - consistency: the submitted answer is actually grounded in that reasoning
      (appears in the thinking text), rather than a non-sequitur -- the model
      reasoned toward one value but output a different one."""
    rewards = []
    for completion in completions:
        text = completion_text(completion)
        thinking = extract_thinking(text)
        length_score = 0.0 if len(thinking) < 20 else min(1.0, len(thinking) / 600)

        answer = extract_answer(text).strip().lower()
        consistent = 1.0 if (answer and answer in thinking.lower()) else 0.0

        rewards.append((length_score + consistent) / 2)
    return rewards


def stepwise_arithmetic(prompts, completions, reference_raw_data, **kwargs) -> list[float]:
    """Rewards showing an explicit running-total chain (a + b = c, c + d = e, ...)
    for multi-item sums, with each shown step arithmetically correct -- verified
    against the completion's own numbers, not the reference (so it still fires
    even if a value was misread, as long as the shown arithmetic is internally
    consistent; grounding_quality is what checks the numbers themselves).

    Rewards two things, averaged:
    - coverage: showing roughly as many chained steps as there are terms to sum
      (estimated from how many decimal amounts appear in reference_raw_data)
    - accuracy: each shown step's arithmetic actually being correct
    """
    rewards = []
    for completion, ref_raw in zip(completions, reference_raw_data):
        thinking = extract_thinking(completion_text(completion))
        equations = _extract_equations(thinking)

        n_terms = len(_MONEY_RE.findall(ref_raw or ""))
        expected_steps = max(0, n_terms - 1)

        if expected_steps <= 1:
            # 0-2 terms: nothing meaningful to chain -- neutral on this axis
            rewards.append(0.5)
            continue

        if not equations:
            # jumped straight from values to a final total with no shown steps --
            # exactly the pattern that reliably produces wrong sums
            rewards.append(0.0)
            continue

        correct = sum(1 for a, op, b, c in equations if _equation_holds(a, op, b, c))
        accuracy = correct / len(equations)
        coverage = min(1.0, len(equations) / expected_steps)
        rewards.append((accuracy + coverage) / 2)
    return rewards


REWARD_FUNCS = [answer_accuracy, grounding_quality, format_coherence, stepwise_arithmetic]

DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "answer_accuracy": 2.5,
    "grounding_quality": 1.0,
    "format_coherence": 0.25,
    "stepwise_arithmetic": 0.75,
}


def reward_weights_list(weights: dict[str, float]) -> list[float]:
    return [weights[f.__name__] for f in REWARD_FUNCS]
