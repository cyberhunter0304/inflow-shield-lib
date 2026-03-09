"""
inflow_shield_lib.utils
-----------------------
Minimal utility functions extracted from llm_guard.util.
Only includes what PromptInjection and Toxicity actually use.
"""
import logging

logger = logging.getLogger(__name__)


def calculate_risk_score(score: float, threshold: float) -> float:
    """
    Normalize a raw model score into a 0.0-1.0 risk score
    relative to the threshold.

    Extracted directly from llm_guard.util.calculate_risk_score.
    - Score below threshold → scales from 0.0 to ~0.5
    - Score above threshold → scales from ~0.5 to 1.0
    """
    if threshold == 0:
        return 1.0
    if threshold == 1:
        return 0.0

    if score < threshold:
        return round(score / threshold * 0.5, 2)
    else:
        return round(0.5 + (score - threshold) / (1 - threshold) * 0.5, 2)


def split_text_by_sentences(text: str) -> list[str]:
    """
    Split text into sentences for sentence-level scanning.
    Simple split on punctuation — matches llm_guard behavior for our use case.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]
