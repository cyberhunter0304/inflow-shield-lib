"""
inflow_shield_lib.toxicity
---------------------------
Toxicity scanner using unitary's unbiased-toxic-roberta model.
Direct transformers call — no llm_guard wrapper.

Model: unitary/unbiased-toxic-roberta
Same model, same logic, same scan() interface as llm_guard.Toxicity.
"""
import logging
import threading
from .utils import calculate_risk_score

logger = logging.getLogger(__name__)

# Model identifier — same as llm_guard DEFAULT_MODEL
_MODEL_PATH = "unitary/unbiased-toxic-roberta"

# Toxic labels the model outputs — same list as llm_guard
_TOXIC_LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

# Module-level cache
_pipeline = None
_pipeline_lock = threading.Lock()


def _get_pipeline():
    """
    Load the transformers pipeline once and cache it.
    Thread-safe via lock.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        logger.info(f"[Toxicity] Loading model: {_MODEL_PATH}")
        try:
            from transformers import pipeline as hf_pipeline
            _pipeline = hf_pipeline(
                task="text-classification",
                model=_MODEL_PATH,
                padding="max_length",
                top_k=None,               # Return all labels + scores
                function_to_apply="sigmoid",
                return_token_type_ids=False,
                max_length=512,
                truncation=True,
            )
            logger.info("[Toxicity] ✅ Model loaded and cached")
        except Exception as e:
            logger.error(f"[Toxicity] Failed to load model: {e}")
            raise

    return _pipeline


class Toxicity:
    """
    Detects toxic content using unitary's unbiased-toxic-roberta model.

    Drop-in replacement for llm_guard.input_scanners.Toxicity.
    Same scan() interface: returns (sanitized_prompt, is_valid, risk_score)

    Detects: toxicity, severe_toxicity, obscene, threat,
             insult, identity_attack, sexual_explicit

    Usage:
        scanner = Toxicity(threshold=0.5)
        _, is_valid, score = scanner.scan("I hate you")
        # is_valid=False, score=0.87
    """

    def __init__(self, *, threshold: float = 0.5):
        """
        Parameters:
            threshold: Toxicity score threshold (0.0-1.0).
                       Default 0.5 — same as llm_guard default.
        """
        self._threshold = threshold
        # Eagerly load model at instantiation
        _get_pipeline()

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        """
        Scan prompt for toxic content.

        Returns:
            (prompt, is_valid, risk_score)
            - is_valid=True  → no toxicity detected
            - is_valid=False → toxicity detected, block this prompt
            - risk_score     → 0.0 to 1.0 normalized score
        """
        if not prompt or prompt.strip() == "":
            return prompt, True, -1.0

        try:
            pipe = _get_pipeline()
            # Model returns list of lists (one per input, each with all labels)
            results_all = pipe([prompt])

            highest_score = 0.0
            toxic_hits = []

            for results_chunk in results_all:
                for result in results_chunk:
                    if result["label"] not in _TOXIC_LABELS:
                        continue

                    score = result["score"]

                    if score > highest_score:
                        highest_score = score

                    if score > self._threshold:
                        toxic_hits.append(result)

            if toxic_hits:
                logger.warning(
                    f"[Toxicity] Detected (highest={highest_score:.2f}, "
                    f"hits={len(toxic_hits)}, threshold={self._threshold})"
                )
                return prompt, False, calculate_risk_score(
                    highest_score, self._threshold
                )

            logger.debug(f"[Toxicity] Clean (highest={highest_score:.2f})")
            return prompt, True, calculate_risk_score(highest_score, self._threshold)

        except Exception as e:
            logger.error(f"[Toxicity] Scan error: {e}")
            # Fail open — don't block on scanner error
            return prompt, True, 0.0
