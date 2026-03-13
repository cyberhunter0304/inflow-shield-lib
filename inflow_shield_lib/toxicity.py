"""
inflow_shield_lib.toxicity
---------------------------
Toxicity scanner using unitary's unbiased-toxic-roberta model.
Direct transformers call — no llm_guard wrapper.

Model: unitary/unbiased-toxic-roberta
Same model, same logic, same scan() interface as llm_guard.Toxicity.

GPU Support:
    Automatically uses CUDA GPU if available, falls back to CPU.
    Set INFLOW_DEVICE=cpu to force CPU even if GPU is present.
"""
import logging
import os
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


def _resolve_device() -> int | str:
    """
    Resolve which device to run inference on.
    - INFLOW_DEVICE=cpu  → force CPU
    - INFLOW_DEVICE=cuda → force GPU (raises if not available)
    - default            → GPU if available, else CPU
    """
    env = os.getenv("INFLOW_DEVICE", "auto").lower()
    if env == "cpu":
        logger.info("[Toxicity] Device forced to CPU via INFLOW_DEVICE env var")
        return -1
    if env == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("INFLOW_DEVICE=cuda but no CUDA GPU found")
        logger.info("[Toxicity] Device forced to CUDA via INFLOW_DEVICE env var")
        return 0
    # Auto-detect
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"[Toxicity] GPU detected: {gpu_name} — using CUDA")
            return 0
        else:
            logger.info("[Toxicity] No GPU detected — using CPU")
            return -1
    except ImportError:
        return -1


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

        device = _resolve_device()
        device_label = "GPU:0" if device == 0 else "CPU"
        logger.info(f"[Toxicity] Loading model: {_MODEL_PATH} on {device_label}")

        try:
            from transformers import pipeline as hf_pipeline
            import torch

            # float16 on GPU for ~2x speed + ~50% memory reduction
            torch_dtype = torch.float16 if device == 0 else torch.float32

            _pipeline = hf_pipeline(
                task="text-classification",
                model=_MODEL_PATH,
                device=device,
                torch_dtype=torch_dtype,
                padding="max_length",
                top_k=None,               # Return all labels + scores
                function_to_apply="sigmoid",
                return_token_type_ids=False,
                max_length=512,
                truncation=True,
            )
            logger.info(f"[Toxicity] ✅ Model loaded on {device_label} (dtype={torch_dtype})")
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

    GPU acceleration is automatic when CUDA is available.
    Force CPU with: INFLOW_DEVICE=cpu

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