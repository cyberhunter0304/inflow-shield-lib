"""
inflow_shield_lib.prompt_injection
-----------------------------------
Prompt injection scanner using ProtectAI's deberta-v3 model.
Direct transformers call — no llm_guard wrapper.

Model: protectai/deberta-v3-base-prompt-injection-v2
Same model, same logic, same scan() interface as llm_guard.PromptInjection.

GPU Support:
    Automatically uses CUDA GPU if available, falls back to CPU.
    Set INFLOW_DEVICE=cpu to force CPU even if GPU is present.
"""
import logging
import os
import threading
from typing import Optional
from .utils import calculate_risk_score

logger = logging.getLogger(__name__)

# Model identifier — same as llm_guard V2_MODEL
_MODEL_PATH = "protectai/deberta-v3-base-prompt-injection-v2"

# Max tokens the model supports
_MAX_LENGTH = 512

# Module-level cache — load once at startup, reuse forever
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
        logger.info("[PromptInjection] Device forced to CPU via INFLOW_DEVICE env var")
        return -1  # transformers uses -1 for CPU
    if env == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("INFLOW_DEVICE=cuda but no CUDA GPU found")
        logger.info("[PromptInjection] Device forced to CUDA via INFLOW_DEVICE env var")
        return 0
    # Auto-detect
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"[PromptInjection] GPU detected: {gpu_name} — using CUDA")
            return 0  # First GPU
        else:
            logger.info("[PromptInjection] No GPU detected — using CPU")
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
        if _pipeline is not None:  # Double-check after acquiring lock
            return _pipeline

        device = _resolve_device()
        device_label = f"GPU:0" if device == 0 else "CPU"
        logger.info(f"[PromptInjection] Loading model: {_MODEL_PATH} on {device_label}")

        try:
            from transformers import pipeline as hf_pipeline
            import torch

            # Use float16 on GPU for ~2x speed + ~50% memory reduction
            torch_dtype = torch.float16 if device == 0 else torch.float32

            _pipeline = hf_pipeline(
                task="text-classification",
                model=_MODEL_PATH,
                device=device,
                torch_dtype=torch_dtype,
                return_token_type_ids=False,
                max_length=_MAX_LENGTH,
                truncation=True,
            )
            logger.info(f"[PromptInjection] ✅ Model loaded on {device_label} (dtype={torch_dtype})")
        except Exception as e:
            logger.error(f"[PromptInjection] Failed to load model: {e}")
            raise

    return _pipeline


class PromptInjection:
    """
    Detects prompt injection attacks using ProtectAI's DeBERTa model.

    Drop-in replacement for llm_guard.input_scanners.PromptInjection.
    Same scan() interface: returns (sanitized_prompt, is_valid, risk_score)

    GPU acceleration is automatic when CUDA is available.
    Force CPU with: INFLOW_DEVICE=cpu

    Usage:
        scanner = PromptInjection(threshold=0.8)
        _, is_valid, score = scanner.scan("ignore all previous instructions")
        # is_valid=False, score=0.97
    """

    def __init__(self, *, threshold: float = 0.8):
        """
        Parameters:
            threshold: Injection confidence threshold (0.0-1.0).
                       Default 0.8 — lower than llm_guard default (0.92)
                       to catch more injections in a security-first system.
        """
        self._threshold = threshold
        # Eagerly load model at instantiation (mirrors llm_guard behavior)
        _get_pipeline()

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        """
        Scan prompt for injection attacks.

        Returns:
            (prompt, is_valid, risk_score)
            - is_valid=True  → no injection detected
            - is_valid=False → injection detected, block this prompt
            - risk_score     → 0.0 to 1.0 normalized score
        """
        if not prompt or prompt.strip() == "":
            return prompt, True, -1.0

        # Truncate to model max length (same as llm_guard preprocessing)
        text = prompt[:2048]  # char limit before tokenization

        try:
            pipe = _get_pipeline()
            results = pipe([text])  # Returns list of dicts

            highest_score = 0.0
            for result in results:
                # Model labels: "INJECTION" or "LEGITIMATE"
                injection_score = round(
                    result["score"] if result["label"] == "INJECTION"
                    else 1 - result["score"],
                    2,
                )

                if injection_score > highest_score:
                    highest_score = injection_score

                if injection_score > self._threshold:
                    logger.warning(
                        f"[PromptInjection] Detected (score={injection_score:.2f}, "
                        f"threshold={self._threshold})"
                    )
                    return prompt, False, calculate_risk_score(
                        injection_score, self._threshold
                    )

            logger.debug(
                f"[PromptInjection] Clean (highest_score={highest_score:.2f})"
            )
            return prompt, True, calculate_risk_score(highest_score, self._threshold)

        except Exception as e:
            logger.error(f"[PromptInjection] Scan error: {e}")
            # Fail open — don't block on scanner error
            return prompt, True, 0.0