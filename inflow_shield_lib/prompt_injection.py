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

GPU Extras (vs original):
    ✅ torch.compile(pipeline.model) — fuses DeBERTa kernels at first warmup
       Uses mode="default" (safe for variable-length inputs).
       Set INFLOW_NO_COMPILE=1 to disable if compile causes issues.
"""
import logging
import os
import threading
from typing import Optional
from .utils import calculate_risk_score

logger = logging.getLogger(__name__)

_MODEL_PATH = "protectai/deberta-v3-base-prompt-injection-v2"
_MAX_LENGTH  = 512

_pipeline      = None
_pipeline_lock = threading.Lock()


def _resolve_device() -> int | str:
    env = os.getenv("INFLOW_DEVICE", "auto").lower()
    if env == "cpu":
        logger.info("[PromptInjection] Device forced to CPU via INFLOW_DEVICE env var")
        return -1
    if env == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("INFLOW_DEVICE=cuda but no CUDA GPU found")
        logger.info("[PromptInjection] Device forced to CUDA via INFLOW_DEVICE env var")
        return 0
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"[PromptInjection] GPU detected: {torch.cuda.get_device_name(0)} — using CUDA")
            return 0
        else:
            logger.info("[PromptInjection] No GPU detected — using CPU")
            return -1
    except ImportError:
        return -1


def _get_pipeline():
    """
    Load the transformers pipeline once and cache it.
    Applies torch.compile() to pipeline.model on GPU for fused kernels.
    Thread-safe via lock.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        device      = _resolve_device()
        device_label = "GPU:0" if device == 0 else "CPU"
        logger.info(f"[PromptInjection] Loading model: {_MODEL_PATH} on {device_label}")

        try:
            from transformers import pipeline as hf_pipeline
            import torch

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

            # ── torch.compile — fuse DeBERTa attention + FFN kernels ─────────
            # mode="default": safe for variable-length inputs (no CUDA graphs).
            # First scan() call will be slow (~10–30s) as kernels compile —
            # this is why run_warmup() exists. Subsequent calls get the speedup.
            # Disable with: INFLOW_NO_COMPILE=1
            if device == 0 and not os.getenv("INFLOW_NO_COMPILE"):
                if hasattr(torch, "compile"):
                    try:
                        _pipeline.model = torch.compile(
                            _pipeline.model,
                            mode="default",
                            fullgraph=False,
                        )
                        logger.info("[PromptInjection] ✅ torch.compile applied (mode=default)")
                    except Exception as e:
                        logger.warning(f"[PromptInjection] torch.compile skipped: {e}")
                else:
                    logger.info("[PromptInjection] torch.compile not available (PyTorch < 2.0)")

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
    torch.compile() is applied automatically on GPU for fused kernels.

    To use a specific CUDA stream (for parallel GPU execution with other scanners),
    set the stream context BEFORE calling scan():
        with torch.cuda.stream(my_stream):
            _, is_valid, score = scanner.scan(prompt)

    Force CPU:         INFLOW_DEVICE=cpu
    Disable compile:   INFLOW_NO_COMPILE=1
    """

    def __init__(self, *, threshold: float = 0.8):
        self._threshold = threshold
        _get_pipeline()

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        """
        Scan prompt for injection attacks.

        Returns:
            (prompt, is_valid, risk_score)
            - is_valid=True  → no injection detected
            - is_valid=False → injection detected, block this prompt
            - risk_score     → 0.0 to 1.0 normalized score

        GPU stream note: wrap this call in torch.cuda.stream(stream)
        from the caller for parallel execution alongside other scanners.
        """
        if not prompt or prompt.strip() == "":
            return prompt, True, -1.0

        text = prompt[:2048]

        try:
            pipe    = _get_pipeline()
            results = pipe([text])

            highest_score = 0.0
            for result in results:
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
                    return prompt, False, calculate_risk_score(injection_score, self._threshold)

            logger.debug(f"[PromptInjection] Clean (highest_score={highest_score:.2f})")
            return prompt, True, calculate_risk_score(highest_score, self._threshold)

        except Exception as e:
            logger.error(f"[PromptInjection] Scan error: {e}")
            return prompt, True, 0.0