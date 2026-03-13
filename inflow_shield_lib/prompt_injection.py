"""
inflow_shield_lib.prompt_injection
-----------------------------------
Prompt injection scanner using ProtectAI's deberta-v3 model.

GPU path (default on CUDA):
    Uses ONNX Runtime with CUDAExecutionProvider — 3–5x faster than
    PyTorch eager mode, no torch.compile recompilation variance.
    Input is padded to max_length=512 so shapes are always fixed,
    enabling CUDA graph capture inside ORT.

CPU fallback:
    Standard HuggingFace pipeline when CUDA is not available.

Set INFLOW_DEVICE=cpu to force CPU even if GPU is present.
Set INFLOW_NO_ORT=1  to skip ONNX Runtime and use PyTorch instead.
"""
import logging
import os
import threading
from .utils import calculate_risk_score

logger = logging.getLogger(__name__)

_MODEL_PATH = "protectai/deberta-v3-base-prompt-injection-v2"
_MAX_LENGTH  = 512

_pipeline      = None
_pipeline_lock = threading.Lock()


def _resolve_device() -> int:
    env = os.getenv("INFLOW_DEVICE", "auto").lower()
    if env == "cpu":
        logger.info("[PromptInjection] Device forced to CPU via INFLOW_DEVICE")
        return -1
    if env == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("INFLOW_DEVICE=cuda but no CUDA GPU found")
        return 0
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"[PromptInjection] GPU detected: {torch.cuda.get_device_name(0)}")
            return 0
        return -1
    except ImportError:
        return -1


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        device = _resolve_device()
        use_ort = device == 0 and not os.getenv("INFLOW_NO_ORT")

        if use_ort:
            # ── ONNX Runtime path ────────────────────────────────────────────
            # ORTModelForSequenceClassification runs with CUDAExecutionProvider,
            # uses fixed-shape inputs, and avoids all torch.compile variance.
            # export=True auto-converts the HuggingFace model to ONNX on first run.
            try:
                from optimum.onnxruntime import ORTModelForSequenceClassification
                from transformers import AutoTokenizer, pipeline as hf_pipeline

                logger.info(f"[PromptInjection] Loading via ONNX Runtime (CUDAExecutionProvider)...")
                model     = ORTModelForSequenceClassification.from_pretrained(
                    _MODEL_PATH,
                    export=True,
                    provider="CUDAExecutionProvider",
                )
                tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH)
                _pipeline = hf_pipeline(
                    task="text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device="cuda:0",
                    padding="max_length",   # fixed shape → no recompile
                    truncation=True,
                    max_length=_MAX_LENGTH,
                    return_token_type_ids=False,
                )
                logger.info("[PromptInjection] ✅ ONNX Runtime loaded (CUDAExecutionProvider, fixed shape)")
                return _pipeline
            except Exception as e:
                logger.warning(f"[PromptInjection] ONNX Runtime failed ({e}) — falling back to PyTorch")

        # ── PyTorch fallback ─────────────────────────────────────────────────
        from transformers import pipeline as hf_pipeline
        import torch

        torch_dtype  = torch.float16 if device == 0 else torch.float32
        device_label = "GPU:0" if device == 0 else "CPU"
        logger.info(f"[PromptInjection] Loading PyTorch pipeline on {device_label} (dtype={torch_dtype})")

        _pipeline = hf_pipeline(
            task="text-classification",
            model=_MODEL_PATH,
            device=device,
            torch_dtype=torch_dtype,
            padding="max_length",   # fixed [1,512] shape → enables CUDA graph capture
            truncation=True,
            max_length=_MAX_LENGTH,
            return_token_type_ids=False,
        )

        # torch.compile with reduce-overhead — safe now that shapes are fixed
        if device == 0 and not os.getenv("INFLOW_NO_COMPILE") and hasattr(torch, "compile"):
            try:
                _pipeline.model = torch.compile(_pipeline.model, mode="reduce-overhead", fullgraph=False)
                logger.info("[PromptInjection] ✅ torch.compile applied (reduce-overhead)")
            except Exception as e:
                logger.warning(f"[PromptInjection] torch.compile skipped: {e}")

        logger.info(f"[PromptInjection] ✅ PyTorch pipeline ready on {device_label}")
    return _pipeline


class PromptInjection:
    """
    Detects prompt injection attacks using ProtectAI's DeBERTa model.
    GPU: ONNX Runtime with CUDAExecutionProvider (default, fastest).
    CPU fallback: HuggingFace pipeline.

    Force CPU:       INFLOW_DEVICE=cpu
    Skip ORT:        INFLOW_NO_ORT=1
    Skip compile:    INFLOW_NO_COMPILE=1
    """

    def __init__(self, *, threshold: float = 0.8):
        self._threshold = threshold
        _get_pipeline()

    def scan(self, prompt: str) -> tuple[str, bool, float]:
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
                    else 1 - result["score"], 2
                )
                if injection_score > highest_score:
                    highest_score = injection_score
                if injection_score > self._threshold:
                    logger.warning(
                        f"[PromptInjection] Detected (score={injection_score:.2f}, "
                        f"threshold={self._threshold})"
                    )
                    return prompt, False, calculate_risk_score(injection_score, self._threshold)

            logger.debug(f"[PromptInjection] Clean (highest={highest_score:.2f})")
            return prompt, True, calculate_risk_score(highest_score, self._threshold)

        except Exception as e:
            logger.error(f"[PromptInjection] Scan error: {e}")
            return prompt, True, 0.0