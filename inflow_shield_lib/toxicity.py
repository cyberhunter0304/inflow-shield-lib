"""
inflow_shield_lib.toxicity
---------------------------
Toxicity scanner using unitary's unbiased-toxic-roberta model.

GPU path (default on CUDA):
    ONNX Runtime with CUDAExecutionProvider.
    padding="max_length" ensures fixed [1,512] input shapes for CUDA graphs.

CPU fallback:
    Standard HuggingFace pipeline.

Set INFLOW_DEVICE=cpu to force CPU.
Set INFLOW_NO_ORT=1  to skip ONNX Runtime.
"""
import logging
import os
import threading
from .utils import calculate_risk_score

logger = logging.getLogger(__name__)

_MODEL_PATH = "unitary/unbiased-toxic-roberta"

_TOXIC_LABELS = [
    "toxicity", "severe_toxicity", "obscene",
    "threat", "insult", "identity_attack", "sexual_explicit",
]

_pipeline      = None
_pipeline_lock = threading.Lock()


def _resolve_device() -> int:
    env = os.getenv("INFLOW_DEVICE", "auto").lower()
    if env == "cpu":
        logger.info("[Toxicity] Device forced to CPU via INFLOW_DEVICE")
        return -1
    if env == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("INFLOW_DEVICE=cuda but no CUDA GPU found")
        return 0
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"[Toxicity] GPU detected: {torch.cuda.get_device_name(0)}")
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

        device  = _resolve_device()
        use_ort = device == 0 and not os.getenv("INFLOW_NO_ORT")

        if use_ort:
            try:
                from optimum.onnxruntime import ORTModelForSequenceClassification
                from transformers import AutoTokenizer, pipeline as hf_pipeline

                logger.info("[Toxicity] Loading via ONNX Runtime (CUDAExecutionProvider)...")
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
                    padding="max_length",
                    top_k=None,
                    function_to_apply="sigmoid",
                    return_token_type_ids=False,
                    max_length=512,
                    truncation=True,
                )
                logger.info("[Toxicity] ✅ ONNX Runtime loaded (CUDAExecutionProvider)")
                return _pipeline
            except Exception as e:
                logger.warning(f"[Toxicity] ONNX Runtime failed ({e}) — falling back to PyTorch")

        # ── PyTorch fallback ─────────────────────────────────────────────────
        from transformers import pipeline as hf_pipeline
        import torch

        torch_dtype  = torch.float16 if device == 0 else torch.float32
        device_label = "GPU:0" if device == 0 else "CPU"
        logger.info(f"[Toxicity] Loading PyTorch pipeline on {device_label} (dtype={torch_dtype})")

        _pipeline = hf_pipeline(
            task="text-classification",
            model=_MODEL_PATH,
            device=device,
            torch_dtype=torch_dtype,
            padding="max_length",
            top_k=None,
            function_to_apply="sigmoid",
            return_token_type_ids=False,
            max_length=512,
            truncation=True,
        )

        if device == 0 and not os.getenv("INFLOW_NO_COMPILE") and hasattr(torch, "compile"):
            try:
                _pipeline.model = torch.compile(_pipeline.model, mode="reduce-overhead", fullgraph=False)
                logger.info("[Toxicity] ✅ torch.compile applied (reduce-overhead, CUDA graphs)")
            except Exception as e:
                logger.warning(f"[Toxicity] torch.compile skipped: {e}")

        logger.info(f"[Toxicity] ✅ PyTorch pipeline ready on {device_label}")
    return _pipeline


class Toxicity:
    """
    Detects toxic content using unitary's unbiased-toxic-roberta model.
    GPU: ONNX Runtime (default). CPU fallback: HuggingFace pipeline.

    Force CPU:    INFLOW_DEVICE=cpu
    Skip ORT:     INFLOW_NO_ORT=1
    """

    def __init__(self, *, threshold: float = 0.5):
        self._threshold = threshold
        _get_pipeline()

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        if not prompt or prompt.strip() == "":
            return prompt, True, -1.0

        try:
            pipe        = _get_pipeline()
            results_all = pipe([prompt])

            highest_score = 0.0
            toxic_hits    = []

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
                return prompt, False, calculate_risk_score(highest_score, self._threshold)

            logger.debug(f"[Toxicity] Clean (highest={highest_score:.2f})")
            return prompt, True, calculate_risk_score(highest_score, self._threshold)

        except Exception as e:
            logger.error(f"[Toxicity] Scan error: {e}")
            return prompt, True, 0.0