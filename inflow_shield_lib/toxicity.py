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

GPU Extras (vs original):
    ✅ torch.compile(pipeline.model) — fuses RoBERTa kernels at first warmup.
       Uses mode="reduce-overhead" (safe: padding="max_length" gives fixed shapes,
       allowing CUDA graph capture for maximum throughput).
       Set INFLOW_NO_COMPILE=1 to disable.
"""
import logging
import os
import threading
from .utils import calculate_risk_score

logger = logging.getLogger(__name__)

_MODEL_PATH = "unitary/unbiased-toxic-roberta"

_TOXIC_LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

_pipeline      = None
_pipeline_lock = threading.Lock()


def _resolve_device() -> int | str:
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
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"[Toxicity] GPU detected: {torch.cuda.get_device_name(0)} — using CUDA")
            return 0
        else:
            logger.info("[Toxicity] No GPU detected — using CPU")
            return -1
    except ImportError:
        return -1


def _get_pipeline():
    """
    Load the transformers pipeline once and cache it.
    Applies torch.compile() to pipeline.model on GPU.

    Toxicity uses padding="max_length" → all inputs are shape [batch, 512].
    Fixed shapes allow mode="reduce-overhead" which uses CUDA graph capture
    for maximum throughput (better than "default" which recompiles per shape).

    Thread-safe via lock.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        device       = _resolve_device()
        device_label = "GPU:0" if device == 0 else "CPU"
        logger.info(f"[Toxicity] Loading model: {_MODEL_PATH} on {device_label}")

        try:
            from transformers import pipeline as hf_pipeline
            import torch

            torch_dtype = torch.float16 if device == 0 else torch.float32

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
            logger.info(f"[Toxicity] ✅ Model loaded on {device_label} (dtype={torch_dtype})")

            # ── torch.compile — fuse RoBERTa kernels with CUDA graph capture ─
            # mode="reduce-overhead": padding="max_length" guarantees fixed input
            # shape [1, 512], so CUDA graphs can be captured → best throughput.
            # Disable with: INFLOW_NO_COMPILE=1
            if device == 0 and not os.getenv("INFLOW_NO_COMPILE"):
                if hasattr(torch, "compile"):
                    try:
                        _pipeline.model = torch.compile(
                            _pipeline.model,
                            mode="reduce-overhead",
                            fullgraph=False,
                        )
                        logger.info("[Toxicity] ✅ torch.compile applied (mode=reduce-overhead, CUDA graphs)")
                    except Exception as e:
                        logger.warning(f"[Toxicity] torch.compile skipped: {e}")
                else:
                    logger.info("[Toxicity] torch.compile not available (PyTorch < 2.0)")

        except Exception as e:
            logger.error(f"[Toxicity] Failed to load model: {e}")
            raise

    return _pipeline


class Toxicity:
    """
    Detects toxic content using unitary's unbiased-toxic-roberta model.

    Drop-in replacement for llm_guard.input_scanners.Toxicity.
    Same scan() interface: returns (sanitized_prompt, is_valid, risk_score)

    GPU acceleration is automatic when CUDA is available.
    torch.compile() is applied automatically on GPU for maximum throughput.

    To use a specific CUDA stream (for parallel GPU execution with other scanners),
    set the stream context BEFORE calling scan():
        with torch.cuda.stream(my_stream):
            _, is_valid, score = scanner.scan(prompt)

    Force CPU:         INFLOW_DEVICE=cpu
    Disable compile:   INFLOW_NO_COMPILE=1
    """

    def __init__(self, *, threshold: float = 0.5):
        self._threshold = threshold
        _get_pipeline()

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        """
        Scan prompt for toxic content.

        Returns:
            (prompt, is_valid, risk_score)
            - is_valid=True  → no toxicity detected
            - is_valid=False → toxicity detected, block this prompt
            - risk_score     → 0.0 to 1.0 normalized score

        GPU stream note: wrap this call in torch.cuda.stream(stream)
        from the caller for parallel execution alongside other scanners.
        """
        if not prompt or prompt.strip() == "":
            return prompt, True, -1.0

        try:
            pipe         = _get_pipeline()
            results_all  = pipe([prompt])

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