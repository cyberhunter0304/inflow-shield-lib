"""
inflow_shield_lib.secrets
--------------------------
Secrets scanner using pure regex patterns.
Replaces llm_guard.Secrets (which used detect-secrets + temp files).

No detect-secrets dependency.
No temp file writes.
Same scan() interface: returns (sanitized_prompt, is_valid, risk_score)

Covers: API keys, tokens, passwords, AWS keys, GitHub tokens,
        Google API keys, Bearer tokens, JWTs, private keys, and more.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Secret patterns — covers the most common secret types
# Each tuple: (compiled_regex, secret_type_label)
# ============================================================================
_SECRET_PATTERNS = [
    # Generic API key assignments
    (re.compile(
        r'\b(?:api[_-]?key|apikey|api_secret)\s*[=:]\s*[\'"]?([a-zA-Z0-9_\-]{20,})[\'"]?',
        re.IGNORECASE
    ), "API_KEY"),

    # OpenAI / Anthropic style keys: sk-..., sk-ant-...
    (re.compile(
        r'\b(sk-[a-zA-Z0-9]{20,})',
        re.IGNORECASE
    ), "API_KEY"),

    # Stripe / generic pk/rk/ak prefixed keys
    (re.compile(
        r'\b(?:pk|rk|ak|sk)_(?:live|test)_[a-zA-Z0-9]{16,}',
        re.IGNORECASE
    ), "API_KEY"),

    # GitHub tokens
    (re.compile(
        r'\b(?:ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36,}',
        re.IGNORECASE
    ), "GITHUB_TOKEN"),

    # Google API keys
    (re.compile(
        r'\bAIza[0-9A-Za-z\-_]{35}',
    ), "GOOGLE_API_KEY"),

    # AWS Access Key ID
    (re.compile(
        r'\b(?:AKIA|ASIA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}\b',
    ), "AWS_ACCESS_KEY"),

    # AWS Secret Access Key assignment
    (re.compile(
        r'\baws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*[\'"]?([a-zA-Z0-9/+=]{40})[\'"]?',
        re.IGNORECASE
    ), "AWS_SECRET_KEY"),

    # Bearer tokens
    (re.compile(
        r'\bBearer\s+([a-zA-Z0-9\._\-]{20,})',
        re.IGNORECASE
    ), "BEARER_TOKEN"),

    # JWT tokens (3 base64 segments separated by dots)
    (re.compile(
        r'\beyJ[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+',
    ), "JWT_TOKEN"),

    # Password assignments
    (re.compile(
        r'\b(?:password|passwd|pwd)\s*[=:]\s*[\'"]?([^\s\'">\s]{6,})[\'"]?',
        re.IGNORECASE
    ), "PASSWORD"),

    # Generic secret/token assignments
    (re.compile(
        r'\b(?:secret|token|auth[_-]?token|access[_-]?token)\s*[=:]\s*[\'"]?([a-zA-Z0-9_\-]{8,})[\'"]?',
        re.IGNORECASE
    ), "SECRET_TOKEN"),

    # Private key headers
    (re.compile(
        r'-----BEGIN\s+(?:RSA\s+|EC\s+|DSA\s+|OPENSSH\s+)?PRIVATE\s+KEY-----',
        re.IGNORECASE
    ), "PRIVATE_KEY"),

    # Azure storage connection strings
    (re.compile(
        r'DefaultEndpointsProtocol=https;AccountName=[^;]+;AccountKey=[a-zA-Z0-9+/=]{88}',
        re.IGNORECASE
    ), "AZURE_STORAGE_KEY"),

    # Slack tokens
    (re.compile(
        r'\bxox[baprs]-[a-zA-Z0-9\-]{10,}',
        re.IGNORECASE
    ), "SLACK_TOKEN"),

    # Twilio tokens
    (re.compile(
        r'\bSK[a-f0-9]{32}\b',
    ), "TWILIO_KEY"),

    # Sendgrid keys
    (re.compile(
        r'\bSG\.[a-zA-Z0-9\-_]{22,}\.[a-zA-Z0-9\-_]{22,}',
    ), "SENDGRID_KEY"),

    # HuggingFace tokens
    (re.compile(
        r'\bhf_[a-zA-Z0-9]{30,}',
    ), "HUGGINGFACE_TOKEN"),
]


class Secrets:
    """
    Detects secrets and credentials using regex patterns.

    Drop-in replacement for llm_guard.input_scanners.Secrets.
    No detect-secrets dependency. No temp file I/O. Pure Python.

    Same scan() interface: returns (sanitized_prompt, is_valid, risk_score)

    Usage:
        scanner = Secrets()
        _, is_valid, score = scanner.scan("my api_key = sk-abc123xyz789...")
        # is_valid=False, score=1.0
    """

    def __init__(self, *, redact_mode: str = "all"):
        """
        Parameters:
            redact_mode: How to redact found secrets in the returned text.
                - "all"     → replace with ******  (default)
                - "partial" → show first 2 and last 2 chars
                - "hash"    → replace with md5 hash
        """
        self._redact_mode = redact_mode

    def _redact(self, value: str) -> str:
        if self._redact_mode == "partial":
            return f"{value[:2]}..{value[-2:]}" if len(value) > 4 else "****"
        elif self._redact_mode == "hash":
            import hashlib
            return hashlib.md5(value.encode()).hexdigest()
        return "******"

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        """
        Scan prompt for secrets and credentials.

        Returns:
            (redacted_prompt, is_valid, risk_score)
            - is_valid=True  → no secrets found
            - is_valid=False → secret detected, block this prompt
            - risk_score     → 1.0 if detected, -1.0 if clean
        """
        if not prompt or prompt.strip() == "":
            return prompt, True, -1.0

        found_types = []
        redacted = prompt

        for pattern, secret_type in _SECRET_PATTERNS:
            matches = pattern.findall(prompt)
            if matches:
                found_types.append(secret_type)
                # Redact all matches in the text
                def redact_match(m):
                    full = m.group(0)
                    # If the pattern has a capture group, redact just that
                    if m.lastindex and m.lastindex >= 1:
                        captured = m.group(1)
                        return full.replace(captured, self._redact(captured))
                    return self._redact(full)

                try:
                    redacted = pattern.sub(redact_match, redacted)
                except Exception:
                    redacted = pattern.sub(self._redact("SECRET"), redacted)

        if found_types:
            logger.warning(f"[Secrets] Detected: {found_types}")
            return redacted, False, 1.0

        logger.debug("[Secrets] Clean — no secrets detected")
        return prompt, True, -1.0
