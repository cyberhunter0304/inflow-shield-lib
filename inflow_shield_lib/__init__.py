"""
inflow_shield_lib
-----------------
Lightweight guardrails library extracted from inFlow Shield.
Replaces llm-guard with only the scanners we actually use.

Usage:
    from inflow_shield_lib import PromptInjection, Toxicity, Secrets, Vault

No llm-guard dependency. Same scan() interface:
    sanitized, is_valid, risk_score = scanner.scan(prompt)
"""

from .vault import Vault
from .prompt_injection import PromptInjection
from .toxicity import Toxicity
from .secrets import Secrets

__all__ = ["Vault", "PromptInjection", "Toxicity", "Secrets"]
__version__ = "1.0.0"
