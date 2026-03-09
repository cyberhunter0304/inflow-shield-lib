"""
inflow_shield_lib.vault
-----------------------
Vault for storing anonymized entity mappings.
Copied directly from llm_guard.vault — it's a plain list wrapper,
no dependencies needed.
"""
from typing import List, Optional, Tuple


class Vault:
    """
    Stores (placeholder, original_value) tuples for PII anonymization.
    Used by Presidio anonymizer to track replacements.

    Example:
        vault = Vault()
        vault.append(("[PERSON_1]", "John Doe"))
        vault.get()  # → [("[PERSON_1]", "John Doe")]
    """

    def __init__(self, tuples: Optional[List[Tuple]] = None):
        self._tuples: List[Tuple] = tuples if tuples is not None else []

    def append(self, new_tuple: Tuple):
        self._tuples.append(new_tuple)

    def extend(self, new_tuples: List[Tuple]):
        self._tuples.extend(new_tuples)

    def remove(self, tuple_to_remove: Tuple):
        self._tuples.remove(tuple_to_remove)

    def get(self) -> List[Tuple]:
        return self._tuples

    def placeholder_exists(self, placeholder: str) -> bool:
        return any(p == placeholder for p, _ in self._tuples)

    def clear(self):
        self._tuples = []

    def __len__(self):
        return len(self._tuples)

    def __repr__(self):
        return f"Vault({len(self._tuples)} entries)"
