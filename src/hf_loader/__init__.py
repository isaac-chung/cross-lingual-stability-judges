"""
HuggingFace dataset loader module.

Converts HuggingFace datasets back to the combined JSON format
expected by the analysis tools (llm_judge, label_recovery).
"""

from .loader import HFLoader

__all__ = ["HFLoader"]
