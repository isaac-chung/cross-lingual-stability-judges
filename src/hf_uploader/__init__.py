"""
HuggingFace Dataset Upload Module

This module provides functionality to upload conversation files to HuggingFace Hub
as structured datasets organized by model with language subsets.
"""

from .config import HFConfig
from .auth import HFAuthenticator
from .dataset_builder import DatasetBuilder
from .uploader import HFUploader

__version__ = "1.0.0"
__all__ = ["HFConfig", "HFAuthenticator", "DatasetBuilder", "HFUploader"]