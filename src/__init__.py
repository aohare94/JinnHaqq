#!/usr/bin/env python3
"""
Qur'anic AI Alignment Project v2.0
─────────────────────────────────────────────────────────────────────────────
A comprehensive system for AI alignment through Qur'anic structural analysis
─────────────────────────────────────────────────────────────────────────────
"""

__version__ = "2.0.0"
__author__ = "Qur'anic AI Alignment Research"
__description__ = "AI alignment through Qur'anic structural discovery and recursive understanding"

from .core.engine import EnhancedLLMEngine, create_engine
from .core.config import AlignmentConfig, get_default_config
from .quran.processor import QuranProcessor
from .utils.gpu import GPUManager
from .utils.arabic import ArabicHandler

__all__ = [
    "EnhancedLLMEngine",
    "create_engine", 
    "AlignmentConfig",
    "get_default_config",
    "QuranProcessor",
    "GPUManager", 
    "ArabicHandler"
]