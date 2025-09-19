"""
Modular Trading System

This package provides a modular architecture for the trading system where
each component (technical indicators, sentiment analysis, regime detection, etc.)
is implemented as an independent, testable module.

Key Features:
- Module isolation and testing
- Dependency resolution
- Performance monitoring
- Auto-optimization
"""

from .base import BaseModule, ModuleOutput, ModuleContract
from .registry import ModuleRegistry
from .pipeline import ModulePipeline

__all__ = [
    'BaseModule',
    'ModuleOutput',
    'ModuleContract',
    'ModuleRegistry',
    'ModulePipeline'
]