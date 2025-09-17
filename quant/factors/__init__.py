"""
Factor Profile System for Stock-Specific Signal Weighting

This package implements stock-specific factor profiles that recognize
different stocks are driven by different fundamental factors.
"""

from .profile_engine import StockFactorProfileEngine, FactorProfile

__all__ = ['StockFactorProfileEngine', 'FactorProfile']