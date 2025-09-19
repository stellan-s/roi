"""
Optimization Engine for Modular Trading System

Automatically discovers optimal module configurations through systematic testing
and genetic algorithm optimization.
"""

from .config_generator import ConfigurationGenerator
from .optimizer import SystemOptimizer
from .evaluator import PerformanceEvaluator

__all__ = [
    'ConfigurationGenerator',
    'SystemOptimizer',
    'PerformanceEvaluator'
]