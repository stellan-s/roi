"""
Base module framework for the modular trading system.

Defines the contracts and interfaces that all modules must implement
to ensure consistency, testability, and composability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd
import time
import numpy as np
from datetime import datetime

@dataclass
class ModuleOutput:
    """Standardized output format for all modules"""
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 1.0
    execution_time_ms: Optional[float] = None

    def __post_init__(self):
        """Validate output after creation"""
        if not isinstance(self.data, dict):
            raise ValueError("ModuleOutput.data must be a dictionary")
        if not 0 <= self.confidence <= 1:
            raise ValueError("ModuleOutput.confidence must be between 0 and 1")

@dataclass
class ModuleContract:
    """Contract that each module must fulfill"""
    name: str
    version: str
    description: str
    input_schema: Dict[str, str]
    output_schema: Dict[str, str]
    performance_sla: Dict[str, float]
    dependencies: List[str] = field(default_factory=list)
    optional_inputs: List[str] = field(default_factory=list)

    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate that inputs match the contract schema"""
        for required_input, expected_type in self.input_schema.items():
            if required_input not in inputs:
                raise ValueError(f"Missing required input: {required_input}")
            # TODO: Add type validation based on expected_type string
        return True

    def validate_outputs(self, output: ModuleOutput) -> bool:
        """Validate that outputs match the contract schema"""
        for required_output in self.output_schema.keys():
            if required_output not in output.data:
                raise ValueError(f"Missing required output: {required_output}")
        return True

class BaseModule(ABC):
    """Base class all trading modules must inherit from"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.contract = self.define_contract()
        self._performance_history = []
        self._test_history = []

    @abstractmethod
    def define_contract(self) -> ModuleContract:
        """Each module must define its contract"""
        pass

    @abstractmethod
    def process(self, inputs: Dict[str, Any]) -> ModuleOutput:
        """Main processing logic - must be implemented by each module"""
        pass

    @abstractmethod
    def test_module(self) -> Dict[str, Any]:
        """Built-in health check - must be implemented by each module"""
        pass

    def execute(self, inputs: Dict[str, Any]) -> ModuleOutput:
        """Execute the module with performance tracking and validation"""
        if not self.enabled:
            return ModuleOutput(
                data={},
                metadata={"status": "disabled"},
                confidence=0.0
            )

        # Validate inputs
        self.contract.validate_inputs(inputs)

        # Execute with timing
        start_time = time.time()
        try:
            result = self.process(inputs)
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            result.execution_time_ms = execution_time

            # Validate outputs
            self.contract.validate_outputs(result)

            # Record performance
            self._record_performance(execution_time, result.confidence)

            return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._record_performance(execution_time, 0.0, error=str(e))
            raise RuntimeError(f"Module {self.contract.name} failed: {e}")

    def benchmark(self, num_runs: int = 10) -> Dict[str, float]:
        """Performance benchmarking with synthetic data"""
        latencies = []
        accuracies = []

        for _ in range(num_runs):
            # Generate test data
            test_inputs = self._generate_test_inputs()

            # Measure performance
            start_time = time.time()
            try:
                result = self.process(test_inputs)
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
                accuracies.append(result.confidence)
            except Exception:
                latencies.append(float('inf'))
                accuracies.append(0.0)

        return {
            "avg_latency_ms": np.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "avg_confidence": np.mean(accuracies),
            "success_rate": sum(1 for l in latencies if l != float('inf')) / num_runs
        }

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            # Run module's own test
            test_result = self.test_module()

            # Run benchmark
            benchmark = self.benchmark(num_runs=5)

            # Check against SLA
            sla_violations = []
            if benchmark["avg_latency_ms"] > self.contract.performance_sla.get("max_latency_ms", float('inf')):
                sla_violations.append(f"Latency too high: {benchmark['avg_latency_ms']:.1f}ms")

            if benchmark["avg_confidence"] < self.contract.performance_sla.get("min_confidence", 0.0):
                sla_violations.append(f"Confidence too low: {benchmark['avg_confidence']:.2f}")

            status = "HEALTHY" if not sla_violations else "DEGRADED"

            return {
                "status": status,
                "test_result": test_result,
                "benchmark": benchmark,
                "sla_violations": sla_violations,
                "performance_history_count": len(self._performance_history)
            }

        except Exception as e:
            return {
                "status": "UNHEALTHY",
                "error": str(e),
                "benchmark": {},
                "sla_violations": [f"Health check failed: {e}"]
            }

    def get_performance_stats(self) -> Dict[str, float]:
        """Get historical performance statistics"""
        if not self._performance_history:
            return {}

        latencies = [p["latency_ms"] for p in self._performance_history]
        confidences = [p["confidence"] for p in self._performance_history]
        errors = [p for p in self._performance_history if p.get("error")]

        return {
            "total_executions": len(self._performance_history),
            "avg_latency_ms": np.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "avg_confidence": np.mean(confidences),
            "error_rate": len(errors) / len(self._performance_history),
            "recent_errors": [e["error"] for e in errors[-5:]]  # Last 5 errors
        }

    @abstractmethod
    def _generate_test_inputs(self) -> Dict[str, Any]:
        """Generate synthetic test data for benchmarking"""
        pass

    def _record_performance(self, latency_ms: float, confidence: float, error: str = None):
        """Record performance metrics"""
        self._performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "latency_ms": latency_ms,
            "confidence": confidence,
            "error": error
        })

        # Keep only last 1000 records to prevent memory bloat
        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-1000:]

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.contract.name}, enabled={self.enabled})"