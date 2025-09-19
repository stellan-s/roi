"""
Module execution pipeline for orchestrating the trading system modules.

The pipeline executes modules in dependency order, handles data flow between modules,
and provides monitoring and debugging capabilities.
"""

from typing import Dict, List, Any, Optional
import time
import logging
from datetime import datetime
import json

from .base import BaseModule, ModuleOutput

logger = logging.getLogger(__name__)

class PipelineExecutionError(Exception):
    """Raised when pipeline execution fails"""
    pass

class ModulePipeline:
    """Executes modules in correct order with dependency resolution"""

    def __init__(self, modules: Dict[str, BaseModule], execution_order: List[str]):
        self.modules = modules
        self.execution_order = execution_order
        self.results: Dict[str, ModuleOutput] = {}
        self.execution_history = []
        self.current_execution_id = None

    def execute(self, initial_inputs: Dict[str, Any]) -> Dict[str, ModuleOutput]:
        """Execute all modules in pipeline"""
        execution_id = f"exec_{int(time.time())}"
        self.current_execution_id = execution_id

        execution_start = time.time()
        execution_log = {
            "execution_id": execution_id,
            "start_time": datetime.now().isoformat(),
            "modules_executed": [],
            "total_time_ms": 0,
            "status": "running"
        }

        try:
            logger.info(f"🚀 Starting pipeline execution {execution_id}")
            logger.info(f"📋 Execution order: {' → '.join(self.execution_order)}")

            # Clear previous results
            self.results = {}
            current_data = initial_inputs.copy()

            # Execute each module in order
            for i, module_name in enumerate(self.execution_order):
                module = self.modules[module_name]

                logger.info(f"[{i+1}/{len(self.execution_order)}] Executing {module_name}...")

                if not module.enabled:
                    logger.info(f"⏭️  Skipping {module_name} (disabled)")
                    execution_log["modules_executed"].append({
                        "module": module_name,
                        "status": "skipped",
                        "reason": "disabled"
                    })
                    continue

                # Prepare inputs for this module
                module_inputs = self._prepare_module_inputs(module, current_data)

                # Execute module
                module_start = time.time()
                try:
                    result = module.execute(module_inputs)
                    module_time = (time.time() - module_start) * 1000

                    # Store results and update current_data
                    self.results[module_name] = result
                    current_data[module_name] = result.data

                    # Log successful execution
                    logger.info(f"✅ {module_name} completed in {module_time:.1f}ms "
                              f"(confidence: {result.confidence:.2f})")

                    execution_log["modules_executed"].append({
                        "module": module_name,
                        "status": "success",
                        "execution_time_ms": module_time,
                        "confidence": result.confidence,
                        "output_keys": list(result.data.keys())
                    })

                except Exception as e:
                    module_time = (time.time() - module_start) * 1000
                    error_msg = f"Module {module_name} failed: {str(e)}"

                    logger.error(f"❌ {error_msg} (after {module_time:.1f}ms)")

                    execution_log["modules_executed"].append({
                        "module": module_name,
                        "status": "error",
                        "execution_time_ms": module_time,
                        "error": str(e)
                    })

                    # Decide whether to continue or fail
                    if self._is_critical_module(module_name):
                        execution_log["status"] = "failed"
                        self.execution_history.append(execution_log)
                        raise PipelineExecutionError(error_msg)
                    else:
                        logger.warning(f"⚠️  Continuing pipeline despite {module_name} failure")

            # Pipeline completed successfully
            total_time = (time.time() - execution_start) * 1000
            execution_log["total_time_ms"] = total_time
            execution_log["status"] = "completed"
            execution_log["end_time"] = datetime.now().isoformat()

            logger.info(f"🎉 Pipeline execution {execution_id} completed in {total_time:.1f}ms")

            self.execution_history.append(execution_log)
            return self.results

        except Exception as e:
            total_time = (time.time() - execution_start) * 1000
            execution_log["total_time_ms"] = total_time
            execution_log["status"] = "failed"
            execution_log["error"] = str(e)
            execution_log["end_time"] = datetime.now().isoformat()

            self.execution_history.append(execution_log)
            raise

    def health_check(self) -> Dict[str, Any]:
        """Run health checks on all modules in the pipeline"""
        logger.info("🏥 Running pipeline health check...")

        health_status = {
            "overall_status": "HEALTHY",
            "modules": {},
            "pipeline_info": {
                "total_modules": len(self.modules),
                "enabled_modules": sum(1 for m in self.modules.values() if m.enabled),
                "execution_order": self.execution_order
            }
        }

        unhealthy_count = 0

        for module_name in self.execution_order:
            module = self.modules[module_name]

            try:
                module_health = module.health_check()
                health_status["modules"][module_name] = module_health

                if module_health["status"] not in ["HEALTHY", "DEGRADED"]:
                    unhealthy_count += 1

                logger.info(f"🔍 {module_name}: {module_health['status']}")

            except Exception as e:
                health_status["modules"][module_name] = {
                    "status": "ERROR",
                    "error": str(e)
                }
                unhealthy_count += 1
                logger.error(f"❌ {module_name}: Health check failed - {e}")

        # Determine overall status
        if unhealthy_count == 0:
            health_status["overall_status"] = "HEALTHY"
        elif unhealthy_count < len(self.modules) / 2:
            health_status["overall_status"] = "DEGRADED"
        else:
            health_status["overall_status"] = "UNHEALTHY"

        logger.info(f"🏥 Overall pipeline health: {health_status['overall_status']}")
        return health_status

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of recent pipeline executions"""
        if not self.execution_history:
            return {"message": "No executions recorded"}

        recent_executions = self.execution_history[-10:]  # Last 10 executions

        summary = {
            "total_executions": len(self.execution_history),
            "recent_executions": len(recent_executions),
            "success_rate": sum(1 for e in recent_executions if e["status"] == "completed") / len(recent_executions),
            "avg_execution_time_ms": sum(e["total_time_ms"] for e in recent_executions) / len(recent_executions),
            "most_recent": recent_executions[-1] if recent_executions else None
        }

        # Module performance summary
        module_stats = {}
        for execution in recent_executions:
            for module_info in execution["modules_executed"]:
                module_name = module_info["module"]
                if module_name not in module_stats:
                    module_stats[module_name] = {
                        "executions": 0,
                        "successes": 0,
                        "total_time_ms": 0,
                        "avg_confidence": 0
                    }

                module_stats[module_name]["executions"] += 1
                if module_info["status"] == "success":
                    module_stats[module_name]["successes"] += 1
                    module_stats[module_name]["total_time_ms"] += module_info.get("execution_time_ms", 0)
                    module_stats[module_name]["avg_confidence"] += module_info.get("confidence", 0)

        # Calculate averages
        for stats in module_stats.values():
            if stats["successes"] > 0:
                stats["avg_time_ms"] = stats["total_time_ms"] / stats["successes"]
                stats["avg_confidence"] = stats["avg_confidence"] / stats["successes"]
                stats["success_rate"] = stats["successes"] / stats["executions"]

        summary["module_performance"] = module_stats
        return summary

    def _prepare_module_inputs(self, module: BaseModule, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs for a specific module based on its contract"""
        contract = module.contract
        module_inputs = {}

        # Add required inputs
        for input_name in contract.input_schema.keys():
            if input_name in current_data:
                module_inputs[input_name] = current_data[input_name]
            else:
                # Try to find input from previous module outputs
                for previous_module, output in self.results.items():
                    if input_name in output.data:
                        module_inputs[input_name] = output.data[input_name]
                        break
                else:
                    raise PipelineExecutionError(
                        f"Required input '{input_name}' not available for module {module.contract.name}"
                    )

        # Add optional inputs if available
        for optional_input in contract.optional_inputs:
            if optional_input in current_data:
                module_inputs[optional_input] = current_data[optional_input]

        return module_inputs

    def _is_critical_module(self, module_name: str) -> bool:
        """Determine if a module is critical (pipeline should fail if it fails)"""
        # For now, consider all modules critical
        # This could be made configurable per module
        return True

    def save_execution_log(self, filepath: str):
        """Save execution history to file"""
        with open(filepath, 'w') as f:
            json.dump({
                "execution_history": self.execution_history,
                "pipeline_info": {
                    "modules": list(self.modules.keys()),
                    "execution_order": self.execution_order
                }
            }, f, indent=2)

    def get_module_data_flow(self) -> Dict[str, Any]:
        """Analyze data flow between modules"""
        if not self.results:
            return {"message": "No execution results available"}

        flow = {
            "modules": {},
            "data_dependencies": {}
        }

        for module_name, result in self.results.items():
            flow["modules"][module_name] = {
                "outputs": list(result.data.keys()),
                "confidence": result.confidence,
                "execution_time_ms": result.execution_time_ms
            }

        # Analyze which modules consume outputs from other modules
        for module_name, module in self.modules.items():
            contract = module.contract
            dependencies = []

            for input_name in contract.input_schema.keys():
                for producer_module, result in self.results.items():
                    if input_name in result.data:
                        dependencies.append({
                            "input": input_name,
                            "producer": producer_module
                        })

            if dependencies:
                flow["data_dependencies"][module_name] = dependencies

        return flow

    def __len__(self):
        return len(self.modules)

    def __contains__(self, module_name: str):
        return module_name in self.modules

    def __repr__(self):
        enabled_count = sum(1 for m in self.modules.values() if m.enabled)
        return f"ModulePipeline({enabled_count}/{len(self.modules)} modules enabled)"