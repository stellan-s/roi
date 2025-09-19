"""
Performance Evaluator for Module Configurations

Evaluates the performance of different module configurations through backtesting
and statistical analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import time

class PerformanceEvaluator:
    """Evaluates module configuration performance through backtesting"""

    def __init__(self, backtest_engine=None):
        self.backtest_engine = backtest_engine
        self.evaluation_history = []

    def evaluate_configuration(self, config: Dict,
                             evaluation_period: Tuple[str, str] = None,
                             target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """Evaluate a single configuration's performance"""

        if evaluation_period is None:
            # Default to last 6 months
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=180)
            evaluation_period = (start_date.isoformat(), end_date.isoformat())

        start_time = time.time()

        try:
            # Run backtest with this configuration
            backtest_result = self._run_backtest_with_config(config, evaluation_period)

            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(backtest_result)

            # Module-specific analysis
            module_analysis = self._analyze_module_contribution(config, backtest_result)

            evaluation_time = time.time() - start_time

            result = {
                "config_fingerprint": self._get_config_fingerprint(config),
                "evaluation_period": evaluation_period,
                "evaluation_time_seconds": evaluation_time,
                "target_metric": target_metric,
                "target_score": metrics.get(target_metric, 0),
                "metrics": metrics,
                "module_analysis": module_analysis,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }

            self.evaluation_history.append(result)
            return result

        except Exception as e:
            evaluation_time = time.time() - start_time

            result = {
                "config_fingerprint": self._get_config_fingerprint(config),
                "evaluation_period": evaluation_period,
                "evaluation_time_seconds": evaluation_time,
                "target_metric": target_metric,
                "target_score": -999,  # Penalty for failed configs
                "metrics": {},
                "module_analysis": {},
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

            self.evaluation_history.append(result)
            return result

    def evaluate_multiple_configurations(self, configs: List[Dict],
                                       target_metric: str = "sharpe_ratio",
                                       max_time_hours: float = None) -> List[Dict]:
        """Evaluate multiple configurations"""
        results = []
        start_time = time.time()
        max_time_seconds = max_time_hours * 3600 if max_time_hours else float('inf')

        print(f"🚀 Starting evaluation of {len(configs)} configurations")
        print(f"📊 Target metric: {target_metric}")

        for i, config in enumerate(configs):
            if time.time() - start_time > max_time_seconds:
                print(f"⏰ Time limit reached after {i} configurations")
                break

            print(f"[{i+1}/{len(configs)}] Evaluating configuration {self._get_config_fingerprint(config)[:8]}")

            result = self.evaluate_configuration(config, target_metric=target_metric)
            results.append(result)

            # Print progress
            if result["status"] == "success":
                score = result["target_score"]
                print(f"✅ Score: {score:.3f}")
            else:
                print(f"❌ Failed: {result['error']}")

            # Early stopping for excellent results
            if result["target_score"] > 2.0:  # Sharpe > 2.0 is excellent
                print(f"🎯 Excellent configuration found! Stopping early.")
                break

        return results

    def compare_configurations(self, results: List[Dict]) -> Dict[str, Any]:
        """Compare multiple configuration results"""
        if not results:
            return {"error": "No results to compare"}

        successful_results = [r for r in results if r["status"] == "success"]

        if not successful_results:
            return {"error": "No successful evaluations to compare"}

        # Sort by target score
        sorted_results = sorted(successful_results,
                              key=lambda x: x["target_score"], reverse=True)

        # Statistical analysis
        scores = [r["target_score"] for r in successful_results]

        comparison = {
            "total_configurations": len(results),
            "successful_configurations": len(successful_results),
            "success_rate": len(successful_results) / len(results),

            "best_configuration": sorted_results[0],
            "worst_configuration": sorted_results[-1],

            "score_statistics": {
                "mean": np.mean(scores),
                "median": np.median(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "q25": np.percentile(scores, 25),
                "q75": np.percentile(scores, 75)
            },

            "top_5_configurations": sorted_results[:5],
            "performance_distribution": self._analyze_performance_distribution(successful_results)
        }

        return comparison

    def analyze_module_impact(self, ablation_results: List[Dict]) -> Dict[str, Any]:
        """Analyze the impact of individual modules"""
        if not ablation_results:
            return {"error": "No ablation results provided"}

        analysis = {
            "module_contributions": {},
            "module_rankings": [],
            "baseline_performance": None
        }

        # Find baseline performance
        baseline_result = next((r for r in ablation_results
                               if "baseline" in r.get("config_fingerprint", "")), None)

        if baseline_result:
            analysis["baseline_performance"] = baseline_result["target_score"]

        # Analyze individual module contributions
        for result in ablation_results:
            fingerprint = result.get("config_fingerprint", "")

            if "only_" in fingerprint:
                # Single module test
                module_name = fingerprint.replace("only_", "").split("_")[0]
                analysis["module_contributions"][module_name] = {
                    "individual_performance": result["target_score"],
                    "status": result["status"]
                }

            elif "without_" in fingerprint:
                # Module removal test
                module_name = fingerprint.replace("without_", "").split("_")[0]
                if module_name in analysis["module_contributions"]:
                    analysis["module_contributions"][module_name]["without_performance"] = result["target_score"]

        # Calculate module impact scores
        baseline_score = analysis["baseline_performance"] or 0

        for module_name, contrib in analysis["module_contributions"].items():
            individual_score = contrib.get("individual_performance", 0)
            without_score = contrib.get("without_performance", baseline_score)

            # Impact when module is alone
            individual_impact = individual_score

            # Impact when module is removed
            removal_impact = baseline_score - without_score

            contrib["individual_impact"] = individual_impact
            contrib["removal_impact"] = removal_impact
            contrib["total_impact"] = (individual_impact + removal_impact) / 2

        # Rank modules by impact
        module_rankings = []
        for module_name, contrib in analysis["module_contributions"].items():
            module_rankings.append({
                "module": module_name,
                "impact_score": contrib["total_impact"],
                "individual_performance": contrib.get("individual_performance", 0),
                "removal_impact": contrib.get("removal_impact", 0)
            })

        analysis["module_rankings"] = sorted(module_rankings,
                                           key=lambda x: x["impact_score"],
                                           reverse=True)

        return analysis

    def _run_backtest_with_config(self, config: Dict, period: Tuple[str, str]) -> Dict:
        """Run backtest with a specific configuration"""
        # This would integrate with the actual backtesting system
        # For now, simulate a backtest result

        if self.backtest_engine:
            return self.backtest_engine.run_backtest(config, period)
        else:
            # Simulate backtest for testing
            return self._simulate_backtest_result(config)

    def _simulate_backtest_result(self, config: Dict) -> Dict:
        """Simulate a backtest result for testing purposes"""
        # Count enabled modules
        enabled_modules = sum(1 for module_config in config['modules'].values()
                            if module_config.get('enabled', False))

        # Simulate performance based on module count and randomness
        base_return = 0.15 + (enabled_modules * 0.02)  # More modules = higher return
        noise = np.random.normal(0, 0.05)
        total_return = base_return + noise

        volatility = max(0.1, 0.2 - (enabled_modules * 0.01) + abs(noise))
        sharpe_ratio = (total_return - 0.02) / volatility  # Risk-free rate = 2%

        max_drawdown = -abs(np.random.normal(0.08, 0.03))

        # Generate some daily returns
        num_days = 180
        daily_returns = np.random.normal(total_return/252, volatility/np.sqrt(252), num_days)

        return {
            "total_return": total_return,
            "annualized_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "volatility": volatility,
            "max_drawdown": max_drawdown,
            "daily_returns": daily_returns.tolist(),
            "total_trades": enabled_modules * 10 + np.random.randint(-5, 5),
            "win_rate": min(0.95, max(0.4, 0.7 + np.random.normal(0, 0.1)))
        }

    def _calculate_comprehensive_metrics(self, backtest_result: Dict) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        metrics = {}

        # Basic metrics
        metrics["total_return"] = backtest_result.get("total_return", 0)
        metrics["annualized_return"] = backtest_result.get("annualized_return", 0)
        metrics["sharpe_ratio"] = backtest_result.get("sharpe_ratio", 0)
        metrics["volatility"] = backtest_result.get("volatility", 0)
        metrics["max_drawdown"] = backtest_result.get("max_drawdown", 0)

        # Advanced metrics
        daily_returns = np.array(backtest_result.get("daily_returns", []))

        if len(daily_returns) > 0:
            # Sortino ratio (downside deviation)
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.01
            metrics["sortino_ratio"] = (metrics["annualized_return"] - 0.02) / (downside_std * np.sqrt(252))

            # Calmar ratio
            metrics["calmar_ratio"] = metrics["annualized_return"] / abs(metrics["max_drawdown"]) if metrics["max_drawdown"] != 0 else 0

            # VaR and CVaR
            metrics["var_95"] = np.percentile(daily_returns, 5)
            metrics["cvar_95"] = daily_returns[daily_returns <= metrics["var_95"]].mean()

            # Skewness and Kurtosis
            metrics["skewness"] = self._calculate_skewness(daily_returns)
            metrics["kurtosis"] = self._calculate_kurtosis(daily_returns)

        # Trading metrics
        metrics["total_trades"] = backtest_result.get("total_trades", 0)
        metrics["win_rate"] = backtest_result.get("win_rate", 0)

        # Risk-adjusted metrics
        metrics["risk_adjusted_return"] = metrics["annualized_return"] / max(0.01, metrics["volatility"])

        return metrics

    def _analyze_module_contribution(self, config: Dict, backtest_result: Dict) -> Dict[str, Any]:
        """Analyze the contribution of each enabled module"""
        enabled_modules = [name for name, module_config in config['modules'].items()
                          if module_config.get('enabled', False)]

        analysis = {
            "enabled_modules": enabled_modules,
            "total_enabled": len(enabled_modules),
            "estimated_contributions": {}
        }

        # Simple heuristic for module contributions
        # In practice, this would require more sophisticated attribution
        total_performance = backtest_result.get("sharpe_ratio", 0)
        base_performance = 0.1  # Baseline
        excess_performance = max(0, total_performance - base_performance)

        if len(enabled_modules) > 0:
            avg_contribution = excess_performance / len(enabled_modules)

            for module in enabled_modules:
                # Add some variation based on module type
                module_weight = {
                    "technical_indicators": 1.2,
                    "regime_detection": 1.5,
                    "risk_management": 1.1,
                    "sentiment_analysis": 0.8,
                    "portfolio_management": 1.0
                }.get(module, 1.0)

                analysis["estimated_contributions"][module] = avg_contribution * module_weight

        return analysis

    def _analyze_performance_distribution(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze the distribution of performance scores"""
        scores = [r["target_score"] for r in results]

        return {
            "excellent_configs": len([s for s in scores if s > 1.5]),  # Sharpe > 1.5
            "good_configs": len([s for s in scores if 1.0 < s <= 1.5]),  # 1.0 < Sharpe <= 1.5
            "average_configs": len([s for s in scores if 0.5 < s <= 1.0]),  # 0.5 < Sharpe <= 1.0
            "poor_configs": len([s for s in scores if s <= 0.5]),  # Sharpe <= 0.5
            "performance_bands": {
                "top_10_percent": np.percentile(scores, 90),
                "top_25_percent": np.percentile(scores, 75),
                "median": np.percentile(scores, 50),
                "bottom_25_percent": np.percentile(scores, 25),
                "bottom_10_percent": np.percentile(scores, 10)
            }
        }

    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns"""
        if len(returns) < 3:
            return 0
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0
        return np.mean(((returns - mean) / std) ** 3)

    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns"""
        if len(returns) < 4:
            return 0
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0
        return np.mean(((returns - mean) / std) ** 4) - 3  # Excess kurtosis

    def _get_config_fingerprint(self, config: Dict) -> str:
        """Generate a fingerprint for the configuration"""
        import hashlib
        import json

        modules_config = config.get('modules', {})
        config_str = json.dumps(modules_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations performed"""
        if not self.evaluation_history:
            return {"message": "No evaluations performed yet"}

        successful_evals = [e for e in self.evaluation_history if e["status"] == "success"]

        return {
            "total_evaluations": len(self.evaluation_history),
            "successful_evaluations": len(successful_evals),
            "success_rate": len(successful_evals) / len(self.evaluation_history),
            "best_score": max(e["target_score"] for e in successful_evals) if successful_evals else 0,
            "average_score": np.mean([e["target_score"] for e in successful_evals]) if successful_evals else 0,
            "total_evaluation_time": sum(e["evaluation_time_seconds"] for e in self.evaluation_history)
        }