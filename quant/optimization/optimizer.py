"""
System Optimizer - Automatically discovers optimal module configurations

Uses genetic algorithms, grid search, and ablation studies to find the best
module configuration for the trading system.
"""

import random
import time
import json
from typing import Dict, List, Any, Tuple
from datetime import datetime
import numpy as np
from pathlib import Path

from .config_generator import ConfigurationGenerator
from .evaluator import PerformanceEvaluator

class SystemOptimizer:
    """Automatically finds optimal module configuration"""

    def __init__(self, base_config: Dict,
                 evaluator: PerformanceEvaluator = None):
        self.base_config = base_config
        self.config_generator = ConfigurationGenerator(base_config=base_config)
        self.evaluator = evaluator or PerformanceEvaluator()

        self.optimization_history = []
        self.best_configurations = []

    def run_optimization(self,
                        duration_hours: float = 24,
                        target_metric: str = "sharpe_ratio",
                        method: str = "genetic") -> Dict[str, Any]:
        """Run optimization for specified duration"""

        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)

        print(f"🚀 Starting {duration_hours}h optimization using {method} method")
        print(f"📊 Target metric: {target_metric}")
        print(f"⏰ Will run until: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")

        optimization_log = {
            "method": method,
            "start_time": datetime.now().isoformat(),
            "duration_hours": duration_hours,
            "target_metric": target_metric,
            "configurations_tested": 0,
            "best_score": -999,
            "best_config": None,
            "status": "running"
        }

        try:
            if method == "genetic":
                result = self._run_genetic_optimization(end_time, target_metric)
            elif method == "grid_search":
                result = self._run_grid_search_optimization(end_time, target_metric)
            elif method == "ablation":
                result = self._run_ablation_study(target_metric)
            elif method == "random":
                result = self._run_random_search(end_time, target_metric)
            else:
                raise ValueError(f"Unknown optimization method: {method}")

            optimization_log.update(result)
            optimization_log["status"] = "completed"
            optimization_log["end_time"] = datetime.now().isoformat()

        except Exception as e:
            optimization_log["status"] = "failed"
            optimization_log["error"] = str(e)
            optimization_log["end_time"] = datetime.now().isoformat()

        self.optimization_history.append(optimization_log)
        return optimization_log

    def run_ablation_study(self, target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """Run comprehensive ablation study"""
        print("🔬 Running ablation study to measure module contributions")

        # Generate ablation configurations
        configs = self.config_generator.generate_ablation_study()

        print(f"📋 Testing {len(configs)} configurations:")
        for name, _ in configs:
            print(f"  - {name}")

        # Evaluate all configurations
        results = []
        for i, (config_name, config) in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] Testing: {config_name}")

            result = self.evaluator.evaluate_configuration(config, target_metric=target_metric)
            result["config_name"] = config_name
            results.append(result)

            if result["status"] == "success":
                print(f"✅ {target_metric}: {result['target_score']:.3f}")
            else:
                print(f"❌ Failed: {result['error']}")

        # Analyze results
        module_impact_analysis = self.evaluator.analyze_module_impact(results)
        comparison = self.evaluator.compare_configurations(results)

        return {
            "method": "ablation",
            "results": results,
            "module_impact_analysis": module_impact_analysis,
            "comparison": comparison,
            "configurations_tested": len(configs),
            "best_score": comparison.get("best_configuration", {}).get("target_score", 0),
            "recommendations": self._generate_ablation_recommendations(module_impact_analysis)
        }

    def _run_genetic_optimization(self, end_time: float, target_metric: str) -> Dict[str, Any]:
        """Run genetic algorithm optimization"""
        population_size = 50
        mutation_rate = 0.3
        crossover_rate = 0.7
        elite_size = 5

        print(f"🧬 Genetic Algorithm Parameters:")
        print(f"   Population size: {population_size}")
        print(f"   Mutation rate: {mutation_rate}")
        print(f"   Crossover rate: {crossover_rate}")
        print(f"   Elite size: {elite_size}")

        # Initialize population
        population = self.config_generator.generate_genetic_population(population_size)
        generation = 0
        best_score = -999
        best_config = None
        configurations_tested = 0

        while time.time() < end_time:
            generation += 1
            print(f"\n🧬 Generation {generation}")

            # Evaluate population
            population_results = []
            for config in population:
                if time.time() >= end_time:
                    break

                result = self.evaluator.evaluate_configuration(config, target_metric=target_metric)
                population_results.append((config, result))
                configurations_tested += 1

                # Track best
                if result["target_score"] > best_score:
                    best_score = result["target_score"]
                    best_config = config
                    print(f"🎯 New best score: {best_score:.3f}")

            if time.time() >= end_time:
                break

            # Sort by fitness
            population_results.sort(key=lambda x: x[1]["target_score"], reverse=True)

            # Print generation stats
            scores = [r[1]["target_score"] for r in population_results if r[1]["status"] == "success"]
            if scores:
                print(f"   Best: {max(scores):.3f}, Avg: {np.mean(scores):.3f}, Worst: {min(scores):.3f}")

            # Selection and reproduction
            new_population = []

            # Elitism - keep best individuals
            for i in range(min(elite_size, len(population_results))):
                if population_results[i][1]["status"] == "success":
                    new_population.append(population_results[i][0])

            # Generate offspring
            while len(new_population) < population_size and time.time() < end_time:
                # Tournament selection
                parent1 = self._tournament_selection(population_results, tournament_size=3)
                parent2 = self._tournament_selection(population_results, tournament_size=3)

                # Crossover
                if random.random() < crossover_rate:
                    child1, child2 = self.config_generator.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                # Mutation
                child1 = self.config_generator.mutate(child1, mutation_rate)
                child2 = self.config_generator.mutate(child2, mutation_rate)

                new_population.extend([child1, child2])

            population = new_population[:population_size]

        return {
            "configurations_tested": configurations_tested,
            "generations": generation,
            "best_score": best_score,
            "best_config": best_config,
            "final_population_size": len(population)
        }

    def _run_grid_search_optimization(self, end_time: float, target_metric: str) -> Dict[str, Any]:
        """Run grid search optimization"""
        configs = self.config_generator.generate_grid_search()

        print(f"🔍 Grid Search: Testing {len(configs)} parameter combinations")

        results = []
        configurations_tested = 0
        best_score = -999
        best_config = None

        for i, config in enumerate(configs):
            if time.time() >= end_time:
                print(f"⏰ Time limit reached after {i} configurations")
                break

            print(f"[{i+1}/{len(configs)}] Testing parameter combination {i+1}")

            result = self.evaluator.evaluate_configuration(config, target_metric=target_metric)
            results.append(result)
            configurations_tested += 1

            if result["target_score"] > best_score:
                best_score = result["target_score"]
                best_config = config
                print(f"🎯 New best score: {best_score:.3f}")

        return {
            "configurations_tested": configurations_tested,
            "best_score": best_score,
            "best_config": best_config,
            "all_results": results
        }

    def _run_random_search(self, end_time: float, target_metric: str) -> Dict[str, Any]:
        """Run random search optimization"""
        print(f"🎲 Random Search: Testing random configurations until time limit")

        configurations_tested = 0
        best_score = -999
        best_config = None
        results = []

        while time.time() < end_time:
            # Generate random configuration
            config = self.config_generator._generate_random_variation()

            print(f"[{configurations_tested+1}] Testing random configuration")

            result = self.evaluator.evaluate_configuration(config, target_metric=target_metric)
            results.append(result)
            configurations_tested += 1

            if result["target_score"] > best_score:
                best_score = result["target_score"]
                best_config = config
                print(f"🎯 New best score: {best_score:.3f}")

            # Early stopping for excellent results
            if best_score > 2.0:
                print(f"🎉 Excellent score found! Stopping early.")
                break

        return {
            "configurations_tested": configurations_tested,
            "best_score": best_score,
            "best_config": best_config,
            "all_results": results
        }

    def _tournament_selection(self, population_results: List[Tuple], tournament_size: int = 3):
        """Tournament selection for genetic algorithm"""
        tournament = random.sample(population_results, min(tournament_size, len(population_results)))
        tournament = [p for p in tournament if p[1]["status"] == "success"]

        if not tournament:
            return random.choice(population_results)[0]

        winner = max(tournament, key=lambda x: x[1]["target_score"])
        return winner[0]

    def _generate_ablation_recommendations(self, module_impact_analysis: Dict) -> List[str]:
        """Generate recommendations based on ablation study"""
        recommendations = []

        rankings = module_impact_analysis.get("module_rankings", [])

        if rankings:
            # Top performing modules
            top_modules = rankings[:3]
            recommendations.append(f"🌟 High-impact modules: {', '.join([m['module'] for m in top_modules])}")

            # Low performing modules
            bottom_modules = rankings[-2:]
            if len(rankings) > 3:
                recommendations.append(f"⚠️  Consider disabling: {', '.join([m['module'] for m in bottom_modules])}")

            # Module combinations
            if len(rankings) >= 2:
                best_combo = f"{rankings[0]['module']} + {rankings[1]['module']}"
                recommendations.append(f"🔄 Try combination: {best_combo}")

        baseline_performance = module_impact_analysis.get("baseline_performance", 0)
        if baseline_performance:
            recommendations.append(f"📊 Baseline performance: {baseline_performance:.3f}")

        return recommendations

    def save_optimization_results(self, results: Dict, output_dir: str = "optimization_results"):
        """Save optimization results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save main results
        results_file = output_path / f"optimization_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save best configuration
        if results.get("best_config"):
            config_file = output_path / f"best_config_{timestamp}.yaml"
            self.config_generator.save_configuration(results["best_config"], str(config_file))

        print(f"💾 Results saved to: {output_path}")
        return str(output_path)

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization runs"""
        if not self.optimization_history:
            return {"message": "No optimizations run yet"}

        successful_runs = [o for o in self.optimization_history if o["status"] == "completed"]

        if not successful_runs:
            return {"message": "No successful optimization runs"}

        best_overall = max(successful_runs, key=lambda x: x.get("best_score", -999))

        return {
            "total_optimization_runs": len(self.optimization_history),
            "successful_runs": len(successful_runs),
            "methods_used": list(set(o["method"] for o in self.optimization_history)),
            "best_overall_score": best_overall.get("best_score", 0),
            "best_method": best_overall.get("method", "unknown"),
            "total_configurations_tested": sum(o.get("configurations_tested", 0) for o in self.optimization_history),
            "recent_runs": self.optimization_history[-5:]  # Last 5 runs
        }

    def recommend_next_optimization(self) -> Dict[str, str]:
        """Recommend next optimization strategy based on history"""
        if not self.optimization_history:
            return {
                "method": "ablation",
                "reason": "Start with ablation study to understand module contributions"
            }

        recent_methods = [o["method"] for o in self.optimization_history[-3:]]

        if "ablation" not in recent_methods:
            return {
                "method": "ablation",
                "reason": "Run ablation study to understand current module performance"
            }

        if "genetic" not in recent_methods:
            return {
                "method": "genetic",
                "reason": "Use genetic algorithm for comprehensive configuration search"
            }

        if "grid_search" not in recent_methods:
            return {
                "method": "grid_search",
                "reason": "Systematic grid search of parameter space"
            }

        return {
            "method": "genetic",
            "reason": "Continue genetic optimization with longer duration"
        }