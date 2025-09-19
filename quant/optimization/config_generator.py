"""
Configuration Generator for Module Optimization

Generates variations of module configurations for systematic testing and optimization.
"""

import copy
import random
import itertools
from typing import Dict, List, Any, Tuple
import numpy as np
import yaml

class ConfigurationGenerator:
    """Generate different module configurations to test"""

    def __init__(self, base_config_path: str = None, base_config: Dict = None):
        if base_config_path:
            with open(base_config_path, 'r') as f:
                self.base_config = yaml.safe_load(f)
        elif base_config:
            self.base_config = copy.deepcopy(base_config)
        else:
            raise ValueError("Must provide either base_config_path or base_config")

        self.parameter_ranges = self.base_config.get('optimization', {}).get('parameter_ranges', {})
        self.mutation_rate = self.base_config.get('optimization', {}).get('mutation_rate', 0.3)

    def generate_configurations(self, max_configs: int = 100) -> List[Dict]:
        """Generate variations of the base config"""
        configurations = []

        # Start with base config
        configurations.append(copy.deepcopy(self.base_config))

        # Generate random variations
        for i in range(max_configs - 1):
            config = self._generate_random_variation()
            configurations.append(config)

        return configurations

    def generate_ablation_study(self) -> List[Tuple[str, Dict]]:
        """Generate configs for ablation study (testing each module's contribution)"""
        configs = []

        # Baseline: all modules enabled
        baseline = copy.deepcopy(self.base_config)
        configs.append(("baseline_all_modules", baseline))

        # Test each module individually
        module_names = list(self.base_config['modules'].keys())

        for module_name in module_names:
            config = self._create_single_module_config(module_name)
            configs.append((f"only_{module_name}", config))

        # Test removing each module one at a time
        for module_name in module_names:
            config = self._create_all_except_config(module_name)
            configs.append((f"without_{module_name}", config))

        # Test pairs of modules
        for module1, module2 in itertools.combinations(module_names, 2):
            config = self._create_dual_module_config(module1, module2)
            configs.append((f"only_{module1}_and_{module2}", config))

        return configs

    def generate_grid_search(self) -> List[Dict]:
        """Generate configurations for grid search optimization"""
        if not self.parameter_ranges:
            return [copy.deepcopy(self.base_config)]

        # Create parameter combinations
        param_combinations = self._generate_parameter_combinations()

        configurations = []
        for param_combo in param_combinations:
            config = copy.deepcopy(self.base_config)
            config = self._apply_parameters(config, param_combo)
            configurations.append(config)

        return configurations

    def generate_genetic_population(self, population_size: int = 50) -> List[Dict]:
        """Generate initial population for genetic algorithm"""
        population = []

        # Include base config
        population.append(copy.deepcopy(self.base_config))

        # Generate random individuals
        for _ in range(population_size - 1):
            individual = self._generate_random_variation()
            population.append(individual)

        return population

    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Genetic algorithm crossover operation"""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        # Module-level crossover
        modules1 = child1['modules']
        modules2 = child2['modules']

        for module_name in modules1.keys():
            if random.random() < 0.5:
                # Swap module configurations
                modules1[module_name], modules2[module_name] = \
                    modules2[module_name], modules1[module_name]

        # Parameter-level crossover
        for module_name, module_config in modules1.items():
            if module_name in modules2:
                self._crossover_module_params(module_config, modules2[module_name])

        return child1, child2

    def mutate(self, config: Dict, mutation_rate: float = None) -> Dict:
        """Genetic algorithm mutation operation"""
        if mutation_rate is None:
            mutation_rate = self.mutation_rate

        mutated = copy.deepcopy(config)

        # Module enable/disable mutations
        for module_name, module_config in mutated['modules'].items():
            if random.random() < mutation_rate * 0.5:  # Lower rate for enable/disable
                module_config['enabled'] = not module_config['enabled']

        # Parameter mutations
        for module_name, module_config in mutated['modules'].items():
            self._mutate_module_parameters(module_config, mutation_rate)

        return mutated

    def _generate_random_variation(self) -> Dict:
        """Generate a random variation of the base config"""
        config = copy.deepcopy(self.base_config)

        # Randomly enable/disable modules
        for module_name, module_config in config['modules'].items():
            if random.random() < 0.3:  # 30% chance to flip
                module_config['enabled'] = not module_config['enabled']

        # Randomly vary parameters
        for module_name, module_config in config['modules'].items():
            self._randomize_module_parameters(module_config)

        return config

    def _create_single_module_config(self, target_module: str) -> Dict:
        """Create config with only one module enabled"""
        config = copy.deepcopy(self.base_config)

        for module_name, module_config in config['modules'].items():
            module_config['enabled'] = (module_name == target_module)

        return config

    def _create_all_except_config(self, excluded_module: str) -> Dict:
        """Create config with all modules enabled except one"""
        config = copy.deepcopy(self.base_config)

        for module_name, module_config in config['modules'].items():
            module_config['enabled'] = (module_name != excluded_module)

        return config

    def _create_dual_module_config(self, module1: str, module2: str) -> Dict:
        """Create config with only two modules enabled"""
        config = copy.deepcopy(self.base_config)

        for module_name, module_config in config['modules'].items():
            module_config['enabled'] = (module_name in [module1, module2])

        return config

    def _generate_parameter_combinations(self) -> List[Dict]:
        """Generate all combinations of parameter ranges"""
        if not self.parameter_ranges:
            return [{}]

        # Create parameter grids
        param_names = []
        param_values = []

        for param_name, param_range in self.parameter_ranges.items():
            param_names.append(param_name)
            if isinstance(param_range, list) and len(param_range) == 2:
                # Range format [min, max]
                if isinstance(param_range[0], int):
                    values = list(range(param_range[0], param_range[1] + 1,
                                      max(1, (param_range[1] - param_range[0]) // 5)))
                else:
                    values = np.linspace(param_range[0], param_range[1], 5).tolist()
            else:
                # List of specific values
                values = param_range
            param_values.append(values)

        # Generate combinations
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)

        return combinations

    def _apply_parameters(self, config: Dict, parameters: Dict) -> Dict:
        """Apply parameter values to config"""
        for param_name, param_value in parameters.items():
            # Find where this parameter belongs
            for module_name, module_config in config['modules'].items():
                if param_name in module_config:
                    module_config[param_name] = param_value
                    break

        return config

    def _randomize_module_parameters(self, module_config: Dict):
        """Randomly vary parameters within a module"""
        for param_name, param_value in module_config.items():
            if param_name in self.parameter_ranges:
                param_range = self.parameter_ranges[param_name]

                if isinstance(param_range, list) and len(param_range) == 2:
                    if isinstance(param_range[0], int):
                        new_value = random.randint(param_range[0], param_range[1])
                    else:
                        new_value = random.uniform(param_range[0], param_range[1])
                    module_config[param_name] = new_value

            elif isinstance(param_value, (int, float)) and param_name != 'enabled':
                # Apply random noise to numeric parameters
                if isinstance(param_value, int):
                    noise = random.randint(-2, 2)
                    module_config[param_name] = max(1, param_value + noise)
                else:
                    noise = random.uniform(-0.1, 0.1) * param_value
                    module_config[param_name] = max(0.01, param_value + noise)

    def _mutate_module_parameters(self, module_config: Dict, mutation_rate: float):
        """Mutate parameters within a module"""
        for param_name, param_value in module_config.items():
            if random.random() < mutation_rate:
                if param_name in self.parameter_ranges:
                    param_range = self.parameter_ranges[param_name]

                    if isinstance(param_range, list) and len(param_range) == 2:
                        if isinstance(param_range[0], int):
                            module_config[param_name] = random.randint(param_range[0], param_range[1])
                        else:
                            module_config[param_name] = random.uniform(param_range[0], param_range[1])

                elif isinstance(param_value, (int, float)) and param_name != 'enabled':
                    # Small random mutation
                    if isinstance(param_value, int):
                        module_config[param_name] = max(1, param_value + random.randint(-1, 1))
                    else:
                        mutation = random.uniform(-0.05, 0.05) * param_value
                        module_config[param_name] = max(0.01, param_value + mutation)

    def _crossover_module_params(self, config1: Dict, config2: Dict):
        """Crossover parameters between two module configurations"""
        for param_name in config1.keys():
            if param_name in config2 and random.random() < 0.5:
                config1[param_name], config2[param_name] = config2[param_name], config1[param_name]

    def save_configuration(self, config: Dict, filepath: str):
        """Save a configuration to file"""
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

    def load_configuration(self, filepath: str) -> Dict:
        """Load a configuration from file"""
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)

    def get_configuration_fingerprint(self, config: Dict) -> str:
        """Generate a unique fingerprint for a configuration"""
        import hashlib
        import json

        # Extract just the modules section for fingerprinting
        modules_config = config.get('modules', {})

        # Convert to string and hash
        config_str = json.dumps(modules_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]