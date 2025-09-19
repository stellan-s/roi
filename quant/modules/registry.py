"""
Module registry system for managing and orchestrating trading modules.

The registry handles module discovery, dependency resolution, and pipeline creation.
"""

from typing import Dict, List, Type, Set
import importlib
import inspect
from collections import defaultdict, deque

from .base import BaseModule, ModuleContract

class DependencyError(Exception):
    """Raised when module dependencies cannot be resolved"""
    pass

class ModuleRegistry:
    """Central registry for all trading modules"""

    def __init__(self):
        self.module_classes: Dict[str, Type[BaseModule]] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.reverse_dependencies: Dict[str, List[str]] = defaultdict(list)

    def register_module(self, module_class: Type[BaseModule]):
        """Register a module class"""
        if not issubclass(module_class, BaseModule):
            raise ValueError(f"{module_class.__name__} must inherit from BaseModule")

        # Create temporary instance to get contract
        temp_instance = module_class(config={'enabled': False})
        contract = temp_instance.define_contract()

        # Store module class
        self.module_classes[contract.name] = module_class

        # Update dependency graph
        self.dependency_graph[contract.name] = contract.dependencies.copy()

        # Update reverse dependencies
        for dependency in contract.dependencies:
            self.reverse_dependencies[dependency].append(contract.name)

        print(f"📦 Registered module: {contract.name} v{contract.version}")

    def auto_discover_modules(self, package_path: str = "quant.modules"):
        """Automatically discover and register modules in a package"""
        try:
            package = importlib.import_module(package_path)

            # Look for module files
            for module_name in ['technical', 'sentiment', 'regime', 'portfolio', 'risk']:
                try:
                    module = importlib.import_module(f"{package_path}.{module_name}")

                    # Find module classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, BaseModule) and
                            obj != BaseModule and
                            not obj.__name__.startswith('Base')):
                            self.register_module(obj)

                except ImportError:
                    # Module doesn't exist yet, skip
                    continue

        except ImportError as e:
            print(f"⚠️  Could not auto-discover modules: {e}")

    def get_module_info(self) -> Dict[str, Dict]:
        """Get information about all registered modules"""
        info = {}

        for name, module_class in self.module_classes.items():
            temp_instance = module_class(config={'enabled': False})
            contract = temp_instance.define_contract()

            info[name] = {
                "class": module_class.__name__,
                "version": contract.version,
                "description": contract.description,
                "dependencies": contract.dependencies,
                "dependents": self.reverse_dependencies.get(name, []),
                "performance_sla": contract.performance_sla
            }

        return info

    def create_pipeline(self, config: Dict[str, Dict]) -> 'ModulePipeline':
        """Create execution pipeline based on configuration"""
        from .pipeline import ModulePipeline

        # Determine which modules are enabled
        enabled_modules = [name for name, module_config in config.items()
                          if module_config.get('enabled', False) and name in self.module_classes]

        if not enabled_modules:
            raise ValueError("No modules enabled in configuration")

        # Resolve dependencies and get execution order
        execution_order = self._resolve_dependencies(enabled_modules)

        # Create module instances
        module_instances = {}
        for module_name in execution_order:
            module_class = self.module_classes[module_name]
            module_config = config.get(module_name, {})
            module_instances[module_name] = module_class(module_config)

        return ModulePipeline(module_instances, execution_order)

    def validate_configuration(self, config: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Validate a module configuration"""
        issues = defaultdict(list)

        for module_name, module_config in config.items():
            if module_name not in self.module_classes:
                issues["unknown_modules"].append(module_name)
                continue

            if not module_config.get('enabled', False):
                continue

            # Check dependencies
            dependencies = self.dependency_graph.get(module_name, [])
            for dep in dependencies:
                if dep not in config:
                    issues["missing_dependencies"].append(f"{module_name} requires {dep}")
                elif not config[dep].get('enabled', False):
                    issues["disabled_dependencies"].append(f"{module_name} requires {dep} to be enabled")

        return dict(issues)

    def _resolve_dependencies(self, modules: List[str]) -> List[str]:
        """Resolve dependencies and return execution order using topological sort"""
        # Add all dependencies to the set
        all_modules = set(modules)
        to_add = deque(modules)

        while to_add:
            module = to_add.popleft()
            dependencies = self.dependency_graph.get(module, [])

            for dep in dependencies:
                if dep not in all_modules:
                    if dep not in self.module_classes:
                        raise DependencyError(f"Unknown dependency: {dep} required by {module}")
                    all_modules.add(dep)
                    to_add.append(dep)

        # Topological sort
        in_degree = {module: 0 for module in all_modules}

        # Calculate in-degrees
        for module in all_modules:
            dependencies = self.dependency_graph.get(module, [])
            for dep in dependencies:
                if dep in all_modules:
                    in_degree[module] += 1

        # Kahn's algorithm
        queue = deque([module for module in all_modules if in_degree[module] == 0])
        execution_order = []

        while queue:
            current = queue.popleft()
            execution_order.append(current)

            # Update in-degrees of dependents
            for dependent in self.reverse_dependencies.get(current, []):
                if dependent in all_modules:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        # Check for circular dependencies
        if len(execution_order) != len(all_modules):
            remaining = all_modules - set(execution_order)
            raise DependencyError(f"Circular dependency detected involving: {remaining}")

        return execution_order

    def get_dependency_tree(self, module_name: str) -> Dict:
        """Get the full dependency tree for a module"""
        if module_name not in self.module_classes:
            raise ValueError(f"Unknown module: {module_name}")

        def build_tree(name: str, visited: Set[str]) -> Dict:
            if name in visited:
                return {"name": name, "circular": True}

            visited.add(name)
            dependencies = self.dependency_graph.get(name, [])

            tree = {
                "name": name,
                "dependencies": [build_tree(dep, visited.copy()) for dep in dependencies]
            }

            return tree

        return build_tree(module_name, set())

    def simulate_module_impact(self, module_name: str, action: str) -> Dict:
        """Simulate the impact of enabling/disabling a module"""
        if action not in ['enable', 'disable']:
            raise ValueError("Action must be 'enable' or 'disable'")

        impact = {
            "directly_affected": [],
            "transitively_affected": [],
            "new_dependencies_needed": [],
            "orphaned_modules": []
        }

        if action == 'disable':
            # Find modules that depend on this one
            impact["directly_affected"] = self.reverse_dependencies.get(module_name, [])

            # Find transitive effects
            affected = set(impact["directly_affected"])
            to_check = deque(impact["directly_affected"])

            while to_check:
                current = to_check.popleft()
                dependents = self.reverse_dependencies.get(current, [])
                for dep in dependents:
                    if dep not in affected:
                        affected.add(dep)
                        to_check.append(dep)
                        impact["transitively_affected"].append(dep)

        elif action == 'enable':
            # Find dependencies that need to be enabled
            dependencies = self.dependency_graph.get(module_name, [])
            impact["new_dependencies_needed"] = dependencies.copy()

            # Find transitive dependencies
            to_check = deque(dependencies)
            all_deps = set(dependencies)

            while to_check:
                current = to_check.popleft()
                sub_deps = self.dependency_graph.get(current, [])
                for dep in sub_deps:
                    if dep not in all_deps:
                        all_deps.add(dep)
                        to_check.append(dep)
                        impact["new_dependencies_needed"].append(dep)

        return impact

    def __len__(self):
        return len(self.module_classes)

    def __contains__(self, module_name: str):
        return module_name in self.module_classes

    def __iter__(self):
        return iter(self.module_classes.keys())