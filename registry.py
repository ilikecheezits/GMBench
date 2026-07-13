"""Registry and discovery for systems, datasets, and metrics."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Callable, Dict, Type

from dataset import BenchmarkDataset
from evaluators import Metric
from workflow import SystemUnderTest

SYSTEM_REGISTRY: Dict[str, Type[SystemUnderTest]] = {}
DATASET_REGISTRY: Dict[str, Callable[[], BenchmarkDataset]] = {}
METRIC_REGISTRY: Dict[str, Type[Metric]] = {}


def register_system(name: str):
    def decorator(cls: Type[SystemUnderTest]) -> Type[SystemUnderTest]:
        SYSTEM_REGISTRY[name] = cls
        return cls

    return decorator


def register_dataset(name: str):
    def decorator(factory: Callable[[], BenchmarkDataset]) -> Callable[[], BenchmarkDataset]:
        DATASET_REGISTRY[name] = factory
        return factory

    return decorator


def register_metric(name: str):
    def decorator(cls: Type[Metric]) -> Type[Metric]:
        METRIC_REGISTRY[name] = cls
        return cls

    return decorator


def discover_package_modules(package_name: str) -> None:
    package = importlib.import_module(package_name)
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return

    for module_info in pkgutil.walk_packages(package_path, package_name + "."):
        importlib.import_module(module_info.name)


def build_metric_suite(metric_names: list[str]) -> list[Metric]:
    metrics: list[Metric] = []
    for name in metric_names:
        metric_cls = METRIC_REGISTRY[name]
        metrics.append(metric_cls())
    return metrics
