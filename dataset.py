"""Dataset abstractions for benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List


@dataclass(slots=True)
class BenchmarkExample:
    """Single benchmark example with input, expected output, and metadata."""

    id: str
    input_text: str
    ground_truth: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkDataset:
    """Collection of examples for one benchmark task."""

    name: str
    task_name: str
    version: str
    examples: List[BenchmarkExample]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[BenchmarkExample]:
        return iter(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    @classmethod
    def from_examples(
        cls,
        name: str,
        task_name: str,
        version: str,
        examples: Iterable[BenchmarkExample],
        metadata: Dict[str, Any] | None = None,
    ) -> "BenchmarkDataset":
        return cls(
            name=name,
            task_name=task_name,
            version=version,
            examples=list(examples),
            metadata=metadata or {},
        )


# Backward compatibility alias.
DatasetExample = BenchmarkExample
