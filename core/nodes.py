from polars import DataType, Expr, DataFrame
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional, Callable


class PomapNode(ABC):
    @property
    @abstractmethod
    def children(self) -> Iterable["PomapNode"]:
        """Return iterable of child nodes."""
        ...


@dataclass
class Leaf(PomapNode):
    label: str
    factory: Callable

    @property
    def children(self):
        return []


@dataclass
class Lift(PomapNode):
    child: PomapNode
    atomics: Iterable[DataType]
    train_mask_for_label: Callable[[DataType], Expr]
    test_mask_for_label: Callable[[DataType], Expr]
    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            self.name = f"Lift: {self.atomics}"
        self.atomics = set(self.atomics)

    @property
    def children(self) -> Iterable["PomapNode"]:
        return [self.child]


@dataclass
class Ensemble(PomapNode):
    models: Iterable[PomapNode]
    name = "Ensemble"

    @property
    def children(self):
        return self.models


@dataclass
class LearnsFrom(PomapNode):
    learner: PomapNode
    learns_from: PomapNode
    learn_logic: Callable[[PomapNode, DataFrame], dict]
    name = "LearnsFrom"

    @property
    def children(self) -> Iterable["PomapNode"]:
        return [self.learner, self.learns_from]
