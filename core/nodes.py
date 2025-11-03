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

    @property
    @abstractmethod
    def tree_repr(self) -> str: ...


@dataclass
class Leaf(PomapNode):
    label: str
    factory: Callable

    @property
    def children(self):
        return []

    @property
    def tree_repr(self) -> str:
        return self.label


@dataclass
class Lift(PomapNode):
    child: PomapNode
    atomics: Iterable[DataType]
    train_mask_for_label: Callable[[DataType], Expr]
    test_mask_for_label: Callable[[DataType], Expr]
    namespace: Optional[str] = None

    def __post_init__(self):
        if self.namespace is None:
            self.namespace = f"Lift: {self.atomics}"
        self.atomics = set(self.atomics)

    @property
    def children(self) -> Iterable["PomapNode"]:
        return [self.child]

    @property
    def tree_repr(self) -> str:
        return f"{self.namespace}: {self.atomics}"


@dataclass
class Ensemble(PomapNode):
    models: Iterable[PomapNode]

    @property
    def children(self):
        return self.models

    @property
    def tree_repr(self) -> str:
        return "Ensemble"


@dataclass
class LearnsFrom(PomapNode):
    learner: PomapNode
    learns_from: PomapNode
    learn_logic: Callable[
        [PomapNode, DataFrame], dict
    ]  # TODO this should actually typehint Model instead of PomapNode, but need to re-organise dirs first

    @property
    def children(self) -> Iterable["PomapNode"]:
        return [self.learner, self.learns_from]

    @property
    def tree_repr(self) -> str:
        return "LearnsFrom"
