from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, runtime_checkable

from polars import DataFrame, DataType, Expr


@runtime_checkable
class ModelProtocol(Protocol):
    def fit(self, df): ...

    def predict(self, df): ...


class PomapNode(ABC):
    name: str

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
    factory: Callable[[], ModelProtocol]

    @property
    def children(self):
        return []

    @property
    def tree_repr(self) -> str:
        return self.label

    @property
    def name(self) -> str:
        return f"Leaf: {self.label}"


@dataclass
class Lift(PomapNode):
    name: str
    child: PomapNode
    values: Iterable[DataType]
    train_filter: Callable[[DataType], Expr]
    test_filter: Callable[[DataType], Expr]
    validation_filter: Callable[[DataType], Expr] | None = None
    aggregate_with: dict[str, Callable[[list[str]], Expr]] | None = None

    def __post_init__(self):
        self.values = set(self.values)

    @property
    def children(self) -> Iterable["PomapNode"]:
        return [self.child]

    @property
    def tree_repr(self) -> str:
        return f"{self.name}: {self.values}"


@dataclass
class Ensemble(PomapNode):
    name: str
    models: Iterable[PomapNode]
    aggregate_with: dict[str, Callable[[list[str]], Expr]] | None = None

    @property
    def children(self):
        return self.models

    @property
    def tree_repr(self) -> str:
        return f"Ensemble: {self.name}"


@dataclass
class LearnsFrom(PomapNode):
    name: str

    learner: PomapNode
    learns_from: PomapNode
    learn_logic: Callable[
        ["PomapNode", DataFrame], dict
    ]  # TODO this should actually typehint Model instead of PomapNode, but need to re-organise dirs first

    @property
    def children(self) -> Iterable["PomapNode"]:
        return [self.learns_from, self.learner]

    @property
    def tree_repr(self) -> str:
        return f"LearnsFrom: {self.name}"


@dataclass
class Feed(PomapNode):
    name: str
    source: PomapNode
    consumer: PomapNode

    @property
    def children(self) -> Iterable["PomapNode"]:
        return [self.source, self.consumer]

    @property
    def tree_repr(self) -> str:
        return f"Feed: {self.name}"
