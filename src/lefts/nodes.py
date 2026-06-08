from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, runtime_checkable

from polars import DataFrame, DataType, Expr


@runtime_checkable
class ModelProtocol(Protocol):
    def fit(self, df): ...

    def predict(self, df): ...


class LeftsNode(ABC):
    name: str

    @property
    @abstractmethod
    def children(self) -> Iterable["LeftsNode"]:
        """Return iterable of child nodes."""
        ...

    @property
    @abstractmethod
    def tree_repr(self) -> str: ...


@dataclass
class Leaf(LeftsNode):
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
class Lift(LeftsNode):
    name: str
    child: LeftsNode
    values: Iterable[DataType]
    train_filter: Callable[[DataType], Expr]
    test_filter: Callable[[DataType], Expr]
    validation_filter: Callable[[DataType], Expr] | None = None
    aggregate_with: Callable[..., Expr] | None = None

    def __post_init__(self):
        self.values = set(self.values)

    @property
    def children(self) -> Iterable["LeftsNode"]:
        return [self.child]

    @property
    def tree_repr(self) -> str:
        return f"{self.name}: {self.values}"


@dataclass
class Split(LeftsNode):
    name: str
    child: LeftsNode
    train_filter: Expr
    test_filter: Expr
    validation_filter: Expr | None = None

    @property
    def children(self) -> Iterable["LeftsNode"]:
        return [self.child]

    @property
    def tree_repr(self) -> str:
        return f"Split: {self.name}"


@dataclass
class Ensemble(LeftsNode):
    name: str
    models: Iterable[LeftsNode]
    aggregate_with: Callable[..., Expr] | None = None

    @property
    def children(self):
        return self.models

    @property
    def tree_repr(self) -> str:
        return f"Ensemble: {self.name}"


@dataclass
class Tune(LeftsNode):
    name: str

    consumer: LeftsNode
    source: LeftsNode
    logic: Callable[
        ["LeftsNode", DataFrame], dict
    ]  # TODO this should actually typehint Model instead of LeftsNode, but need to re-organise dirs first

    @property
    def children(self) -> Iterable["LeftsNode"]:
        return [self.source, self.consumer]

    @property
    def tree_repr(self) -> str:
        return f"Tune: {self.name}"


@dataclass
class Feed(LeftsNode):
    name: str
    source: LeftsNode
    consumer: LeftsNode

    @property
    def children(self) -> Iterable["LeftsNode"]:
        return [self.source, self.consumer]

    @property
    def tree_repr(self) -> str:
        return f"Feed: {self.name}"
