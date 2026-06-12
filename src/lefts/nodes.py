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


@dataclass
class Leaf(LeftsNode):
    label: str
    factory: Callable[[], ModelProtocol]

    @property
    def children(self):
        return []

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
        if len(set(self.values)) != len(self.values):
            raise ValueError("values must contain no duplicates")
        self.values = list(self.values)

    @property
    def children(self) -> Iterable["LeftsNode"]:
        return [self.child]


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


@dataclass
class Ensemble(LeftsNode):
    name: str
    models: Iterable[LeftsNode]
    aggregate_with: Callable[..., Expr] | None = None

    @property
    def children(self):
        return self.models


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


@dataclass
class Feed(LeftsNode):
    name: str
    source: LeftsNode
    consumer: LeftsNode

    @property
    def children(self) -> Iterable["LeftsNode"]:
        return [self.source, self.consumer]
