from polars import DataType, Expr, DataFrame
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional, Callable, Protocol, runtime_checkable

@runtime_checkable
class ModelProtocol(Protocol):

    def fit(self, df):
        ...

    def predict(self, df):
        ...


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
        return f'Leaf: {self.label}'


@dataclass
class Lift(PomapNode):
    name: str
    child: PomapNode
    atomics: Iterable[DataType]
    train_mask_for_label: Callable[[DataType], Expr]
    test_mask_for_label: Callable[[DataType], Expr]
    validation_mask_for_label: Callable[[DataType], Expr] = None

    def __post_init__(self):
        self.atomics = set(self.atomics)

    @property
    def children(self) -> Iterable["PomapNode"]:
        return [self.child]

    @property
    def tree_repr(self) -> str:
        return f"{self.name}: {self.atomics}"


@dataclass
class Ensemble(PomapNode):
    name: str
    models: Iterable[PomapNode]

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
        [PomapNode, DataFrame], dict
    ]  # TODO this should actually typehint Model instead of PomapNode, but need to re-organise dirs first

    @property
    def children(self) -> Iterable["PomapNode"]:
        return [self.learner, self.learns_from]

    @property
    def tree_repr(self) -> str:
        return f"LearnsFrom: {self.name}"
