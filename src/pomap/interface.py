from dataclasses import dataclass
from typing import Any, Callable, Iterable

from polars import DataFrame

from .interpreter import _collect_labels, _fit, _Model, _print_tree
from .nodes import Ensemble, Feed, Leaf, LearnsFrom, Lift


@dataclass
class Model(_Model):
    def fit(self, df: DataFrame):
        models, hyperparameters = _fit(self.root, df)
        self.models = models
        self.hyperparameters = hyperparameters

    def print_tree(self):
        print(_print_tree(self.root))

    def view_labels_dataframe(self) -> DataFrame:
        raise NotImplementedError()

    def collect_labels(self) -> Iterable[str]:
        return _collect_labels(self.root)


def leaf(model_constructor: Callable[..., Any], label: str) -> Model:
    leaf_node = Leaf(label=label, factory=model_constructor)
    return Model(leaf_node)


def lift(
    model: Model,
    values,
    name,
    train_filter,
    test_filter,
    validation_filter=None,
) -> Model:
    lifted = Lift(
        child=model.root,
        values=values,
        name=name,
        train_filter=train_filter,
        test_filter=test_filter,
        validation_filter=validation_filter,
    )

    return Model(lifted)


def ensemble(name: str, *models):
    roots = [model.root for model in models]
    node = Ensemble(name, roots)

    return Model(node)


def learn_from(
    name, learner: Model, learns_from: Model, logic: Callable[[Model, DataFrame], dict]
):
    node = LearnsFrom(
        name=name, learner=learner.root, learns_from=learns_from.root, learn_logic=logic
    )

    return Model(node)


def feed(name: str, source: Model, consumer: Model) -> Model:
    node = Feed(name=name, source=source.root, consumer=consumer.root)

    return Model(node)
