from dataclasses import dataclass
from typing import Any, Callable, Iterable

from polars import DataFrame

from .interpreter.fit import _fit
from .interpreter.predict import _Model
from .interpreter.masks import _collect_masks
from .interpreter.labels import _print_tree, _collect_labels
from .nodes import Ensemble, Feed, Leaf, Tune, Lift, Split
from .validation import _validate


@dataclass
class Model(_Model):
    def __post_init__(self):
        _validate(self.root)

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

    def mark_train_validation_test_rows(self, df: DataFrame) -> DataFrame:
        """
        Annotate `df` with boolean columns describing whether each
        row belongs to the train, test and (if applicable) validation
        sets for each sub model.
        """
        new_cols = []
        for label, masks in _collect_masks(self.root).items():
            new_cols.append(masks["train"].alias(f"{label}__train"))
            new_cols.append(masks["test"].alias(f"{label}__test"))
            if masks["validation"] is not None:
                new_cols.append(masks["validation"].alias(f"{label}__validation"))
        return df.with_columns(new_cols)


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
    aggregate_with=None,
) -> Model:
    lifted = Lift(
        child=model.root,
        values=values,
        name=name,
        train_filter=train_filter,
        test_filter=test_filter,
        validation_filter=validation_filter,
        aggregate_with=aggregate_with,
    )

    return Model(lifted)


def split(
    name: str,
    model: Model,
    train_filter,
    test_filter,
    validation_filter=None,
) -> Model:
    node = Split(
        name=name,
        child=model.root,
        train_filter=train_filter,
        test_filter=test_filter,
        validation_filter=validation_filter,
    )

    return Model(node)


def ensemble(name: str, *models, aggregate_with=None):
    roots = [model.root for model in models]
    node = Ensemble(name, roots, aggregate_with=aggregate_with)

    return Model(node)


def tune(
    name: str, consumer: Model, source: Model, logic: Callable[[Model, DataFrame], dict]
):
    node = Tune(name=name, consumer=consumer.root, source=source.root, logic=logic)

    return Model(node)


def feed(name: str, source: Model, consumer: Model) -> Model:
    node = Feed(name=name, source=source.root, consumer=consumer.root)

    return Model(node)
