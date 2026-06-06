from dataclasses import dataclass
from typing import Any, Callable, Iterable

from polars import DataFrame

from .interpreter import _fit, _Model, _print_tree, _collect_labels, _collect_masks
from .nodes import Ensemble, Feed, Leaf, LearnsFrom, Lift, Split
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
        for label, train_mask, validation_mask, test_mask in _collect_masks(self.root):
            new_cols.append(train_mask.alias(f"{label}__train"))
            new_cols.append(test_mask.alias(f"{label}__test"))
            if validation_mask is not None:
                new_cols.append(validation_mask.alias(f"{label}__validation"))
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
