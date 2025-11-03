from .interpreter import _Model, _fit, _print_tree, _collect_labels
from dataclasses import dataclass
from polars import DataFrame
from .nodes import Lift, Ensemble, LearnsFrom, Leaf
from typing import Callable, Iterable, Any
from .label import Label


@dataclass
class Model(_Model):
    def fit(self, df: DataFrame):
        models, hyperparameters = _fit(self.root, df)
        self.models = models
        self.hyperparameters = hyperparameters

    def print_tree(self):
        _print_tree(self.root)

    def view_labels_dataframe(self) -> DataFrame: ...  # TODO

    def collect_labels(self) -> Iterable[Label]:
        return _collect_labels(self.root)


def ready(model_constructor: Callable[..., Any], label: str) -> Model:
    leaf_node = Leaf(label=label, factory=model_constructor)
    return Model(leaf_node)


def lift(
    model: Model, atomics, name, train_mask_for_label, test_mask_for_label
) -> Model:
    lifted = Lift(
        child=model.root,
        atomics=atomics,
        name=name,
        train_mask_for_label=train_mask_for_label,
        test_mask_for_label=test_mask_for_label,
    )

    return Model(lifted)


def ensemble(*models):
    node = Ensemble(models)

    return Model(node)


def learn_from(learner: Model, learns_from, logic: Callable[[Model, DataFrame], dict]):
    node = LearnsFrom(
        learner=learner.root, learns_from=learns_from.root, learn_logic=logic
    )

    return Model(node)
