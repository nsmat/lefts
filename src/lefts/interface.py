import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal

from polars import DataFrame, Expr

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

    def fit(
        self,
        df: DataFrame,
        logging: Literal["print", "capture", "drop"] = "print",
        errors: Literal["raise", "store"] = "raise",
    ):
        """
        Fit every leaf model in the tree.

        Parameters
        ----------
        logging
            How to handle each leaf model's stdout/stderr: ``'print'`` leaves it
            untouched, ``'drop'`` discards it, ``'capture'`` collects it into
            ``self.logs`` keyed by model label.
        errors
            How to handle a leaf model that raises during fit: ``'raise'`` halts the
            whole fit, ``'capture'`` records the exception in ``self.exceptions`` keyed
            by model label and continues fitting the remaining models.
        """
        logs, exceptions = {}, {}
        models, hyperparameters = _fit(
            self.root,
            df,
            logging=logging,
            errors=errors,
            logs=logs,
            exceptions=exceptions,
        )
        self.models = models
        self.hyperparameters = hyperparameters
        self.logs = logs
        self.exceptions = exceptions

        if exceptions:
            warnings.warn(
                f"{len(exceptions)} model(s) failed to train: {sorted(exceptions)}",
                UserWarning,
                stacklevel=2,
            )

    def print_tree(self, print_all_labels: bool = False):
        print(_print_tree(self.root, print_all_labels=print_all_labels))

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
    """
    Converts a model into the format required for transformation by lefts.

    Parameters
    ----------
    model_constructor
        A constructor that creates a model with ``.fit()`` and ``.predict()`` methods.
    label
        A label for keeping track of this model.
    """

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
    """
    Creates multiple copies of a model that are trained on (possibly overlapping) train, test and validation sets.

    Parameters
    ----------
    model
        A lefts Model object.
    values
        Values to lift the model over. One copy of the model will be trained for each value.
    name
        The name of the lift transformation. Has no effect on model training, but controls how the resulting
        models are labelled and addressed: each leaf beneath the lift gets a label of the form
        ``"<leaf label>[<name>=<value>]"``. When ``aggregate_with`` is set, the per-value columns are instead
        collapsed into a single output column named ``name``.
    train_filter
        A function mapping each value in ``values`` to a boolean Polars expression indicating whether a given
        row is in the train set associated with that value.
    test_filter
        A function mapping each value in ``values`` to a boolean Polars expression indicating whether a given
        row is in the test set associated with that value.
    validation_filter
        A function mapping each value in ``values`` to a boolean Polars expression indicating whether a given
        row is in the validation set associated with that value.
    aggregate_with
        A function that postprocesses the output columns of the lift. It is called on the set of columns
        output by the lifted ``.predict()``.
    """
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
    train_filter: Expr,
    test_filter: Expr,
    validation_filter: Expr | None = None,
) -> Model:
    """
    Restricts a model to train, test and (optionally) validate on defined subsets of the available data.

    Parameters
    ----------
    name
        A name used to keep track of this lefts operation in the workflow. Has no effect on model training.
    model
        A lefts Model object.
    train_filter
        A boolean Polars expression that indicates whether a given row is in the train set.
    test_filter
        A boolean Polars expression that indicates whether a given row is in the test set.
    validation_filter
        A boolean Polars expression that indicates whether a given row is in the validation set.
    """
    node = Split(
        name=name,
        child=model.root,
        train_filter=train_filter,
        test_filter=test_filter,
        validation_filter=validation_filter,
    )

    return Model(node)


def ensemble(name: str, *models, aggregate_with=None):
    """
    Binds multiple models into a unified model that fits and predicts all of them in parallel.

    Parameters
    ----------
    name
        A name used to keep track of this lefts operation in the workflow. Has no effect on model training.
    models
        lefts Model objects.
    aggregate_with
        A function that postprocesses the output columns of the ensemble ``.predict()`` method.
    """
    roots = [model.root for model in models]
    node = Ensemble(name, roots, aggregate_with=aggregate_with)

    return Model(node)


def tune(
    name: str, consumer: Model, source: Model, logic: Callable[[Model, DataFrame], dict]
):
    """
    Learn hyperparameters by fitting the source model, applying customisable logic, then passing the resulting
    dictionary of hyperparameters to the consumer.

    Parameters
    ----------
    name
        A name used to keep track of this lefts operation in the workflow. Has no effect on model training.
    consumer
        A lefts Model object. Its leaf factories are instantiated using the outputs of ``logic`` as keyword
        arguments.
    source
        A lefts Model object. It is fitted first; the fitted model is then handed to ``logic`` to derive the
        hyperparameters.
    logic
        A callable ``(fitted_source_model, df) -> dict`` that reads the fitted source and returns the
        hyperparameters to apply when fitting the consumer.
    """
    node = Tune(name=name, consumer=consumer.root, source=source.root, logic=logic)

    return Model(node)


def feed(name: str, source: Model, consumer: Model) -> Model:
    """
    Chains two models: the source's predictions are available to the consumer as a feature or target during .fit and .predict.
    Parameters
    ----------
    name: A name used to keep track of this lefts operation in the workflow. Has no effect on model training.
    source: A lefts Model object, which provides the output of its predict to the consumer.
    consumer: A lefts Model object, which has access to the prediction output of the consumer.

    Returns
    -------

    """
    node = Feed(name=name, source=source.root, consumer=consumer.root)

    return Model(node)
