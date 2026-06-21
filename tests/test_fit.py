from dataclasses import dataclass
import sys

import polars as pl
import pytest

from lefts.nodes import Lift, Leaf, Split, Ensemble, Tune, Feed
from lefts.interpreter.fit import _fit, UpstreamFitFailure
from conftest import MockModel, ConsumerModel


@dataclass
class _NoisyModel:
    """A model that writes to stdout and stderr while fitting."""

    message: str

    def fit(self, training_set):
        print(self.message)
        print(f"{self.message}-err", file=sys.stderr)

    def predict(self, df):
        return [0] * len(df)


@dataclass
class _FailingModel:
    """A model whose fit always raises."""

    error_message: str = "boom"

    def fit(self, training_set):
        raise RuntimeError(self.error_message)

    def predict(self, df):
        return [0] * len(df)


# ── Lift ──────────────────────────────────────────────────────────


def test_fit_lift_fans_out(lift_x, test_dataframe):
    models, *_ = _fit(lift_x, test_dataframe)
    assert models["model-x[category=a]"].seen == [1, 2, 3]
    assert models["model-x[category=b]"].seen == [4, 5, 6]
    assert models["model-x[category=c]"].seen == [7, 8, 9]


# ── Split ─────────────────────────────────────────────────────────


def test_fit_split_applies_train_filter(model_x, test_dataframe):
    node = Split(
        name="tt",
        child=model_x,
        train_filter=pl.col("x") < 5,
        test_filter=pl.lit(True),
    )
    models, *_ = _fit(node, test_dataframe)
    assert models["model-x"].seen == [1, 2, 3, 4]


def test_fit_split_validation_passthrough(test_dataframe):
    leaf_node = Leaf(label="m", factory=lambda: MockModel(x_column="x"))
    node = Split(
        name="tt",
        child=leaf_node,
        train_filter=pl.col("x") < 5,
        test_filter=pl.col("x") >= 8,
        validation_filter=pl.col("x").is_between(5, 7, closed="both"),
    )
    models, *_ = _fit(node, test_dataframe)
    assert models["m"].seen == [1, 2, 3, 4]
    assert models["m"].val_seen == [5, 6, 7]


# TODO: this tests composition, not fit behaviour itself - let's move to test_composition.py later
def test_fit_split_inside_lift(model_x, test_dataframe):
    inner = Split(
        name="tt",
        child=model_x,
        train_filter=pl.col("x") > 1,
        test_filter=pl.lit(True),
    )
    outer = Lift(
        name="category",
        child=inner,
        values=["a"],
        train_filter=lambda v: pl.col("category") == pl.lit(v),
        test_filter=lambda v: pl.col("category") == pl.lit(v),
    )
    models, *_ = _fit(outer, test_dataframe)
    assert set(models.keys()) == {"model-x[category=a]"}
    # Train filters resolve to category==a (1, 2, 3) AND x>1, implies x in [2, 3]
    assert models["model-x[category=a]"].seen == [2, 3]


# TODO this tests composition - not fit behaviour itself - let's move to test_composition.py later
def test_fit_lift_inside_split(model_x, test_dataframe):
    inner = Lift(
        name="category",
        child=model_x,
        values=["a"],
        train_filter=lambda v: pl.col("category") == pl.lit(v),
        test_filter=lambda v: pl.col("category") == pl.lit(v),
    )
    outer = Split(
        name="tt",
        child=inner,
        train_filter=pl.col("x") > 1,
        test_filter=pl.lit(True),
    )
    models, *_ = _fit(outer, test_dataframe)
    assert set(models.keys()) == {"model-x[category=a]"}
    # train: x>1 AND category=a → x in [2, 3]
    assert models["model-x[category=a]"].seen == [2, 3]


# ── Feed ──────────────────────────────────────────────────────────


def test_fit_feed_basic(test_dataframe):
    source_leaf = Leaf(label="source", factory=lambda: MockModel(x_column="x"))
    consumer_leaf = Leaf(
        label="consumer", factory=lambda: ConsumerModel(source_col="source")
    )
    node = Feed(name="test_feed", source=source_leaf, consumer=consumer_leaf)
    models, *_ = _fit(node, test_dataframe)

    expected_source_training = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert models["source"].seen == expected_source_training
    assert models["consumer"].seen == [expected_source_training]


# ── Tune ────────────────────────────────────────────────────


@dataclass
class _OffsetModel:
    offset: float
    value: float = None

    def fit(self, training_set):
        self.value = training_set["x"].mean() + self.offset

    def predict(self, df):
        return [self.value] * len(df)


def _mean_of_source_training_data(model, df):
    return {"offset": model.predict(df)["source"].list.mean().first()}


def test_fit_tune_threads_hyperparameters(test_dataframe):
    source_leaf = Leaf(label="source", factory=lambda: MockModel(x_column="x"))
    consumer_leaf = Leaf(
        label="consumer",
        factory=lambda offset=0.0: _OffsetModel(offset=offset),
    )

    node = Tune(
        name="test",
        consumer=consumer_leaf,
        source=source_leaf,
        logic=_mean_of_source_training_data,
    )
    models, hyperparameters, *_ = _fit(node, test_dataframe)

    # source's training data is the full x column
    assert models["source"].seen == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # logic computes the mean of source's training data → 45/9 = 5.0
    assert hyperparameters["offset"] == 5.0
    # consumer.value = mean(x) + offset = 5.0 + 5.0 = 10.0
    assert models["consumer"].value == 10.0


# ── Ensemble ──────────────────────────────────────────────────────────


def test_fit_ensemble_fits_each_child(test_dataframe):
    a = Leaf(label="model-a", factory=lambda: MockModel(x_column="x"))
    b = Leaf(label="model-b", factory=lambda: MockModel(x_column="x"))
    node = Ensemble(name="ens", models=[a, b])
    models, *_ = _fit(node, test_dataframe)

    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert models["model-a"].seen == expected
    assert models["model-b"].seen == expected


# ── logging ───────────────────────────────────────────────────────────


def _noisy_ensemble():
    a = Leaf(label="model-a", factory=lambda: _NoisyModel(message="hello-a"))
    b = Leaf(label="model-b", factory=lambda: _NoisyModel(message="hello-b"))
    return Ensemble(name="ens", models=[a, b])


def test_fit_logging_capture_collects_output_by_label(test_dataframe, capsys):
    _, _, logs, _ = _fit(_noisy_ensemble(), test_dataframe, logging="capture")

    assert "hello-a" in logs["model-a"]
    assert "hello-a-err" in logs["model-a"]
    assert "hello-b" in logs["model-b"]
    # Captured output does not leak to the real streams.
    captured = capsys.readouterr()
    assert "hello-a" not in captured.out
    assert "hello-a" not in captured.err


def test_fit_logging_drop_suppresses_output(test_dataframe, capsys):
    _, _, logs, _ = _fit(_noisy_ensemble(), test_dataframe, logging="drop")

    assert logs == {}
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_fit_logging_print_is_default(test_dataframe, capsys):
    _fit(_noisy_ensemble(), test_dataframe)

    captured = capsys.readouterr()
    assert "hello-a" in captured.out
    assert "hello-a-err" in captured.err


# ── errors ────────────────────────────────────────────────────────────


def test_fit_errors_capture_records_and_continues(test_dataframe):
    good = Leaf(label="good", factory=lambda: MockModel(x_column="x"))
    bad = Leaf(label="bad", factory=lambda: _FailingModel(error_message="nope"))
    node = Ensemble(name="ens", models=[good, bad])

    models, _, _, exceptions = _fit(node, test_dataframe, errors="capture")

    # The good model still fits and is returned.
    assert models["good"].seen == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # The failing model is recorded and omitted from models.
    assert "bad" not in models
    assert isinstance(exceptions["bad"], RuntimeError)
    assert str(exceptions["bad"]) == "nope"


def test_fit_errors_raise_is_default(test_dataframe):
    bad = Leaf(label="bad", factory=lambda: _FailingModel())
    with pytest.raises(RuntimeError, match="boom"):
        _fit(bad, test_dataframe)


def test_fit_errors_capture_feed_skips_consumer_on_source_failure(test_dataframe):
    source_leaf = Leaf(label="source", factory=lambda: _FailingModel())
    consumer_leaf = Leaf(
        label="consumer", factory=lambda: ConsumerModel(source_col="source")
    )
    node = Feed(name="test_feed", source=source_leaf, consumer=consumer_leaf)

    models, _, _, exceptions = _fit(node, test_dataframe, errors="capture")

    assert "consumer" not in models
    assert isinstance(exceptions["source"], RuntimeError)
    assert isinstance(exceptions["test_feed"], UpstreamFitFailure)


def test_fit_errors_capture_tune_skips_consumer_on_source_failure(test_dataframe):
    source_leaf = Leaf(label="source", factory=lambda: _FailingModel())
    consumer_leaf = Leaf(
        label="consumer", factory=lambda offset=0.0: _OffsetModel(offset=offset)
    )
    node = Tune(
        name="test_tune",
        consumer=consumer_leaf,
        source=source_leaf,
        logic=_mean_of_source_training_data,
    )

    models, _, _, exceptions = _fit(node, test_dataframe, errors="capture")

    assert "consumer" not in models
    assert isinstance(exceptions["source"], RuntimeError)
    assert isinstance(exceptions["test_tune"], UpstreamFitFailure)

