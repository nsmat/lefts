import sys
from dataclasses import dataclass

import pytest

from lefts.nodes import Leaf, Ensemble, Feed, Tune
from lefts.interpreter.fit import _fit, UpstreamFitFailure
from conftest import MockModel, ConsumerModel


@dataclass
class _NoisyModel:
    message: str

    def fit(self, training_set):
        print(self.message)
        print(f"{self.message}-err", file=sys.stderr)

    def predict(self, df):
        return [0] * len(df)


@dataclass
class _FailingModel:
    error_message: str = "boom"

    def fit(self, training_set):
        raise RuntimeError(self.error_message)

    def predict(self, df):
        return [0] * len(df)


def _noisy_ensemble():
    a = Leaf(label="model-a", factory=lambda: _NoisyModel(message="hello-a"))
    b = Leaf(label="model-b", factory=lambda: _NoisyModel(message="hello-b"))
    return Ensemble(name="ens", models=[a, b])


# ── logging ───────────────────────────────────────────────────────────


def test_fit_logging_capture_collects_output_by_label(test_dataframe, capsys):
    _, _, logs, _ = _fit(_noisy_ensemble(), test_dataframe, logging="capture")

    assert "hello-a" in logs["model-a"]
    assert "hello-a-err" in logs["model-a"]
    assert "hello-b" in logs["model-b"]

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


def test_fit_rejects_unknown_logging_mode(test_dataframe):
    leaf = Leaf(label="m", factory=lambda: MockModel(x_column="x"))
    with pytest.raises(ValueError, match="logging"):
        _fit(leaf, test_dataframe, logging="bogus")


# ── errors ────────────────────────────────────────────────────────────


def test_fit_errors_capture_records_and_continues(test_dataframe):
    good = Leaf(label="good", factory=lambda: MockModel(x_column="x"))
    bad = Leaf(label="bad", factory=lambda: _FailingModel(error_message="nope"))
    node = Ensemble(name="ens", models=[good, bad])

    models, _, _, exceptions = _fit(node, test_dataframe, errors="capture")

    assert models["good"].seen == [1, 2, 3, 4, 5, 6, 7, 8, 9]
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
    consumer_leaf = Leaf(label="consumer", factory=lambda: MockModel(x_column="x"))
    node = Tune(
        name="test_tune",
        consumer=consumer_leaf,
        source=source_leaf,
        logic=lambda model, df: {},
    )

    models, _, _, exceptions = _fit(node, test_dataframe, errors="capture")

    assert "consumer" not in models
    assert isinstance(exceptions["source"], RuntimeError)
    assert isinstance(exceptions["test_tune"], UpstreamFitFailure)


def test_fit_rejects_unknown_errors_mode(test_dataframe):
    leaf = Leaf(label="m", factory=lambda: MockModel(x_column="x"))
    with pytest.raises(ValueError, match="errors"):
        _fit(leaf, test_dataframe, errors="bogus")
