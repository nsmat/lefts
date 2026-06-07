import polars as pl

from pomap.nodes import Lift, Leaf, Split, Ensemble, Feed
from pomap.interpreter import _collect_labels


# ── Lift ──────────────────────────────────────────────────────────


def test_labels_lift_decorates_per_value(lift_x):
    assert set(_collect_labels(lift_x)) == {
        "model-x[category=a]",
        "model-x[category=b]",
        "model-x[category=c]",
    }


# ── Split ─────────────────────────────────────────────────────────


def test_labels_split_not_decorated(model_x):
    node = Split(
        name="tt",
        child=model_x,
        train_filter=pl.col("x") < 10,
        test_filter=pl.lit(True),
    )
    assert set(_collect_labels(node)) == {"model-x"}


# ── Ensemble ──────────────────────────────────────────────────────


def test_labels_ensemble_yields_each_child(ensemble_x1_x2):
    assert set(_collect_labels(ensemble_x1_x2)) == {"model-x", "model-x2"}


# ── Aggregation halting (Lift + Ensemble) ────────────────────────


def test_labels_aggregate_halts(model_x, model_x2):
    inner = Ensemble(
        name="inner",
        models=[model_x, model_x2],
        aggregate_with=pl.mean_horizontal,
    )
    outer = Lift(
        name="fold",
        child=inner,
        values=[1, 2, 3],
        train_filter=lambda v: pl.lit(True),
        test_filter=lambda v: pl.lit(True),
        aggregate_with=pl.mean_horizontal,
    )
    assert set(_collect_labels(outer)) == {"fold"}
    assert set(_collect_labels(inner)) == {"inner"}


# ── Feed ──────────────────────────────────────────────────────────


def test_labels_feed_yields_source_and_consumer():
    source_leaf = Leaf(label="source", factory=lambda: None)
    consumer_leaf = Leaf(label="consumer", factory=lambda: None)
    node = Feed(name="test_feed", source=source_leaf, consumer=consumer_leaf)
    assert set(_collect_labels(node)) == {"source", "consumer"}
