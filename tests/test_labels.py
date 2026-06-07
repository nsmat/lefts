import polars as pl

from pomap.nodes import Lift, Leaf, Split, Ensemble, Feed, LearnsFrom
from pomap.interpreter import _collect_labels


# ── Lift ──────────────────────────────────────────────────────────


def test_labels_lift_decorates_per_value(lift_x):
    assert set(_collect_labels(lift_x)) == {
        "model-x[category=a]",
        "model-x[category=b]",
        "model-x[category=c]",
    }


def test_labels_nested_lifts_compose_dimensions(model_x):
    inner = Lift(
        name="sign",
        child=model_x,
        values=["pos"],
        train_filter=lambda v: pl.col("x") > 0,
        test_filter=lambda v: pl.col("x") > 0,
    )
    outer = Lift(
        name="category",
        child=inner,
        values=["a"],
        train_filter=lambda v: pl.col("category") == pl.lit(v),
        test_filter=lambda v: pl.col("category") == pl.lit(v),
    )
    # Outer Lift's dimension appears first, inner's second.
    assert set(_collect_labels(outer)) == {"model-x[category=a, sign=pos]"}


# ── Split ─────────────────────────────────────────────────────────


def test_labels_split_not_decorated(model_x):
    node = Split(
        name="tt",
        child=model_x,
        train_filter=pl.col("x") < 5,
        test_filter=pl.lit(True),
    )
    assert set(_collect_labels(node)) == {"model-x"}


# ── Ensemble ──────────────────────────────────────────────────────


def test_labels_ensemble_yields_each_child():
    a = Leaf(label="model-a", factory=lambda: None)
    b = Leaf(label="model-b", factory=lambda: None)
    ensemble = Ensemble(name="ens", models=[a, b])
    assert set(_collect_labels(ensemble)) == {"model-a", "model-b"}


# ── Aggregation halting (Lift + Ensemble) ────────────────────────


def test_labels_aggregate_halts():
    a = Leaf(label="model-a", factory=lambda: None)
    b = Leaf(label="model-b", factory=lambda: None)
    inner = Ensemble(
        name="inner",
        models=[a, b],
        aggregate_with=pl.coalesce,
    )
    outer = Lift(
        name="fold",
        child=inner,
        values=[1, 2, 3],
        train_filter=lambda v: pl.lit(True),
        test_filter=lambda v: pl.lit(True),
        aggregate_with=pl.coalesce,
    )
    assert set(_collect_labels(outer)) == {"fold"}
    assert set(_collect_labels(inner)) == {"inner"}


# ── Feed ──────────────────────────────────────────────────────────


def test_labels_feed_yields_source_and_consumer():
    source_leaf = Leaf(label="source", factory=lambda: None)
    consumer_leaf = Leaf(label="consumer", factory=lambda: None)
    node = Feed(name="test_feed", source=source_leaf, consumer=consumer_leaf)
    assert set(_collect_labels(node)) == {"source", "consumer"}


# ── LearnsFrom ────────────────────────────────────────────────────


def test_labels_learns_from_yields_both_subtrees():
    source_leaf = Leaf(label="source", factory=lambda: None)
    learner_leaf = Leaf(label="learner", factory=lambda: None)
    node = LearnsFrom(
        name="lf",
        learner=learner_leaf,
        learns_from=source_leaf,
        learn_logic=lambda model, df: {},
    )
    assert set(_collect_labels(node)) == {"source", "learner"}
