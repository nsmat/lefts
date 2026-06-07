from dataclasses import dataclass
import polars as pl
from polars.testing import assert_frame_equal

from pomap.nodes import Lift, Leaf, Split, Ensemble, LearnsFrom, Feed
from pomap.interpreter import _fit, _predict
from conftest import MockModel, ConsumerModel


# ── Lift ──────────────────────────────────────────────────────────


def test_predict_lift_per_value_columns(lift_x, test_dataframe):
    models, _ = _fit(lift_x, test_dataframe)
    predictions = _predict(lift_x, models, test_dataframe)

    # All rows within a category have identical prediction columns, so we can
    # deduplicate to one row per category.
    distinct = predictions.select(
        "category",
        "model-x[category=a]",
        "model-x[category=b]",
        "model-x[category=c]",
    ).unique(subset=["category"], maintain_order=True)

    # Expected pattern is that for category i:
    # On rows where category = i we see all training data that belonged to category i
    # On rows where category != i we generate no prediction (None)
    expected = pl.DataFrame(
        {
            "category": ["a", "b", "c"],
            "model-x[category=a]": [[1, 2, 3], None, None],
            "model-x[category=b]": [None, [4, 5, 6], None],
            "model-x[category=c]": [None, None, [7, 8, 9]],
        },
        schema={
            "category": pl.String,
            "model-x[category=a]": pl.List(pl.Int64),
            "model-x[category=b]": pl.List(pl.Int64),
            "model-x[category=c]": pl.List(pl.Int64),
        },
    )
    assert_frame_equal(distinct, expected)


# ── Split ─────────────────────────────────────────────────────────


def test_predict_split_applies_test_filter(model_x, test_dataframe):
    node = Split(
        name="tt",
        child=model_x,
        train_filter=pl.col("x") < 5,
        test_filter=pl.col("x") >= 5,
    )
    models, _ = _fit(node, test_dataframe)
    predictions = _predict(node, models, test_dataframe)

    # We add a column that says whether the row was in the test set
    # So that we can simplify the output down to in test/not in test.
    distinct = (
        predictions
        .with_columns(in_test=pl.col("x") >= 5)
        .select("in_test", "model-x")
        .unique(subset=["in_test"], maintain_order=True)
    )

    # The expected pattern is that we produce None outside the test set
    # And reproduce the training data on the test set.
    expected = pl.DataFrame(
        {
            "in_test": [False, True],
            "model-x": [None, [1, 2, 3, 4]],
        },
        schema={"in_test": pl.Boolean, "model-x": pl.List(pl.Int64)},
    )
    assert_frame_equal(distinct, expected)


# ── Ensemble ──────────────────────────────────────────────────────


def test_predict_ensemble_separate_columns(test_dataframe):
    a = Leaf(label="model-a", factory=lambda: MockModel(x_column="x"))
    b = Leaf(label="model-b", factory=lambda: MockModel(x_column="x"))
    ensemble = Ensemble(name="ens", models=[a, b])
    models, _ = _fit(ensemble, test_dataframe)
    predictions = _predict(ensemble, models, test_dataframe)

    distinct = predictions.select("model-a", "model-b").unique()

    # The expectation is that both models produce the train data everywhere
    expected = pl.DataFrame(
        {
            "model-a": [[1, 2, 3, 4, 5, 6, 7, 8, 9]],
            "model-b": [[1, 2, 3, 4, 5, 6, 7, 8, 9]],
        },
        schema={"model-a": pl.List(pl.Int64), "model-b": pl.List(pl.Int64)},
    )
    assert_frame_equal(distinct, expected)


def test_predict_ensemble_aggregate_collapses(test_dataframe):
    # In this test we sum across the ensemble, so the final output
    # Should be double the x-column, on every row.
    a = Leaf(label="model-a", factory=lambda: MockModel(x_column="x"))
    b = Leaf(label="model-b", factory=lambda: MockModel(x_column="x"))
    ensemble = Ensemble(
        name="combined",
        models=[a, b],
        aggregate_with=pl.sum_horizontal,
    )
    models, _ = _fit(ensemble, test_dataframe)
    predictions = _predict(ensemble, models, test_dataframe)

    assert "model-a" not in predictions.columns
    assert "model-b" not in predictions.columns

    distinct = predictions.select("combined").unique()
    expected = pl.DataFrame(
        {"combined": [[2, 4, 6, 8, 10, 12, 14, 16, 18]]},
        schema={"combined": pl.List(pl.Int64)},
    )
    assert_frame_equal(distinct, expected)


def test_predict_ensemble_nested_aggregates(test_dataframe):
    # We aggregate one ensemble by summing, then the second, providing
    # 3x the input column as the final prediction.
    inner_a = Leaf(label="inner-a", factory=lambda: MockModel(x_column="x"))
    inner_b = Leaf(label="inner-b", factory=lambda: MockModel(x_column="x"))
    outer_c = Leaf(label="outer-c", factory=lambda: MockModel(x_column="x"))

    inner = Ensemble(
        name="inner", models=[inner_a, inner_b], aggregate_with=pl.sum_horizontal
    )
    outer = Ensemble(
        name="outer", models=[inner, outer_c], aggregate_with=pl.sum_horizontal
    )

    models, _ = _fit(outer, test_dataframe)
    predictions = _predict(outer, models, test_dataframe)

    for intermediate in ("inner", "inner-a", "inner-b", "outer-c"):
        assert intermediate not in predictions.columns

    distinct = predictions.select("outer").unique()
    expected = pl.DataFrame(
        {"outer": [[3, 6, 9, 12, 15, 18, 21, 24, 27]]},
        schema={"outer": pl.List(pl.Int64)},
    )
    assert_frame_equal(distinct, expected)


# ── Feed ──────────────────────────────────────────────────────────


def test_predict_feed_source_then_consumer(test_dataframe):
    source_leaf = Leaf(label="source", factory=lambda: MockModel(x_column="x"))
    consumer_leaf = Leaf(
        label="consumer", factory=lambda: ConsumerModel(source_col="source")
    )
    node = Feed(name="test_feed", source=source_leaf, consumer=consumer_leaf)
    models, _ = _fit(node, test_dataframe)
    predictions = _predict(node, models, test_dataframe)

    distinct = predictions.select("source", "consumer").unique()
    expected = pl.DataFrame(
        {
            "source": [[1, 2, 3, 4, 5, 6, 7, 8, 9]],
            "consumer": [[1, 2, 3, 4, 5, 6, 7, 8, 9]],
        },
        schema={"source": pl.List(pl.Int64), "consumer": pl.List(pl.Int64)},
    )
    assert_frame_equal(distinct, expected)


# ── LearnsFrom ────────────────────────────────────────────────────


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


def test_predict_learns_from(test_dataframe):
    source_leaf = Leaf(label="source", factory=lambda: MockModel(x_column="x"))
    learner_leaf = Leaf(
        label="learner",
        factory=lambda offset=0.0: _OffsetModel(offset=offset),
    )

    node = LearnsFrom(
        name="test",
        learner=learner_leaf,
        learns_from=source_leaf,
        learn_logic=_mean_of_source_training_data,
    )
    models, _ = _fit(node, test_dataframe)
    predictions = _predict(node, models, test_dataframe)

    distinct = predictions.select("source", "learner").unique()

    # Expectation is that source sees the training data.
    # The learner should see the mean of the training data + the mean of the training data
    # = 5 + 5 = 10
    expected = pl.DataFrame(
        {
            "source": [[1, 2, 3, 4, 5, 6, 7, 8, 9]],
            "learner": [10.0],
        },
        schema={"source": pl.List(pl.Int64), "learner": pl.Float64},
    )
    assert_frame_equal(distinct, expected)