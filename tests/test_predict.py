from dataclasses import dataclass
import polars as pl

from pomap.nodes import Lift, Leaf, Split, Ensemble, LearnsFrom, Feed
from pomap.interpreter import _fit, _predict
from conftest import MockModel, ConsumerModel


# ── Lift ──────────────────────────────────────────────────────────


def test_predict_lift_per_value_columns(lift_x, test_dataframe):
    models, _ = _fit(lift_x, test_dataframe)
    predictions = _predict(lift_x, models, test_dataframe)

    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    for cat in ["a", "b", "c"]:
        rows = predictions.filter(category=cat)
        col = f"model-x[category={cat}]"
        assert rows[col].to_list() == [expected[cat]] * rows.height


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

    expected_training = [1, 2, 3, 4]
    on_test = predictions.filter(pl.col("x") >= 5)
    assert on_test["model-x"].to_list() == [expected_training] * on_test.height

    off_test = predictions.filter(pl.col("x") < 5)
    assert off_test["model-x"].is_null().all()


# ── Ensemble ──────────────────────────────────────────────────────


def test_predict_ensemble_separate_columns(test_dataframe):
    a = Leaf(label="model-a", factory=lambda: MockModel(x_column="x"))
    b = Leaf(label="model-b", factory=lambda: MockModel(x_column="x"))
    ensemble = Ensemble(name="ens", models=[a, b])
    models, _ = _fit(ensemble, test_dataframe)
    predictions = _predict(ensemble, models, test_dataframe)

    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert predictions["model-a"].to_list() == [expected] * 9
    assert predictions["model-b"].to_list() == [expected] * 9


def test_predict_ensemble_aggregate_collapses(test_dataframe):
    a = Leaf(label="model-a", factory=lambda: MockModel(x_column="x"))
    b = Leaf(label="model-b", factory=lambda: MockModel(x_column="x"))
    ensemble = Ensemble(
        name="combined",
        models=[a, b],
        aggregate_with=pl.coalesce,
    )
    models, _ = _fit(ensemble, test_dataframe)
    predictions = _predict(ensemble, models, test_dataframe)

    assert "combined" in predictions.columns
    assert "model-a" not in predictions.columns
    assert "model-b" not in predictions.columns

    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert predictions["combined"].to_list() == [expected] * 9


def test_predict_ensemble_nested_aggregates(test_dataframe):
    inner_a = Leaf(label="inner-a", factory=lambda: MockModel(x_column="x"))
    inner_b = Leaf(label="inner-b", factory=lambda: MockModel(x_column="x"))
    outer_c = Leaf(label="outer-c", factory=lambda: MockModel(x_column="x"))

    inner = Ensemble(
        name="inner",
        models=[inner_a, inner_b],
        aggregate_with=pl.coalesce,
    )
    outer = Ensemble(
        name="outer",
        models=[inner, outer_c],
        aggregate_with=pl.coalesce,
    )

    models, _ = _fit(outer, test_dataframe)
    predictions = _predict(outer, models, test_dataframe)

    for intermediate in ("inner", "inner-a", "inner-b", "outer-c"):
        assert intermediate not in predictions.columns
    assert "outer" in predictions.columns

    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert predictions["outer"].to_list() == [expected] * 9


# ── Feed ──────────────────────────────────────────────────────────


def test_predict_feed_source_then_consumer(test_dataframe):
    source_leaf = Leaf(label="source", factory=lambda: MockModel(x_column="x"))
    consumer_leaf = Leaf(
        label="consumer", factory=lambda: ConsumerModel(source_col="source")
    )
    node = Feed(name="test_feed", source=source_leaf, consumer=consumer_leaf)
    models, _ = _fit(node, test_dataframe)
    predictions = _predict(node, models, test_dataframe)

    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert predictions["source"].to_list() == [expected] * 9
    assert predictions["consumer"].to_list() == [expected] * 9


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
    preds = model.predict(df)
    training_values = preds["source"][0]
    return {"offset": sum(training_values) / len(training_values)}


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

    # learner.value = mean(x) + offset = 5.0 + 5.0 = 10.0
    assert (predictions["learner"] == 10.0).all()
