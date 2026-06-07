from dataclasses import dataclass
import polars as pl

from pomap.nodes import Lift, Leaf, Split, Ensemble, LearnsFrom, Feed
from pomap.interpreter import _fit, _predict
from conftest import MockModel, ConsumerModel


# ── Lift ──────────────────────────────────────────────────────────


def test_predict_lift_per_value_columns(lift_x, test_dataframe):
    models, _ = _fit(lift_x, test_dataframe)
    predictions = _predict(lift_x, models, test_dataframe)

    expected = {"a": 3, "b": 15, "c": 6}
    for cat in ["a", "b", "c"]:
        unique = predictions.filter(category=cat)[f"model-x[category={cat}]"].unique()
        assert unique.shape[0] == 1
        assert unique.item(0) == expected[cat]


# ── Split ─────────────────────────────────────────────────────────


def test_predict_split_applies_test_filter(model_x, test_dataframe):
    node = Split(
        name="tt",
        child=model_x,
        train_filter=pl.col("x") < 10,
        test_filter=pl.col("x") >= 10,
    )
    models, _ = _fit(node, test_dataframe)
    predictions = _predict(node, models, test_dataframe)

    on_test = predictions.filter(pl.col("x") >= 10)
    assert (on_test["model-x"] == 4.5).all()
    off_test = predictions.filter(pl.col("x") < 10)
    assert off_test["model-x"].is_null().all()


# ── Ensemble ──────────────────────────────────────────────────────


def test_predict_ensemble_separate_columns(ensemble_x1_x2, test_dataframe):
    models, _ = _fit(ensemble_x1_x2, test_dataframe)
    predictions = _predict(ensemble_x1_x2, models, test_dataframe)
    assert (predictions["model-x"] == 8.0).all()
    assert (predictions["model-x2"] == -8.0).all()


def test_predict_ensemble_aggregate_collapses(model_x, model_x2, test_dataframe):
    ensemble = Ensemble(
        name="avg",
        models=[model_x, model_x2],
        aggregate_with=pl.mean_horizontal,
    )
    models, _ = _fit(ensemble, test_dataframe)
    predictions = _predict(ensemble, models, test_dataframe)
    assert (predictions["avg"] == 0.0).all()
    assert "model-x" not in predictions.columns
    assert "model-x2" not in predictions.columns


def test_predict_ensemble_nested_aggregates(test_dataframe):
    inner_x = Leaf(label="inner-x", factory=lambda: MockModel(x_column="x"))
    inner_x2 = Leaf(label="inner-x2", factory=lambda: MockModel(x_column="x2"))
    outer_x = Leaf(label="outer-x", factory=lambda: MockModel(x_column="x"))

    inner = Ensemble(
        name="inner",
        models=[inner_x, inner_x2],
        aggregate_with=pl.mean_horizontal,
    )
    outer = Ensemble(
        name="outer",
        models=[inner, outer_x],
        aggregate_with=pl.mean_horizontal,
    )

    models, _ = _fit(outer, test_dataframe)
    predictions = _predict(outer, models, test_dataframe)
    assert (predictions["outer"] == 4.0).all()
    for intermediate in ("inner", "inner-x", "inner-x2", "outer-x"):
        assert intermediate not in predictions.columns


# ── Feed ──────────────────────────────────────────────────────────


def test_predict_feed_source_then_consumer(test_dataframe):
    source_leaf = Leaf(label="source", factory=lambda: MockModel(x_column="x"))
    consumer_leaf = Leaf(
        label="consumer", factory=lambda: ConsumerModel(source_col="source")
    )
    node = Feed(name="test_feed", source=source_leaf, consumer=consumer_leaf)
    models, _ = _fit(node, test_dataframe)
    predictions = _predict(node, models, test_dataframe)
    assert (predictions["source"] == 8.0).all()
    assert (predictions["consumer"] == 8.0).all()


# ── LearnsFrom ────────────────────────────────────────────────────


@dataclass
class _OffsetModel:
    offset: float
    value: float = None

    def fit(self, training_set):
        self.value = training_set["x"].mean() + self.offset

    def predict(self, df):
        return [self.value] * len(df)


def test_predict_learns_from(test_dataframe):
    source_leaf = Leaf(label="source", factory=lambda: MockModel(x_column="x"))
    learner_leaf = Leaf(
        label="learner",
        factory=lambda offset=0.0: _OffsetModel(offset=offset),
    )

    def learn_logic(model, df):
        preds = model.predict(df)
        return {"offset": preds["source"].mean()}

    node = LearnsFrom(
        name="test",
        learner=learner_leaf,
        learns_from=source_leaf,
        learn_logic=learn_logic,
    )
    models, _ = _fit(node, test_dataframe)
    predictions = _predict(node, models, test_dataframe)
    assert (predictions["source"] == 8.0).all()
    assert (predictions["learner"] == 16.0).all()
