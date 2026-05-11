import pytest
from dataclasses import dataclass
import polars as pl

from pomap.nodes import Lift, Leaf, Ensemble, LearnsFrom, Feed
from pomap.interpreter import _collect_labels, _fit, _predict


@pytest.fixture
def test_dataframe():
    df_a = pl.DataFrame({"x": [2.0, 3.0, 4.0], "category": ["a", "a", "a"]})

    df_b = pl.DataFrame({"x": [10.0, 15.0, 20.0], "category": ["b", "b", "b"]})

    df_c = pl.DataFrame({"x": [4.0, 6.0, 8.0], "category": ["c", "c", "c"]})

    df = pl.concat([df_a, df_b, df_c])
    df = df.with_columns(x2=-pl.col("x"))

    return df


@dataclass
class MockModel:
    x_column: str
    value = None

    def fit(self, training_set: pl.DataFrame):
        self.value = training_set[self.x_column].mean()

    def predict(self, df: pl.DataFrame):
        return [self.value] * df.shape[0]


@pytest.fixture
def leaf_factory():
    """Fixture returning a helper that can make Leaf objects for any column."""

    def _make_leaf(x_column: str):
        return Leaf(
            label=f"model-{x_column}", factory=lambda: MockModel(x_column=x_column)
        )

    return _make_leaf


@pytest.fixture
def model_x(leaf_factory):
    return leaf_factory("x")


@pytest.fixture
def model_x2(leaf_factory):
    return leaf_factory("x2")


@pytest.fixture
def lift_x(model_x):
    return Lift(
        name="category",
        child=model_x,
        values=["a", "b", "c"],
        train_filter=lambda atomic: pl.col("category") == pl.lit(atomic),
        test_filter=lambda atomic: pl.col("category") == pl.lit(atomic),
    )


@pytest.fixture
def ensemble_x1_x2(model_x, model_x2):
    return Ensemble(name="ensemble", models=[model_x, model_x2])


def test_labels_lift_x(lift_x):
    expected_labels_x1 = {
        "model-x[category=a]",
        "model-x[category=b]",
        "model-x[category=c]",
    }

    assert set(_collect_labels(lift_x)) == expected_labels_x1


def test_labels_ensemble(ensemble_x1_x2):
    expected_labels_ensemble = {"model-x", "model-x2"}

    assert set(_collect_labels(ensemble_x1_x2)) == expected_labels_ensemble


def test_fit_lift_x(lift_x, test_dataframe):
    models, _ = _fit(lift_x, test_dataframe)

    assert models["model-x[category=a]"].value == 3
    assert models["model-x[category=b]"].value == 15
    assert models["model-x[category=c]"].value == 6


def test_predict_lift_x(lift_x, test_dataframe):
    models, _ = _fit(lift_x, test_dataframe)
    predictions = _predict(lift_x, models, test_dataframe)

    expected = {"a": 3, "b": 15, "c": 6}
    for cat in ["a", "b", "c"]:
        unique = predictions.filter(category=cat)[f"model-x[category={cat}]"].unique()
        assert unique.shape[0] == 1
        assert unique.item(0) == expected[cat]


def test_predict_ensemble_x(ensemble_x1_x2, test_dataframe):
    models, _ = _fit(ensemble_x1_x2, test_dataframe)

    predictions = _predict(ensemble_x1_x2, models, test_dataframe)

    assert (predictions["model-x"] == 8.0).all()
    assert (predictions["model-x2"] == -8.0).all()


def test_fit_double_lift(model_x, test_dataframe):
    """
    Tests that model labels are well formed after
    sequential lifts.
    """
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
    models, _ = _fit(outer, test_dataframe)
    assert set(models.keys()) == set(_collect_labels(outer))
    assert "model-x[category=a, sign=pos]" in models


@dataclass
class OffsetModel:
    """Predicts the training mean of x plus a fixed offset."""

    offset: float
    value: float = None

    def fit(self, training_set: pl.DataFrame):
        self.value = training_set["x"].mean() + self.offset

    def predict(self, df: pl.DataFrame):
        return [self.value] * df.shape[0]


@pytest.fixture
def learns_from_node():
    source_leaf = Leaf(label="source", factory=lambda: MockModel(x_column="x"))
    learner_leaf = Leaf(
        label="learner",
        factory=lambda offset=0.0: OffsetModel(offset=offset),
    )

    def learn_logic(model, df):
        preds = model.predict(df)
        return {"offset": preds["source"].mean()}

    return LearnsFrom(
        name="test",
        learner=learner_leaf,
        learns_from=source_leaf,
        learn_logic=learn_logic,
    )


def test_fit_learns_from(learns_from_node, test_dataframe):
    models, hyperparameters = _fit(learns_from_node, test_dataframe)

    assert models["source"].value == 8.0
    assert hyperparameters["offset"] == 8.0
    assert models["learner"].value == 16.0


def test_predict_learns_from(learns_from_node, test_dataframe):
    models, _ = _fit(learns_from_node, test_dataframe)
    predictions = _predict(learns_from_node, models, test_dataframe)

    assert (predictions["source"] == 8.0).all()
    assert (predictions["learner"] == 16.0).all()


@dataclass
class ConsumerModel:
    """Trains on a named column produced by a source model."""

    source_col: str
    value: float = None

    def fit(self, training_set: pl.DataFrame):
        self.value = training_set[self.source_col].mean()

    def predict(self, df: pl.DataFrame):
        return [self.value] * len(df)


@pytest.fixture
def feed_node():
    source_leaf = Leaf(label="source", factory=lambda: MockModel(x_column="x"))
    consumer_leaf = Leaf(
        label="consumer", factory=lambda: ConsumerModel(source_col="source")
    )
    return Feed(name="test_feed", source=source_leaf, consumer=consumer_leaf)


def test_labels_feed(feed_node):
    assert set(_collect_labels(feed_node)) == {"source", "consumer"}


def test_fit_feed(feed_node, test_dataframe):
    models, _ = _fit(feed_node, test_dataframe)

    # source trains on all x: mean = 8.0
    assert models["source"].value == 8.0
    # consumer trains on the augmented df where the "source" column is 8.0 everywhere
    assert models["consumer"].value == 8.0


def test_predict_feed(feed_node, test_dataframe):
    models, _ = _fit(feed_node, test_dataframe)
    predictions = _predict(feed_node, models, test_dataframe)

    assert (predictions["source"] == 8.0).all()
    assert (predictions["consumer"] == 8.0).all()


def test_fit_feed_augmentation_is_used(test_dataframe):
    # Verifies that the consumer actually receives the source column during training.
    # Source predicts the mean of x (8.0). Consumer sums the means of x and source,
    # so consumer.value should be 16.0 — only achievable if the source column was present.
    source_leaf = Leaf(label="source", factory=lambda: MockModel(x_column="x"))
    consumer_leaf = Leaf(
        label="consumer",
        factory=lambda: ConsumerModel(source_col="source"),
    )

    # Patch ConsumerModel to sum x mean + source mean to make augmentation observable
    @dataclass
    class SummingConsumer:
        value: float = None

        def fit(self, training_set: pl.DataFrame):
            self.value = training_set["x"].mean() + training_set["source"].mean()

        def predict(self, df: pl.DataFrame):
            return [self.value] * len(df)

    source_leaf = Leaf(label="source", factory=lambda: MockModel(x_column="x"))
    consumer_leaf = Leaf(label="consumer", factory=lambda: SummingConsumer())
    node = Feed(name="test", source=source_leaf, consumer=consumer_leaf)

    models, _ = _fit(node, test_dataframe)
    # x mean = 8.0, source prediction mean = 8.0, so consumer.value = 16.0
    assert models["consumer"].value == 16.0
