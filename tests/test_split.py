import pytest
from dataclasses import dataclass
import polars as pl

from pomap.nodes import Lift, Leaf, Split, Feed
from pomap.interpreter import _fit, _predict, _collect_labels


@pytest.fixture
def test_dataframe():
    df_a = pl.DataFrame({"x": [2.0, 3.0, 4.0], "category": ["a", "a", "a"]})
    df_b = pl.DataFrame({"x": [10.0, 15.0, 20.0], "category": ["b", "b", "b"]})
    df_c = pl.DataFrame({"x": [4.0, 6.0, 8.0], "category": ["c", "c", "c"]})
    return pl.concat([df_a, df_b, df_c])


@dataclass
class MockModel:
    x_column: str
    value: float = None

    def fit(self, training_set: pl.DataFrame):
        self.value = training_set[self.x_column].mean()

    def predict(self, df: pl.DataFrame):
        return [self.value] * df.shape[0]


@pytest.fixture
def model_x():
    return Leaf(label="model-x", factory=lambda: MockModel(x_column="x"))


def test_labels_split_not_decorated(model_x):
    """Split must not extend the label context."""
    node = Split(
        name="tt",
        child=model_x,
        train_filter=pl.col("x") < 10,
        test_filter=pl.lit(True),
    )
    assert set(_collect_labels(node)) == {"model-x"}


def test_fit_split_applies_train_filter(model_x, test_dataframe):
    node = Split(
        name="tt",
        child=model_x,
        train_filter=pl.col("x") < 10,
        test_filter=pl.lit(True),
    )
    models, _ = _fit(node, test_dataframe)
    # train: x in [2, 3, 4, 4, 6, 8], mean = 4.5
    assert models["model-x"].value == 4.5


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


def test_split_inside_lift(model_x, test_dataframe):
    """Lift extends label context; Split inside refines the row mask without further decoration."""
    inner = Split(
        name="tt",
        child=model_x,
        train_filter=pl.col("x") > 3,
        test_filter=pl.lit(True),
    )
    outer = Lift(
        name="category",
        child=inner,
        values=["a"],
        train_filter=lambda v: pl.col("category") == pl.lit(v),
        test_filter=lambda v: pl.col("category") == pl.lit(v),
    )
    models, _ = _fit(outer, test_dataframe)

    assert set(models.keys()) == {"model-x[category=a]"}
    # train: category=a AND x>3 → only x=4.0
    assert models["model-x[category=a]"].value == 4.0


def test_lift_inside_split(model_x, test_dataframe):
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
        train_filter=pl.col("x") > 2,
        test_filter=pl.lit(True),
    )
    models, _ = _fit(outer, test_dataframe)

    assert set(models.keys()) == {"model-x[category=a]"}
    # train: x>2 AND category=a → x in [3, 4], mean = 3.5
    assert models["model-x[category=a]"].value == 3.5


def test_split_validation_filter(test_dataframe):
    @dataclass
    class ValModel:
        train_mean: float = None
        val_mean: float = None

        def fit(self, training_set, validation_set):
            self.train_mean = training_set["x"].mean()
            self.val_mean = validation_set["x"].mean()

        def predict(self, df):
            return [self.train_mean] * len(df)

    leaf_node = Leaf(label="m", factory=lambda: ValModel())
    node = Split(
        name="tt",
        child=leaf_node,
        train_filter=pl.col("x") < 5,
        test_filter=pl.col("x") >= 10,
        validation_filter=pl.col("x").is_in([6.0, 8.0]),
    )
    models, _ = _fit(node, test_dataframe)

    # train: x in [2,3,4,4], mean = 3.25
    assert models["m"].train_mean == 3.25
    # val: x in [6, 8], mean = 7.0
    assert models["m"].val_mean == 7.0


def test_split_wrapping_feed_consumer_sees_bare_source_column(test_dataframe):
    """Consumer references the source's prediction column by literal name.

    Documents that Split does NOT decorate labels, so the consumer's literal
    column reference still works. Also documents the known leakage: the source
    fits on all rows because Feed re-roots `_fit`, dropping Split's filter.
    """
    source_leaf = Leaf(label="source", factory=lambda: MockModel(x_column="x"))

    @dataclass
    class SourcePlusXConsumer:
        value: float = None

        def fit(self, training_set):
            self.value = (training_set["x"] + training_set["source"]).mean()

        def predict(self, df):
            return [self.value] * len(df)

    consumer_leaf = Leaf(label="consumer", factory=lambda: SourcePlusXConsumer())
    feed_node = Feed(name="d", source=source_leaf, consumer=consumer_leaf)
    node = Split(
        name="tt",
        child=feed_node,
        train_filter=pl.col("x") < 10,
        test_filter=pl.lit(True),
    )
    models, _ = _fit(node, test_dataframe)

    # Both stored under bare labels — Split doesn't decorate.
    assert set(models.keys()) == {"source", "consumer"}

    # Source LEAKS: trains on all 9 rows (Feed's fresh-root drops Split's filter).
    # Mean x = 72 / 9 = 8.0
    assert models["source"].value == 8.0

    # Consumer respects Split's train mask: x < 10 → x in [2,3,4,4,6,8], n=6.
    # "source" column on those rows = 8.0 (full-data source predicts everywhere).
    # consumer.value = mean(x + source) over those rows = (2+3+4+4+6+8)/6 + 8.0
    #                = 4.5 + 8.0 = 12.5
    assert models["consumer"].value == 12.5