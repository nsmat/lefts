from dataclasses import dataclass
import polars as pl

from pomap.nodes import Lift, Leaf, Split, LearnsFrom, Feed
from pomap.interpreter import _fit
from conftest import MockModel, ConsumerModel


# ── Lift ──────────────────────────────────────────────────────────


def test_fit_lift_fans_out(lift_x, test_dataframe):
    models, _ = _fit(lift_x, test_dataframe)
    assert models["model-x[category=a]"].value == 3
    assert models["model-x[category=b]"].value == 15
    assert models["model-x[category=c]"].value == 6


def test_fit_lift_nested_decorates_labels(model_x, test_dataframe):
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
    assert set(models.keys()) == {"model-x[category=a, sign=pos]"}


# ── Split ─────────────────────────────────────────────────────────


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


def test_fit_split_validation_passthrough(test_dataframe):
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
    assert models["m"].train_mean == 3.25
    assert models["m"].val_mean == 7.0


# Composition-flavored fit tests — may migrate to test_commutativity.py later
def test_fit_split_inside_lift(model_x, test_dataframe):
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
        train_filter=pl.col("x") > 2,
        test_filter=pl.lit(True),
    )
    models, _ = _fit(outer, test_dataframe)
    assert set(models.keys()) == {"model-x[category=a]"}
    # train: x>2 AND category=a → x in [3, 4], mean = 3.5
    assert models["model-x[category=a]"].value == 3.5


# ── Feed ──────────────────────────────────────────────────────────


def test_fit_feed_basic(test_dataframe):
    source_leaf = Leaf(label="source", factory=lambda: MockModel(x_column="x"))
    consumer_leaf = Leaf(
        label="consumer", factory=lambda: ConsumerModel(source_col="source")
    )
    node = Feed(name="test_feed", source=source_leaf, consumer=consumer_leaf)
    models, _ = _fit(node, test_dataframe)
    # source trains on all x: mean = 8.0
    assert models["source"].value == 8.0
    # consumer trains on the augmented df where the "source" column is 8.0 everywhere
    assert models["consumer"].value == 8.0


def test_fit_feed_augmentation_is_used(test_dataframe):
    @dataclass
    class SummingConsumer:
        value: float = None

        def fit(self, training_set):
            self.value = training_set["x"].mean() + training_set["source"].mean()

        def predict(self, df):
            return [self.value] * len(df)

    source_leaf = Leaf(label="source", factory=lambda: MockModel(x_column="x"))
    consumer_leaf = Leaf(label="consumer", factory=lambda: SummingConsumer())
    node = Feed(name="test", source=source_leaf, consumer=consumer_leaf)
    models, _ = _fit(node, test_dataframe)
    # x mean = 8.0, source prediction mean = 8.0, so consumer.value = 16.0
    assert models["consumer"].value == 16.0


# ── LearnsFrom ────────────────────────────────────────────────────


@dataclass
class _OffsetModel:
    offset: float
    value: float = None

    def fit(self, training_set):
        self.value = training_set["x"].mean() + self.offset

    def predict(self, df):
        return [self.value] * len(df)


def test_fit_learns_from_threads_hyperparameters(test_dataframe):
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
    models, hyperparameters = _fit(node, test_dataframe)
    assert models["source"].value == 8.0
    assert hyperparameters["offset"] == 8.0
    assert models["learner"].value == 16.0
