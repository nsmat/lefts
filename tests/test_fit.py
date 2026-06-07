from dataclasses import dataclass
import polars as pl

from pomap.nodes import Lift, Leaf, Split, LearnsFrom, Feed
from pomap.interpreter import _fit
from conftest import MockModel, ConsumerModel


# ── Lift ──────────────────────────────────────────────────────────


def test_fit_lift_fans_out(lift_x, test_dataframe):
    models, _ = _fit(lift_x, test_dataframe)
    assert models["model-x[category=a]"].seen == [2, 3, 4]
    assert models["model-x[category=b]"].seen == [10, 15, 20]
    assert models["model-x[category=c]"].seen == [4, 6, 8]


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
    assert models["model-x"].seen == [2, 3, 4, 4, 6, 8]


def test_fit_split_validation_passthrough(test_dataframe):
    @dataclass
    class ValModel:
        train_seen: list = None
        val_seen: list = None

        def fit(self, training_set, validation_set):
            self.train_seen = training_set["x"].to_list()
            self.val_seen = validation_set["x"].to_list()

        def predict(self, df):
            return [self.train_seen] * len(df)

    leaf_node = Leaf(label="m", factory=lambda: ValModel())
    node = Split(
        name="tt",
        child=leaf_node,
        train_filter=pl.col("x") < 5,
        test_filter=pl.col("x") >= 10,
        validation_filter=pl.col("x").is_in([6, 8]),
    )
    models, _ = _fit(node, test_dataframe)
    assert models["m"].train_seen == [2, 3, 4, 4]
    assert models["m"].val_seen == [6, 8]


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
    # train: category=a AND x>3 → only x=4
    assert models["model-x[category=a]"].seen == [4]


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
    # train: x>2 AND category=a → x in [3, 4]
    assert models["model-x[category=a]"].seen == [3, 4]


# ── Feed ──────────────────────────────────────────────────────────


def test_fit_feed_basic(test_dataframe):
    source_leaf = Leaf(label="source", factory=lambda: MockModel(x_column="x"))
    consumer_leaf = Leaf(
        label="consumer", factory=lambda: ConsumerModel(source_col="source")
    )
    node = Feed(name="test_feed", source=source_leaf, consumer=consumer_leaf)
    models, _ = _fit(node, test_dataframe)

    expected_source_training = [2, 3, 4, 10, 15, 20, 4, 6, 8]
    assert models["source"].seen == expected_source_training
    # Consumer's training rows each received the source's training list as their
    # "source" feature, so its `.seen` is N copies of that list.
    assert models["consumer"].seen == [expected_source_training] * 9


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


def test_fit_learns_from_threads_hyperparameters(test_dataframe):
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
    models, hyperparameters = _fit(node, test_dataframe)

    # source's training data is the full x column
    assert models["source"].seen == [2, 3, 4, 10, 15, 20, 4, 6, 8]
    # learn_logic computes the mean of source's training data → 72/9 = 8.0
    assert hyperparameters["offset"] == 8.0
    # learner.value = mean(x) + offset = 8.0 + 8.0 = 16.0
    assert models["learner"].value == 16.0
