import warnings
import pytest
from dataclasses import dataclass
import polars as pl

from pomap.interface import leaf, split, feed


@dataclass
class MockModel:
    x_column: str
    value: float = None

    def fit(self, training_set: pl.DataFrame):
        self.value = training_set[self.x_column].mean()

    def predict(self, df: pl.DataFrame):
        return [self.value] * len(df)


@dataclass
class ConsumerModel:
    """Reads the source column during fit; predicts whatever it learned."""

    source_col: str
    value: float = None

    def fit(self, training_set: pl.DataFrame):
        self.value = training_set[self.source_col].mean()

    def predict(self, df: pl.DataFrame):
        return [self.value] * len(df)


@pytest.fixture
def test_dataframe():
    df_a = pl.DataFrame({"x": [2.0, 3.0, 4.0]})
    df_b = pl.DataFrame({"x": [10.0, 15.0, 20.0]})
    return pl.concat([df_a, df_b])


def test_plain_feed_no_warnings(test_dataframe):
    src = leaf(lambda: MockModel(x_column="x"), "src")
    cons = leaf(lambda: ConsumerModel(source_col="src"), "cons")
    model = feed("d", source=src, consumer=cons)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(test_dataframe)


def test_split_above_feed_no_leakage(test_dataframe):
    src = leaf(lambda: MockModel(x_column="x"), "src")
    cons = leaf(lambda: ConsumerModel(source_col="src"), "cons")
    inner = feed("d", source=src, consumer=cons)
    model = split(
        "tt",
        inner,
        train_filter=pl.col("x") < 10,
        test_filter=pl.col("x") >= 10,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # NaN-augmentation warning is expected here
        model.fit(test_dataframe)

    # Train rows are x<10: [2, 3, 4], mean = 3.0. Source must NOT have seen x>=10.
    assert model.models["src"].value == 3.0


def test_split_above_feed_warns_nan_augmentation(test_dataframe):
    """consumer.train ⊄ source.test under shared Split → augmentation NaN warning."""
    src = leaf(lambda: MockModel(x_column="x"), "src")
    cons = leaf(lambda: ConsumerModel(source_col="src"), "cons")
    inner = feed("d", source=src, consumer=cons)
    model = split(
        "tt",
        inner,
        train_filter=pl.col("x") < 10,
        test_filter=pl.col("x") >= 10,
    )

    with pytest.warns(
        UserWarning, match="rows in consumer's train set are not in source's test set"
    ):
        model.fit(test_dataframe)


def test_asymmetric_source_consumer_warns_leak(test_dataframe):
    """source.train ⊋ consumer.train → potential-leak warning."""
    teacher_leaf = leaf(lambda: MockModel(x_column="x"), "teacher")
    student_leaf = leaf(lambda: ConsumerModel(source_col="teacher"), "student")

    # Source trains on all rows; consumer trains on x<10 only and tests on x>=10.
    # This is the leakage shape: source has seen consumer's test rows.
    src = teacher_leaf
    cons = split(
        "cons_tt",
        student_leaf,
        train_filter=pl.col("x") < 10,
        test_filter=pl.col("x") >= 10,
    )
    model = feed("d", source=src, consumer=cons)

    with pytest.warns(
        UserWarning,
        match="source's train set contains .* rows not in consumer's train set",
    ):
        model.fit(test_dataframe)
