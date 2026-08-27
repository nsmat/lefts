import warnings
import pytest
import polars as pl

from lefts.interface import leaf, split, feed
from conftest import MockModel, ConsumerModel


def test_plain_feed_no_warnings(test_dataframe):
    """A plain Feed with no row filters should fit without emitting any warning.
    Predict-side correctness for this shape is covered by test_predict_feed_source_then_consumer."""
    src = leaf(lambda: MockModel(x_column="x"), "src")
    cons = leaf(lambda: ConsumerModel(source_col="src"), "cons")
    model = feed("d", source=src, consumer=cons)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(test_dataframe)


def test_split_above_feed_warns_nan_augmentation(test_dataframe):
    """
    Test that when the test set of the source is a subset of the train set of the
    consumer, we generate a warning (since this will leave NaNs in fed features, which
    may be problematic).
    """
    src = leaf(lambda: MockModel(x_column="x"), "src")
    cons = leaf(lambda: ConsumerModel(source_col="src"), "cons")
    inner = feed("d", source=src, consumer=cons)
    model = split(
        "tt",
        inner,
        train_filter=pl.col("x") < 5,
        test_filter=pl.col("x") >= 5,
    )

    with pytest.warns(
        UserWarning, match="rows in consumer's train set are not in source's test set"
    ):
        model.fit(test_dataframe)


def test_asymmetric_source_consumer_warns_leak(test_dataframe):
    """source.train ⊋ consumer.train → potential-leak warning."""
    teacher_leaf = leaf(lambda: MockModel(x_column="x"), "teacher")
    student_leaf = leaf(lambda: ConsumerModel(source_col="teacher"), "student")

    # Source trains on all rows; consumer trains on x<5 only and tests on x>=5.
    model = feed(
        "d",
        source=teacher_leaf,
        consumer=split(
            "cons_tt",
            student_leaf,
            train_filter=pl.col("x") < 5,
            test_filter=pl.col("x") >= 5,
        ),
    )

    with pytest.warns(
        UserWarning,
        match="source's train set contains .* rows not in consumer's train set",
    ):
        model.fit(test_dataframe)
