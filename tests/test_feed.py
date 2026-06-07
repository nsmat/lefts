import warnings
import pytest
import polars as pl
from polars.testing import assert_frame_equal

from pomap.interface import leaf, lift, split, feed
from conftest import MockModel, ConsumerModel


def test_plain_feed_no_warnings(test_dataframe):
    src = leaf(lambda: MockModel(x_column="x"), "src")
    cons = leaf(lambda: ConsumerModel(source_col="src"), "cons")
    model = feed("d", source=src, consumer=cons)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Will crash (failing test) if fitting throws any warnings

        model.fit(test_dataframe)

    predictions = model.predict(test_dataframe)
    distinct = predictions.select("src", "cons").unique()

    # Given both source and consumer are leaves, they train on
    # The entire dataframe (x \in [1, ..., 9]
    expected = pl.DataFrame(
        {
            "src": [[1, 2, 3, 4, 5, 6, 7, 8, 9]],
            "cons": [[1, 2, 3, 4, 5, 6, 7, 8, 9]],
        },
        schema={"src": pl.List(pl.Int64), "cons": pl.List(pl.Int64)},
    )
    assert_frame_equal(distinct, expected)


def test_split_above_feed_no_leakage(test_dataframe):
    src = leaf(lambda: MockModel(x_column="x"), "src")
    cons = leaf(lambda: ConsumerModel(source_col="src"), "cons")
    inner = feed("d", source=src, consumer=cons)
    model = split(
        "tt",
        inner,
        train_filter=pl.col("x") < 5,
        test_filter=pl.col("x") >= 5,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # NaN-augmentation warning is expected here - we ignore it
        model.fit(test_dataframe)

    # Source's training data is exactly the Split's train rows; nothing leaked in.
    assert model.models["src"].seen == [1, 2, 3, 4]

    predictions = model.predict(test_dataframe)
    distinct = (
        predictions.with_columns(in_test=pl.col("x") >= 5)
        .select("in_test", "src", "cons")
        .unique(subset=["in_test"], maintain_order=True)
    )
    expected = pl.DataFrame(
        {
            "in_test": [False, True],
            "src": [None, [1, 2, 3, 4]],
            "cons": [None, [1, 2, 3, 4]],
        },
        schema={
            "in_test": pl.Boolean,
            "src": pl.List(pl.Int64),
            "cons": pl.List(pl.Int64),
        },
    )
    assert_frame_equal(distinct, expected)


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


def test_feed_with_lift_in_source_and_consumer(test_dataframe):
    teacher = leaf(lambda: MockModel(x_column="x"), "teacher")
    student = leaf(lambda: ConsumerModel(source_col="teacher_signal"), "student")

    # CV cross-fitting via Lift inside source. Each teacher fold v trains on
    # `fold != v` and predicts on `fold == v`; the coalesce produces a single
    # column of out-of-fold predictions covering all rows.
    source = lift(
        teacher,
        name="teacher_signal",
        values=[0, 1, 2],
        train_filter=lambda v: pl.col("fold") != v,
        test_filter=lambda v: pl.col("fold") == v,
        aggregate_with=pl.coalesce,
    )
    consumer = lift(
        student,
        name="cv_fold",
        values=[0, 1, 2],
        train_filter=lambda v: pl.col("fold") != v,
        test_filter=lambda v: pl.col("fold") == v,
    )

    model = feed("d", source=source, consumer=consumer)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(test_dataframe)

    # Fit-side: each teacher trained only on rows where fold != v.
    assert model.models["teacher[teacher_signal=0]"].seen == [4, 5, 6, 7, 8, 9]
    assert model.models["teacher[teacher_signal=1]"].seen == [1, 2, 3, 7, 8, 9]
    assert model.models["teacher[teacher_signal=2]"].seen == [1, 2, 3, 4, 5, 6]

    # student[cv_fold=0] trains on fold=1 and fold=2 rows; teacher_signal per
    # row is the training list of whichever teacher was responsible for that
    # fold, and no row's own x appears in the list it saw (OOF).
    assert model.models["student[cv_fold=0]"].seen == (
        [[1, 2, 3, 7, 8, 9]] * 3  # fold=1 rows ← teacher[teacher_signal=1].seen
        + [[1, 2, 3, 4, 5, 6]] * 3  # fold=2 rows ← teacher[teacher_signal=2].seen
    )

    # Predict-side currently fails — `_predict` is a flat loop and runs
    # `_apply_aggregations` only at the end, so the consumer's `.predict` reads
    # `pl.col("teacher_signal")` before that column has been produced. Tracked
    # as "Predict-time aggregation ordering for Feed" in CLAUDE.md. Leaving the
    # call in deliberately so the test fails until the fix lands.
    predictions = model.predict(test_dataframe)
    distinct = (
        predictions.select(
            "fold",
            "teacher_signal",
            "student[cv_fold=0]",
            "student[cv_fold=1]",
            "student[cv_fold=2]",
        ).unique(subset=["fold"], maintain_order=True)
    )
    expected = pl.DataFrame(
        {
            "fold": [0, 1, 2],
            "teacher_signal": [
                [4, 5, 6, 7, 8, 9],
                [1, 2, 3, 7, 8, 9],
                [1, 2, 3, 4, 5, 6],
            ],
            "student[cv_fold=0]": [[4, 5, 6, 7, 8, 9], None, None],
            "student[cv_fold=1]": [None, [1, 2, 3, 7, 8, 9], None],
            "student[cv_fold=2]": [None, None, [1, 2, 3, 4, 5, 6]],
        },
        schema={
            "fold": pl.Int64,
            "teacher_signal": pl.List(pl.Int64),
            "student[cv_fold=0]": pl.List(pl.Int64),
            "student[cv_fold=1]": pl.List(pl.Int64),
            "student[cv_fold=2]": pl.List(pl.Int64),
        },
    )
    assert_frame_equal(distinct, expected)