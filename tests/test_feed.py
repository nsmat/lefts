import warnings
import pytest
import polars as pl
from polars.testing import assert_frame_equal

from pomap.interface import leaf, lift, split, feed
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
    """
    Tests combining lift with feed. The sources and consumers are each
    lifted separately, then passed through a feed node.
    """
    teacher = leaf(lambda: MockModel(x_column="x"), "teacher")
    student = leaf(lambda: ConsumerModel(source_col="cv_teacher"), "student")

    # CV cross-fitting via Lift inside source. Each teacher fold v trains on
    # `fold != v` and predicts on `fold == v`; the coalesce produces a single
    # column of out-of-fold predictions covering all rows.
    source = lift(
        teacher,
        name="cv_teacher",
        values=[0, 1, 2],
        train_filter=lambda v: pl.col("fold") != v,
        test_filter=lambda v: pl.col("fold") == v,
        aggregate_with=pl.coalesce,
    )
    consumer = lift(
        student,
        name="cv_student",
        values=[0, 1, 2],
        train_filter=lambda v: pl.col("fold") != v,
        test_filter=lambda v: pl.col("fold") == v,
    )

    model = feed("d", source=source, consumer=consumer)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(test_dataframe)

    # teacher[cv_teacher=i] trains on every row where fold != i — i.e. the data
    # with fold i excluded. That same list is what the teacher predicts on its
    # held-out rows (fold == i), so it's also what shows up in the `cv_teacher`
    # column for fold=i rows after the coalesce.
    data_excluding_fold_0 = [4, 5, 6, 7, 8, 9]
    data_excluding_fold_1 = [1, 2, 3, 7, 8, 9]
    data_excluding_fold_2 = [1, 2, 3, 4, 5, 6]

    assert model.models["teacher[cv_teacher=0]"].seen == data_excluding_fold_0
    assert model.models["teacher[cv_teacher=1]"].seen == data_excluding_fold_1
    assert model.models["teacher[cv_teacher=2]"].seen == data_excluding_fold_2


    # The student will see the predictions of the teacher - recall that the teacher
    # Just stores all the training data it saw. Hence, we expect that on each fold i
    # The student will be passed all the data from fold != i by the teacher.
    assert model.models["student[cv_student=0]"].seen == [
        data_excluding_fold_1,
        data_excluding_fold_2,
    ]

    # Predict-side currently fails — `_predict` is a flat loop and runs
    # `_apply_aggregations` only at the end, so the consumer's `.predict` reads
    # `pl.col("cv_teacher")` before that column has been produced. Tracked
    # as "Predict-time aggregation ordering for Feed" in CLAUDE.md. Leaving the
    # call in deliberately so the test fails until the fix lands.
    predictions = model.predict(test_dataframe)
    distinct = (
        predictions.select(
            "fold",
            "cv_teacher",
            "student[cv_student=0]",
            "student[cv_student=1]",
            "student[cv_student=2]",
        ).unique(subset=["fold"], maintain_order=True)
    )
    expected = pl.DataFrame(
        {
            "fold": [0, 1, 2],
            "cv_teacher": [
                data_excluding_fold_0,
                data_excluding_fold_1,
                data_excluding_fold_2,
            ],
            "student[cv_student=0]": [data_excluding_fold_0, None, None],
            "student[cv_student=1]": [None, data_excluding_fold_1, None],
            "student[cv_student=2]": [None, None, data_excluding_fold_2],
        },
        schema={
            "fold": pl.Int64,
            "cv_teacher": pl.List(pl.Int64),
            "student[cv_student=0]": pl.List(pl.Int64),
            "student[cv_student=1]": pl.List(pl.Int64),
            "student[cv_student=2]": pl.List(pl.Int64),
        },
    )
    assert_frame_equal(distinct, expected)