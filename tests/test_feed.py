import warnings
import pytest
import polars as pl

from pomap.interface import leaf, lift, split, feed
from conftest import MockModel, ConsumerModel


def test_plain_feed_no_warnings(test_dataframe):
    src = leaf(lambda: MockModel(x_column="x"), "src")
    cons = leaf(lambda: ConsumerModel(source_col="src"), "cons")
    model = feed("d", source=src, consumer=cons)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(test_dataframe)

    predictions = model.predict(test_dataframe)
    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert predictions["src"].to_list() == [expected] * 9
    assert predictions["cons"].to_list() == [expected] * 9


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
        warnings.simplefilter("ignore")  # NaN-augmentation warning is expected here
        model.fit(test_dataframe)

    # Source's training data is exactly the Split's train rows; nothing leaked in.
    assert model.models["src"].seen == [1, 2, 3, 4]

    predictions = model.predict(test_dataframe)
    expected = [1, 2, 3, 4]
    on_test = predictions.filter(pl.col("x") >= 5)
    assert on_test["src"].to_list() == [expected] * 5
    assert on_test["cons"].to_list() == [expected] * 5
    off_test = predictions.filter(pl.col("x") < 5)
    assert off_test["src"].is_null().all()
    assert off_test["cons"].is_null().all()


def test_split_above_feed_warns_nan_augmentation(test_dataframe):
    """consumer.train ⊄ source.test under shared Split → augmentation NaN warning."""
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

    # Predict still runs cleanly — only fit-time augmentation is affected by the NaN gap.
    predictions = model.predict(test_dataframe)
    expected = [1, 2, 3, 4]
    on_test = predictions.filter(pl.col("x") >= 5)
    assert on_test["cons"].to_list() == [expected] * 5


def test_asymmetric_source_consumer_warns_leak(test_dataframe):
    """source.train ⊋ consumer.train → potential-leak warning."""
    teacher_leaf = leaf(lambda: MockModel(x_column="x"), "teacher")
    student_leaf = leaf(lambda: ConsumerModel(source_col="teacher"), "student")

    # Source trains on all rows; consumer trains on x<5 only and tests on x>=5.
    src = teacher_leaf
    cons = split(
        "cons_tt",
        student_leaf,
        train_filter=pl.col("x") < 5,
        test_filter=pl.col("x") >= 5,
    )
    model = feed("d", source=src, consumer=cons)

    with pytest.warns(
        UserWarning,
        match="source's train set contains .* rows not in consumer's train set",
    ):
        model.fit(test_dataframe)

    predictions = model.predict(test_dataframe)
    expected_all = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert predictions["teacher"].to_list() == [expected_all] * 9
    on_test = predictions.filter(pl.col("x") >= 5)
    assert on_test["student"].to_list() == [expected_all] * 5
    off_test = predictions.filter(pl.col("x") < 5)
    assert off_test["student"].is_null().all()


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

    # student[cv_fold=0] trains on fold=1 and fold=2 rows. For each of those rows
    # the teacher_signal value is the training list of whichever teacher was
    # responsible for that fold; no row's own x appears in the list it saw (OOF).
    assert model.models["student[cv_fold=0]"].seen == (
        [[1, 2, 3, 7, 8, 9]] * 3  # fold=1 rows (x=4,5,6) ← teacher[teacher_signal=1].seen
        + [[1, 2, 3, 4, 5, 6]] * 3  # fold=2 rows (x=7,8,9) ← teacher[teacher_signal=2].seen
    )

    # Predict-side: currently broken — `_predict` is a flat loop and runs
    # `_apply_aggregations` only at the end, so the consumer's `.predict` tries
    # to read `pl.col("teacher_signal")` before that column has been produced.
    # Tracked as "Predict-time aggregation ordering for Feed" in CLAUDE.md.
    # This call is left in deliberately so the test fails until the fix lands.
    predictions = model.predict(test_dataframe)
    # Once predict works, each row's teacher_signal should be the OOF list for
    # that row's fold, and the student's prediction should be the same list
    # (passthrough).
    assert predictions["teacher_signal"].to_list() == [
        [4, 5, 6, 7, 8, 9],  # fold=0 rows
        [4, 5, 6, 7, 8, 9],
        [4, 5, 6, 7, 8, 9],
        [1, 2, 3, 7, 8, 9],  # fold=1 rows
        [1, 2, 3, 7, 8, 9],
        [1, 2, 3, 7, 8, 9],
        [1, 2, 3, 4, 5, 6],  # fold=2 rows
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
    ]