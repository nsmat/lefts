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


def test_asymmetric_source_consumer_warns_leak(test_dataframe):
    """source.train ⊋ consumer.train → potential-leak warning."""
    teacher_leaf = leaf(lambda: MockModel(x_column="x"), "teacher")
    student_leaf = leaf(lambda: ConsumerModel(source_col="teacher"), "student")

    # Source trains on all rows; consumer trains on x<5 only and tests on x>=5.
    # This is the leakage shape: source has seen consumer's test rows.
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


def test_feed_with_lift_in_source_and_consumer():
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7, 8],
            "fold": [0, 0, 1, 1, 2, 2, 3, 3],
        }
    )

    teacher = leaf(lambda: MockModel(x_column="x"), "teacher")
    student = leaf(lambda: ConsumerModel(source_col="teacher_signal"), "student")

    # CV cross-fitting via Lift inside source. Each teacher fold v trains on
    # `fold != v` and predicts on `fold == v`; the coalesce produces a single
    # column of OOF predictions covering all rows.
    source = lift(
        teacher,
        name="teacher_signal",
        values=[0, 1, 2, 3],
        train_filter=lambda v: pl.col("fold") != v,
        test_filter=lambda v: pl.col("fold") == v,
        aggregate_with=pl.coalesce,
    )
    consumer = lift(
        student,
        name="cv_fold",
        values=[0, 1, 2, 3],
        train_filter=lambda v: pl.col("fold") != v,
        test_filter=lambda v: pl.col("fold") == v,
    )

    model = feed("d", source=source, consumer=consumer)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(df)

    expected_keys = {f"teacher[teacher_signal={v}]" for v in range(4)} | {
        f"student[cv_fold={v}]" for v in range(4)
    }
    assert set(model.models.keys()) == expected_keys

    # Each teacher trained only on rows where fold != v.
    assert model.models["teacher[teacher_signal=0]"].seen == [3, 4, 5, 6, 7, 8]
    assert model.models["teacher[teacher_signal=3]"].seen == [1, 2, 3, 4, 5, 6]

    # OOF check on the student side: for each of student[cv_fold=0]'s training
    # rows, the teacher_signal it saw was produced by a teacher that excluded
    # that row, so the row's own x value must NOT appear in the teacher_signal
    # list it was trained on.
    student_seen = model.models["student[cv_fold=0]"].seen
    student_train_xs = df.filter(pl.col("fold") != 0)["x"].to_list()
    for x_val, teacher_signal_list in zip(student_train_xs, student_seen):
        assert x_val not in teacher_signal_list
