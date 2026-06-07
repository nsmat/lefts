import warnings
import pytest
from dataclasses import dataclass
import polars as pl

from pomap.interface import leaf, lift, split, feed


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
    """Acts as a passthrough for the teacher, so we can inspect
    Whatever the teacher was feeding"""

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


def test_feed_with_lift_in_source_and_consumer():
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "fold": [0, 0, 1, 1, 2, 2, 3, 3],
        }
    )

    teacher = leaf(lambda: MockModel(x_column="x"), "teacher")
    student = leaf(lambda: ConsumerModel(source_col="teacher_signal"), "student")

    # The following trains a 'leak free' teacher signal
    # For fold 0, the teacher trains on folds 1, 2, 3
    # And gives predictions on fold 0.

    # Likewise For fold 1, the student trains on folds 0, 2, 3
    # It's teacher signal on fold 1 rows will be predicted
    # only from data that was available to fold 1 at test time
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

    # Each teacher trained on rows where fold != v — no leakage.
    # fold=0 leaf trains on x in [3,4,5,6,7,8], mean = 5.5
    assert model.models["teacher[teacher_signal=0]"].value == 5.5
    # fold=3 leaf trains on x in [1,2,3,4,5,6], mean = 3.5
    assert model.models["teacher[teacher_signal=3]"].value == 3.5
