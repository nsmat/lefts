"""
Composition tests: one node wrapping another, exercising fit AND predict together.

Every other test_*.py file tests a single node (or a single interpreter function) in
isolation. Those pass even when the *seam* between two nodes is broken - the Tune
train-mask leak was invisible to `test_masks_tune_passthrough` (masks were collected
correctly) and to `test_fit_tune_threads_hyperparameters` (a bare Tune has no outer
mask to propagate). This file covers that seam directly: build outer(inner(...)) from
MockModels, fit it, assert exactly which rows each leaf trained on, then predict and
assert the values flow through.

MockModel makes this legible: `.seen` is the training data it received, and `.predict`
echoes that same list back on its test rows, so a single tree tells us both what was
trained on and how it feeds prediction. The matrix below covers each ordered pair of
the five transforming nodes (Lift, Split, Ensemble, Feed, Tune) as outer x inner.
"""

import warnings
from dataclasses import dataclass

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from lefts.interface import leaf, lift, split, ensemble, tune, feed
from conftest import MockModel, ConsumerModel


@dataclass
class OffsetModel:
    """
    A Tune consumer whose fitted state exposes the applied hyperparameter:
    value = mean(training x) + offset. Makes it easy to see both which rows it
    trained on (via the mean) and which offset the Tune propagated.
    """

    offset: float
    value: float = None

    def fit(self, training_set):
        self.value = training_set["x"].mean() + self.offset

    def predict(self, df):
        return [self.value] * len(df)


def _mean_of_source(model, df):
    """Tune logic: read the source's predictions off the `source` column."""
    return {"offset": model.predict(df)["source"].list.mean().first()}


def _mean_of_t_src(model, df):
    """Tune logic for the feed-source-tune tree, whose source column is `t_src`."""
    return {"offset": model.predict(df)["t_src"].list.mean().first()}


FULL = [1, 2, 3, 4, 5, 6, 7, 8, 9]


# ── Split as outer ──────────────────────────────────────────────────


def test_split_over_lift_conjoins_masks(test_dataframe):
    m = leaf(lambda: MockModel(x_column="x"), "m")
    lifted = lift(
        m,
        values=["a", "b", "c"],
        name="category",
        train_filter=lambda v: pl.col("category") == v,
        test_filter=lambda v: pl.col("category") == v,
    )
    model = split(
        "tt",
        lifted,
        train_filter=(pl.col("x") % 3) != 0,
        test_filter=(pl.col("x") % 3) == 0,
    )
    model.fit(test_dataframe)

    # train = (category == v) & (x % 3 != 0)
    assert model.fitted["m[category=a]"].seen == [1, 2]
    assert model.fitted["m[category=b]"].seen == [4, 5]
    assert model.fitted["m[category=c]"].seen == [7, 8]

    pred = model.predict(test_dataframe)
    # test = (category == v) & (x % 3 == 0)
    assert pred["m[category=a]"].drop_nulls().to_list() == [[1, 2]]
    assert pred["m[category=b]"].drop_nulls().to_list() == [[4, 5]]
    assert pred["m[category=c]"].drop_nulls().to_list() == [[7, 8]]


def test_split_over_split_conjoins_masks(test_dataframe):
    m = leaf(lambda: MockModel(x_column="x"), "m")
    inner = split("inner", m, train_filter=pl.col("x") >= 3, test_filter=pl.lit(True))
    model = split("outer", inner, train_filter=pl.col("x") <= 6, test_filter=pl.col("x") >= 7)
    model.fit(test_dataframe)

    # train = (x <= 6) & (x >= 3)
    assert model.fitted["m"].seen == [3, 4, 5, 6]

    pred = model.predict(test_dataframe)
    # test = (x >= 7) & True
    assert pred["m"].drop_nulls().to_list() == [[3, 4, 5, 6]] * 3


def test_split_over_ensemble_applies_to_each_member(test_dataframe):
    a = leaf(lambda: MockModel(x_column="x"), "m-a")
    b = leaf(lambda: MockModel(x_column="x"), "m-b")
    model = split(
        "tt", ensemble("ens", a, b), train_filter=pl.col("x") <= 6, test_filter=pl.col("x") >= 7
    )
    model.fit(test_dataframe)

    assert model.fitted["m-a"].seen == [1, 2, 3, 4, 5, 6]
    assert model.fitted["m-b"].seen == [1, 2, 3, 4, 5, 6]

    pred = model.predict(test_dataframe)
    assert pred["m-a"].drop_nulls().to_list() == [[1, 2, 3, 4, 5, 6]] * 3
    assert pred["m-b"].drop_nulls().to_list() == [[1, 2, 3, 4, 5, 6]] * 3


def test_split_over_feed_no_leakage(test_dataframe):
    src = leaf(lambda: MockModel(x_column="x"), "src")
    cons = leaf(lambda: ConsumerModel(source_col="src"), "cons")
    model = split(
        "tt",
        feed("d", source=src, consumer=cons),
        train_filter=pl.col("x") < 5,
        test_filter=pl.col("x") >= 5,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # NaN-augmentation warning is expected here
        model.fit(test_dataframe)

    # Source's training data is exactly the Split's train rows; nothing leaked in.
    assert model.fitted["src"].seen == [1, 2, 3, 4]

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


def test_split_over_tune_propagates_train_mask(test_dataframe):
    # Regression test for the Tune train-mask leak: the enclosing Split restricts
    # training rows to x < 5, so the source must only see [1, 2, 3, 4]. Before the
    # fix the Tune re-rooted its source fit and trained on all nine rows.
    source = leaf(lambda: MockModel(x_column="x"), "source")
    consumer = leaf(lambda offset=0.0: OffsetModel(offset=offset), "consumer")
    tuned = tune("tn", consumer=consumer, source=source, logic=_mean_of_source)
    model = split("tt", tuned, train_filter=pl.col("x") < 5, test_filter=pl.col("x") >= 5)
    model.fit(test_dataframe)

    assert model.fitted["source"].seen == [1, 2, 3, 4]
    # logic reads the (unrestricted) source predictions -> mean([1,2,3,4]) = 2.5
    assert model.hyperparameters["offset"] == 2.5
    # consumer trains on x < 5: mean([1,2,3,4]) + 2.5 = 5.0
    assert model.fitted["consumer"].value == 5.0

    pred = model.predict(test_dataframe)
    assert pred["source"].drop_nulls().to_list() == [[1, 2, 3, 4]] * 5
    assert pred["consumer"].drop_nulls().to_list() == [5.0] * 5


# ── Lift as outer ──────────────────────────────────────────────────


def test_lift_over_split_conjoins_masks(test_dataframe):
    # Same leaf masks as test_split_over_lift - Lift and Split commute.
    m = leaf(lambda: MockModel(x_column="x"), "m")
    inner = split(
        "tt", m, train_filter=(pl.col("x") % 3) != 0, test_filter=(pl.col("x") % 3) == 0
    )
    model = lift(
        inner,
        values=["a", "b", "c"],
        name="category",
        train_filter=lambda v: pl.col("category") == v,
        test_filter=lambda v: pl.col("category") == v,
    )
    model.fit(test_dataframe)

    assert model.fitted["m[category=a]"].seen == [1, 2]
    assert model.fitted["m[category=b]"].seen == [4, 5]
    assert model.fitted["m[category=c]"].seen == [7, 8]

    pred = model.predict(test_dataframe)
    assert pred["m[category=a]"].drop_nulls().to_list() == [[1, 2]]
    assert pred["m[category=b]"].drop_nulls().to_list() == [[4, 5]]
    assert pred["m[category=c]"].drop_nulls().to_list() == [[7, 8]]


def test_lift_over_lift_composes_dimensions(test_dataframe):
    m = leaf(lambda: MockModel(x_column="x"), "m")
    inner = lift(
        m,
        values=["a", "b", "c"],
        name="category",
        train_filter=lambda v: pl.col("category") == v,
        test_filter=lambda v: pl.col("category") == v,
    )
    model = lift(
        inner,
        values=[0, 1],
        name="parity",
        train_filter=lambda v: (pl.col("x") % 2) == v,
        test_filter=lambda v: (pl.col("x") % 2) == v,
    )
    model.fit(test_dataframe)

    # train = (x % 2 == parity) & (category == v); outer dimension comes first in the label
    assert model.fitted["m[parity=0, category=a]"].seen == [2]
    assert model.fitted["m[parity=0, category=b]"].seen == [4, 6]
    assert model.fitted["m[parity=0, category=c]"].seen == [8]
    assert model.fitted["m[parity=1, category=a]"].seen == [1, 3]
    assert model.fitted["m[parity=1, category=b]"].seen == [5]
    assert model.fitted["m[parity=1, category=c]"].seen == [7, 9]

    pred = model.predict(test_dataframe)
    assert pred["m[parity=0, category=b]"].drop_nulls().to_list() == [[4, 6]] * 2
    assert pred["m[parity=1, category=c]"].drop_nulls().to_list() == [[7, 9]] * 2


def test_lift_over_ensemble_distributes(test_dataframe):
    a = leaf(lambda: MockModel(x_column="x"), "m-a")
    b = leaf(lambda: MockModel(x_column="x"), "m-b")
    model = lift(
        ensemble("ens", a, b),
        values=["a", "b", "c"],
        name="category",
        train_filter=lambda v: pl.col("category") == v,
        test_filter=lambda v: pl.col("category") == v,
    )
    model.fit(test_dataframe)

    assert model.fitted["m-a[category=a]"].seen == [1, 2, 3]
    assert model.fitted["m-b[category=c]"].seen == [7, 8, 9]

    pred = model.predict(test_dataframe)
    assert pred["m-a[category=b]"].drop_nulls().to_list() == [[4, 5, 6]] * 3
    assert pred["m-b[category=a]"].drop_nulls().to_list() == [[1, 2, 3]] * 3


def test_lift_over_feed_is_rejected(test_dataframe):
    # Lift-above-Feed is structurally rejected by the validator at construction time.
    src = leaf(lambda: MockModel(x_column="x"), "src")
    cons = leaf(lambda: ConsumerModel(source_col="src"), "cons")
    fed = feed("d", source=src, consumer=cons)
    with pytest.raises(ValueError, match="Lift as an ancestor"):
        lift(
            fed,
            values=["a", "b", "c"],
            name="category",
            train_filter=lambda v: pl.col("category") == v,
            test_filter=lambda v: pl.col("category") == v,
        )


def test_lift_over_tune_is_rejected(test_dataframe):
    # Lift-above-Tune is rejected by the validator for the same reason as Lift-above-Feed:
    # the Tune re-runs its source inside `logic` with an un-decorated label context, which
    # the Lift's label decoration breaks. Express per-value tuning by Lifting inside source.
    source = leaf(lambda: MockModel(x_column="x"), "source")
    consumer = leaf(lambda offset=0.0: OffsetModel(offset=offset), "consumer")
    tuned = tune("tn", consumer=consumer, source=source, logic=_mean_of_source)
    with pytest.raises(ValueError, match="Lift as an ancestor"):
        lift(
            tuned,
            values=["a", "b", "c"],
            name="category",
            train_filter=lambda v: pl.col("category") == v,
            test_filter=lambda v: pl.col("category") == v,
        )


# ── Ensemble as outer ──────────────────────────────────────────────


def test_ensemble_over_lift(test_dataframe):
    m = leaf(lambda: MockModel(x_column="x"), "m")
    lifted = lift(
        m,
        values=["a", "b", "c"],
        name="category",
        train_filter=lambda v: pl.col("category") == v,
        test_filter=lambda v: pl.col("category") == v,
    )
    solo = leaf(lambda: MockModel(x_column="x"), "solo")
    model = ensemble("ens", lifted, solo)
    model.fit(test_dataframe)

    assert model.fitted["m[category=a]"].seen == [1, 2, 3]
    assert model.fitted["m[category=c]"].seen == [7, 8, 9]
    assert model.fitted["solo"].seen == FULL

    pred = model.predict(test_dataframe)
    assert pred["m[category=b]"].drop_nulls().to_list() == [[4, 5, 6]] * 3
    assert pred["solo"].drop_nulls().to_list() == [FULL] * 9


def test_ensemble_over_split(test_dataframe):
    m = leaf(lambda: MockModel(x_column="x"), "m")
    bounded = split("tt", m, train_filter=pl.col("x") <= 4, test_filter=pl.col("x") >= 5)
    solo = leaf(lambda: MockModel(x_column="x"), "solo")
    model = ensemble("ens", bounded, solo)
    model.fit(test_dataframe)

    assert model.fitted["m"].seen == [1, 2, 3, 4]
    assert model.fitted["solo"].seen == FULL

    pred = model.predict(test_dataframe)
    assert pred["m"].drop_nulls().to_list() == [[1, 2, 3, 4]] * 5
    assert pred["solo"].drop_nulls().to_list() == [FULL] * 9


def test_ensemble_over_ensemble_nested_aggregation(test_dataframe):
    # Sum the inner ensemble, then sum that into the outer ensemble -> 3x the input.
    inner_a = leaf(lambda: MockModel(x_column="x"), "inner-a")
    inner_b = leaf(lambda: MockModel(x_column="x"), "inner-b")
    outer_c = leaf(lambda: MockModel(x_column="x"), "outer-c")
    inner = ensemble("inner", inner_a, inner_b, aggregate_with=pl.sum_horizontal)
    model = ensemble("outer", inner, outer_c, aggregate_with=pl.sum_horizontal)
    model.fit(test_dataframe)

    for label in ("inner-a", "inner-b", "outer-c"):
        assert model.fitted[label].seen == FULL

    predictions = model.predict(test_dataframe)
    for intermediate in ("inner", "inner-a", "inner-b", "outer-c"):
        assert intermediate not in predictions.columns
    distinct = predictions.select("outer").unique()
    expected = pl.DataFrame(
        {"outer": [[3, 6, 9, 12, 15, 18, 21, 24, 27]]},
        schema={"outer": pl.List(pl.Int64)},
    )
    assert_frame_equal(distinct, expected)


def test_ensemble_over_feed(test_dataframe):
    src = leaf(lambda: MockModel(x_column="x"), "src")
    cons = leaf(lambda: ConsumerModel(source_col="src"), "cons")
    fed = feed("d", source=src, consumer=cons)
    solo = leaf(lambda: MockModel(x_column="x"), "solo")
    model = ensemble("ens", fed, solo)
    model.fit(test_dataframe)

    assert model.fitted["src"].seen == FULL
    assert model.fitted["cons"].seen == [FULL]
    assert model.fitted["solo"].seen == FULL

    pred = model.predict(test_dataframe)
    assert pred["src"].drop_nulls().to_list() == [FULL] * 9
    assert pred["cons"].drop_nulls().to_list() == [FULL] * 9
    assert pred["solo"].drop_nulls().to_list() == [FULL] * 9


def test_ensemble_over_tune(test_dataframe):
    source = leaf(lambda: MockModel(x_column="x"), "source")
    consumer = leaf(lambda offset=0.0: OffsetModel(offset=offset), "consumer")
    tuned = tune("tn", consumer=consumer, source=source, logic=_mean_of_source)
    solo = leaf(lambda: MockModel(x_column="x"), "solo")
    model = ensemble("ens", tuned, solo)
    model.fit(test_dataframe)

    assert model.fitted["source"].seen == FULL
    assert model.hyperparameters["offset"] == 5.0  # mean([1..9])
    assert model.fitted["consumer"].value == 10.0  # mean([1..9]) + 5.0
    assert model.fitted["solo"].seen == FULL

    pred = model.predict(test_dataframe)
    assert pred["consumer"].drop_nulls().to_list() == [10.0] * 9
    assert pred["solo"].drop_nulls().to_list() == [FULL] * 9


# ── Feed as outer (inner node in the source) ───────────────────────


def test_feed_with_lift_in_source_and_consumer(test_dataframe):
    # CV cross-fitting via Lift inside source: each teacher fold trains on `fold != v`
    # and predicts on `fold == v`; the coalesce produces out-of-fold predictions
    # covering all rows, which the student then consumes.
    teacher = leaf(lambda: MockModel(x_column="x"), "teacher")
    student = leaf(lambda: ConsumerModel(source_col="cv_teacher"), "student")

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

    data_excluding_fold_0 = [4, 5, 6, 7, 8, 9]
    data_excluding_fold_1 = [1, 2, 3, 7, 8, 9]
    data_excluding_fold_2 = [1, 2, 3, 4, 5, 6]

    assert model.fitted["teacher[cv_teacher=0]"].seen == data_excluding_fold_0
    assert model.fitted["teacher[cv_teacher=1]"].seen == data_excluding_fold_1
    assert model.fitted["teacher[cv_teacher=2]"].seen == data_excluding_fold_2

    # On fold 0 the student trains on the OOF teacher predictions from folds 1 and 2.
    assert model.fitted["student[cv_student=0]"].seen == [
        data_excluding_fold_1,
        data_excluding_fold_2,
    ]

    predictions = model.predict(test_dataframe)
    distinct = predictions.select(
        "fold",
        "cv_teacher",
        "student[cv_student=0]",
        "student[cv_student=1]",
        "student[cv_student=2]",
    ).unique(subset=["fold"], maintain_order=True)
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


def test_feed_with_split_in_source(test_dataframe):
    src = leaf(lambda: MockModel(x_column="x"), "src")
    bounded = split("src_tt", src, train_filter=pl.col("x") <= 4, test_filter=pl.lit(True))
    cons = leaf(lambda: ConsumerModel(source_col="src"), "cons")
    model = feed("d", source=bounded, consumer=cons)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # source test covers all rows -> no NaN/leak warning
        model.fit(test_dataframe)

    assert model.fitted["src"].seen == [1, 2, 3, 4]
    assert model.fitted["cons"].seen == [[1, 2, 3, 4]]

    pred = model.predict(test_dataframe)
    assert pred["src"].drop_nulls().to_list() == [[1, 2, 3, 4]] * 9
    assert pred["cons"].drop_nulls().to_list() == [[1, 2, 3, 4]] * 9


def test_feed_with_ensemble_in_source(test_dataframe):
    s1 = leaf(lambda: MockModel(x_column="x"), "s1")
    s2 = leaf(lambda: MockModel(x_column="x"), "s2")
    src = ensemble("src_ens", s1, s2)
    cons = leaf(lambda: ConsumerModel(source_col="s1"), "cons")
    model = feed("d", source=src, consumer=cons)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(test_dataframe)

    assert model.fitted["s1"].seen == FULL
    assert model.fitted["s2"].seen == FULL
    assert model.fitted["cons"].seen == [FULL]

    pred = model.predict(test_dataframe)
    assert pred["cons"].drop_nulls().to_list() == [FULL] * 9


def test_feed_with_feed_in_source(test_dataframe):
    a = leaf(lambda: MockModel(x_column="x"), "a")
    b = leaf(lambda: ConsumerModel(source_col="a"), "b")
    inner = feed("inner", source=a, consumer=b)
    c = leaf(lambda: ConsumerModel(source_col="b"), "c")
    model = feed("outer", source=inner, consumer=c)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(test_dataframe)

    assert model.fitted["a"].seen == FULL
    assert model.fitted["b"].seen == [FULL]
    assert model.fitted["c"].seen == [FULL]

    pred = model.predict(test_dataframe)
    assert pred["c"].drop_nulls().to_list() == [FULL] * 9


def test_feed_with_tune_in_source(test_dataframe):
    # "Tune the teacher": the teacher is tuned, then feeds a student.
    t_src = leaf(lambda: MockModel(x_column="x"), "t_src")
    teacher = leaf(lambda offset=0.0: OffsetModel(offset=offset), "teacher")
    tuned = tune("tn", consumer=teacher, source=t_src, logic=_mean_of_t_src)
    student = leaf(lambda: ConsumerModel(source_col="teacher"), "student")
    model = feed("d", source=tuned, consumer=student)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(test_dataframe)

    assert model.fitted["t_src"].seen == FULL
    assert model.hyperparameters["offset"] == 5.0
    assert model.fitted["teacher"].value == 10.0
    assert model.fitted["student"].seen == [10.0]

    pred = model.predict(test_dataframe)
    assert pred["student"].drop_nulls().to_list() == [10.0] * 9


# ── Tune as outer (inner node in the source) ───────────────────────
#
# These use a constant `logic` so the assertions isolate mask/data propagation into
# the source and hyperparameter propagation into the consumer, not the logic arithmetic.


def test_tune_with_lift_in_source(test_dataframe):
    s = leaf(lambda: MockModel(x_column="x"), "s")
    lifted = lift(
        s,
        values=["a", "b", "c"],
        name="category",
        train_filter=lambda v: pl.col("category") == v,
        test_filter=lambda v: pl.col("category") == v,
    )
    c = leaf(lambda offset=0.0: OffsetModel(offset=offset), "c")
    model = tune("tn", consumer=c, source=lifted, logic=lambda m, df: {"offset": 3.0})
    model.fit(test_dataframe)

    assert model.fitted["s[category=a]"].seen == [1, 2, 3]
    assert model.fitted["s[category=c]"].seen == [7, 8, 9]
    assert model.fitted["c"].value == 8.0  # mean([1..9]) + 3.0

    pred = model.predict(test_dataframe)
    assert pred["s[category=b]"].drop_nulls().to_list() == [[4, 5, 6]] * 3
    assert pred["c"].drop_nulls().to_list() == [8.0] * 9


def test_tune_with_split_in_source(test_dataframe):
    s = leaf(lambda: MockModel(x_column="x"), "s")
    bounded = split("src_tt", s, train_filter=pl.col("x") <= 4, test_filter=pl.lit(True))
    c = leaf(lambda offset=0.0: OffsetModel(offset=offset), "c")
    model = tune("tn", consumer=c, source=bounded, logic=lambda m, df: {"offset": 3.0})
    model.fit(test_dataframe)

    assert model.fitted["s"].seen == [1, 2, 3, 4]
    assert model.fitted["c"].value == 8.0

    pred = model.predict(test_dataframe)
    assert pred["s"].drop_nulls().to_list() == [[1, 2, 3, 4]] * 9
    assert pred["c"].drop_nulls().to_list() == [8.0] * 9


def test_tune_with_ensemble_in_source(test_dataframe):
    s1 = leaf(lambda: MockModel(x_column="x"), "s1")
    s2 = leaf(lambda: MockModel(x_column="x"), "s2")
    src = ensemble("src_ens", s1, s2)
    c = leaf(lambda offset=0.0: OffsetModel(offset=offset), "c")
    model = tune("tn", consumer=c, source=src, logic=lambda m, df: {"offset": 3.0})
    model.fit(test_dataframe)

    assert model.fitted["s1"].seen == FULL
    assert model.fitted["s2"].seen == FULL
    assert model.fitted["c"].value == 8.0

    pred = model.predict(test_dataframe)
    assert pred["c"].drop_nulls().to_list() == [8.0] * 9


def test_tune_with_feed_in_source(test_dataframe):
    fs = leaf(lambda: MockModel(x_column="x"), "fs")
    fc = leaf(lambda: ConsumerModel(source_col="fs"), "fc")
    fed = feed("d", source=fs, consumer=fc)
    c = leaf(lambda offset=0.0: OffsetModel(offset=offset), "c")
    model = tune("tn", consumer=c, source=fed, logic=lambda m, df: {"offset": 3.0})
    model.fit(test_dataframe)

    assert model.fitted["fs"].seen == FULL
    assert model.fitted["fc"].seen == [FULL]
    assert model.fitted["c"].value == 8.0

    pred = model.predict(test_dataframe)
    assert pred["c"].drop_nulls().to_list() == [8.0] * 9


def test_tune_with_tune_in_source(test_dataframe):
    inner_src = leaf(lambda: MockModel(x_column="x"), "is")
    ic = leaf(lambda offset=0.0: OffsetModel(offset=offset), "ic")
    inner = tune("inner", consumer=ic, source=inner_src, logic=lambda m, df: {"offset": 2.0})
    oc = leaf(lambda offset=0.0: OffsetModel(offset=offset), "oc")
    model = tune("outer", consumer=oc, source=inner, logic=lambda m, df: {"offset": 3.0})
    model.fit(test_dataframe)

    assert model.fitted["is"].seen == FULL
    assert model.fitted["ic"].value == 7.0  # mean([1..9]) + 2.0 (inner offset)
    assert model.fitted["oc"].value == 8.0  # mean([1..9]) + 3.0 (outer offset)

    pred = model.predict(test_dataframe)
    assert pred["oc"].drop_nulls().to_list() == [8.0] * 9
