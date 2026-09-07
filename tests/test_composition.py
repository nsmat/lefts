"""
Composition tests: one node wrapping another, exercising fit AND predict together.

Every other test_*.py file tests a single node (or a single interpreter function) in
isolation. Those pass even when the *seam* between two nodes is broken - the Tune
train-mask leak was invisible to `test_masks_tune_passthrough` (masks were collected
correctly) and to `test_fit_tune_threads_hyperparameters` (a bare Tune has no outer
mask to propagate). This file covers that seam.

It has two layers:

1. An **invariant layer** - three properties that must hold for *any* composed tree,
   run over a spread of tree shapes. These are absolute and value-free, so a single
   check catches a whole class of bug across every shape. Built from MockModel-style
   probes so `.trained_on` is a direct read-out of the rows each leaf actually saw.
2. A handful of **anchor** tests that pin the exact values the invariants can't
   express: mask arithmetic, label strings, aggregated outputs, data-flow values,
   tuned hyperparameters, and the construction guards.
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


# ══════════════════════════════════════════════════════════════════════
# Invariant layer
#
# Instead of hand-computing an expected value for every (outer, inner) pair, we
# assert three properties that must hold for *any* composed tree, then run them
# over a spread of tree shapes. These are absolute and value-free: unlike the
# commutativity of two orderings (a relative check that can't catch a bug shared
# by both), a single invariant catches a whole class of bug across every shape -
# the Tune train-mask leak included. The exact-value anchor tests further down
# pin the things invariants can't express (mask arithmetic, hyperparameter values,
# label strings).
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Probe:
    """Leaf probe: records the x-values it trained on and echoes them back on predict."""

    trained_on: list = None

    def fit(self, training_set):
        self.trained_on = training_set["x"].to_list()

    def predict(self, df):
        return [self.trained_on] * len(df)


@dataclass
class ConsumerProbe:
    """Feed consumer probe: records its own training rows for the leakage invariant."""

    source_col: str
    trained_on: list = None

    def fit(self, training_set):
        self.trained_on = training_set["x"].to_list()

    def predict(self, df):
        return df[self.source_col].to_list()


@dataclass
class OffsetProbe:
    """Tune consumer probe: records its own training rows for the leakage invariant."""

    offset: float = 0.0
    trained_on: list = None

    def fit(self, training_set):
        self.trained_on = training_set["x"].to_list()

    def predict(self, df):
        return [self.offset] * len(df)


def assert_no_leakage(model, df):
    """
    Every leaf trains on exactly the rows its collected train mask permits.

    This cross-checks two independently-computed things that must agree: what `_fit`
    actually trained each leaf on (`probe.trained_on`) and what `_collect_masks` says
    it should have (`mark_train_validation_test_rows`). The Tune train-mask leak was
    precisely a divergence between these two - the seam a single-path test can't see.
    """
    marked = model.mark_train_validation_test_rows(df)
    for label, fitted in model.fitted.items():
        permitted = set(marked.filter(pl.col(f"{label}__train"))["x"].to_list())
        assert set(fitted.trained_on) == permitted, (
            f"{label} trained on {sorted(fitted.trained_on)}, mask permits {sorted(permitted)}"
        )


def assert_predict_replicates_train(model, df):
    """
    Each leaf Probe echoes its training data back, and only on its test rows.

    This is the probe contract that makes the other invariants observable - if it
    holds, a prediction column *is* a faithful read-out of what that leaf trained on.
    Only applies to plain Probe leaves whose column survives into the output (an
    aggregation collapses them, a Feed/Tune consumer has a different contract).
    """
    marked = model.mark_train_validation_test_rows(df)
    pred = model.predict(df)
    for label, fitted in model.fitted.items():
        if not isinstance(fitted, Probe) or label not in pred.columns:
            continue
        emitted = pred.filter(pl.col(label).is_not_null())
        test_rows = set(marked.filter(pl.col(f"{label}__test"))["x"].to_list())
        assert set(emitted["x"].to_list()) == test_rows
        assert all(value == fitted.trained_on for value in emitted[label].to_list())


def assert_labels_match_columns(model, df):
    """The public label set is exactly the set of columns predict adds to the frame."""
    produced = set(model.predict(df).columns) - set(df.columns)
    assert produced == set(model.collect_labels())


def _cat_lift(model):
    """A three-way Lift over `category` where each value trains and tests on its own rows."""
    return lift(
        model,
        values=["a", "b", "c"],
        name="category",
        train_filter=lambda v: pl.col("category") == v,
        test_filter=lambda v: pl.col("category") == v,
    )


def _shapes():
    """A spread of composed trees (built from probes) to run every invariant over."""
    return {
        "leaf": lambda: leaf(lambda: Probe(), "m"),
        "split_over_lift": lambda: split(
            "tt", _cat_lift(leaf(lambda: Probe(), "m")),
            train_filter=(pl.col("x") % 3) != 0, test_filter=(pl.col("x") % 3) == 0,
        ),
        "lift_over_split": lambda: _cat_lift(
            split("tt", leaf(lambda: Probe(), "m"),
                  train_filter=(pl.col("x") % 3) != 0, test_filter=(pl.col("x") % 3) == 0)
        ),
        "lift_over_lift": lambda: lift(
            _cat_lift(leaf(lambda: Probe(), "m")),
            values=[0, 1], name="parity",
            train_filter=lambda v: (pl.col("x") % 2) == v,
            test_filter=lambda v: (pl.col("x") % 2) == v,
        ),
        "split_over_split": lambda: split(
            "outer",
            split("inner", leaf(lambda: Probe(), "m"),
                  train_filter=pl.col("x") >= 3, test_filter=pl.lit(True)),
            train_filter=pl.col("x") <= 6, test_filter=pl.col("x") >= 7,
        ),
        "split_over_ensemble": lambda: split(
            "tt", ensemble("ens", leaf(lambda: Probe(), "m-a"), leaf(lambda: Probe(), "m-b")),
            train_filter=pl.col("x") <= 6, test_filter=pl.col("x") >= 7,
        ),
        "lift_over_ensemble": lambda: _cat_lift(
            ensemble("ens", leaf(lambda: Probe(), "m-a"), leaf(lambda: Probe(), "m-b"))
        ),
        "ensemble_over_split": lambda: ensemble(
            "ens",
            split("tt", leaf(lambda: Probe(), "m"),
                  train_filter=pl.col("x") <= 4, test_filter=pl.col("x") >= 5),
            leaf(lambda: Probe(), "solo"),
        ),
        "nested_ensemble_aggregate": lambda: ensemble(
            "outer",
            ensemble("inner", leaf(lambda: Probe(), "inner-a"), leaf(lambda: Probe(), "inner-b"),
                     aggregate_with=pl.sum_horizontal),
            leaf(lambda: Probe(), "outer-c"),
            aggregate_with=pl.sum_horizontal,
        ),
        "lift_coalesce_oof": lambda: lift(
            leaf(lambda: Probe(), "teacher"),
            values=[0, 1, 2], name="cv",
            train_filter=lambda v: pl.col("fold") != v,
            test_filter=lambda v: pl.col("fold") == v,
            aggregate_with=pl.coalesce,
        ),
        "split_over_feed": lambda: split(
            "tt",
            feed("d", source=leaf(lambda: Probe(), "src"),
                 consumer=leaf(lambda: ConsumerProbe(source_col="src"), "cons")),
            train_filter=pl.col("x") < 5, test_filter=pl.col("x") >= 5,
        ),
        "split_over_tune": lambda: split(
            "tt",
            tune("tn", source=leaf(lambda: Probe(), "source"),
                 consumer=leaf(lambda offset=0.0: OffsetProbe(offset=offset), "consumer"),
                 logic=lambda m, df: {"offset": 3.0}),
            train_filter=pl.col("x") < 5, test_filter=pl.col("x") >= 5,
        ),
        "ensemble_over_tune": lambda: ensemble(
            "ens",
            tune("tn", source=leaf(lambda: Probe(), "source"),
                 consumer=leaf(lambda offset=0.0: OffsetProbe(offset=offset), "consumer"),
                 logic=lambda m, df: {"offset": 3.0}),
            leaf(lambda: Probe(), "solo"),
        ),
        "feed_source_split": lambda: feed(
            "d",
            source=split("src_tt", leaf(lambda: Probe(), "src"),
                         train_filter=pl.col("x") <= 4, test_filter=pl.lit(True)),
            consumer=leaf(lambda: ConsumerProbe(source_col="src"), "cons"),
        ),
        "feed_source_lift_crossfit": lambda: feed(
            "d",
            source=lift(leaf(lambda: Probe(), "teacher"),
                        values=[0, 1, 2], name="cv_teacher",
                        train_filter=lambda v: pl.col("fold") != v,
                        test_filter=lambda v: pl.col("fold") == v,
                        aggregate_with=pl.coalesce),
            consumer=leaf(lambda: ConsumerProbe(source_col="cv_teacher"), "student"),
        ),
        "tune_source_lift": lambda: tune(
            "tn",
            source=_cat_lift(leaf(lambda: Probe(), "s")),
            consumer=leaf(lambda offset=0.0: OffsetProbe(offset=offset), "c"),
            logic=lambda m, df: {"offset": 3.0},
        ),
        "tune_source_feed": lambda: tune(
            "tn",
            source=feed("d", source=leaf(lambda: Probe(), "fs"),
                        consumer=leaf(lambda: ConsumerProbe(source_col="fs"), "fc")),
            consumer=leaf(lambda offset=0.0: OffsetProbe(offset=offset), "c"),
            logic=lambda m, df: {"offset": 3.0},
        ),
        "feed_source_tune": lambda: feed(
            "d",
            source=tune("tn", source=leaf(lambda: Probe(), "t_src"),
                        consumer=leaf(lambda offset=0.0: OffsetProbe(offset=offset), "teacher"),
                        logic=lambda m, df: {"offset": 3.0}),
            consumer=leaf(lambda: ConsumerProbe(source_col="teacher"), "student"),
        ),
    }


_SHAPES = _shapes()


@pytest.mark.parametrize("shape", _SHAPES, ids=list(_SHAPES))
def test_invariant_no_leakage(shape, test_dataframe):
    model = _SHAPES[shape]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # NaN-augmentation warnings are orthogonal here
        model.fit(test_dataframe)
    assert_no_leakage(model, test_dataframe)


@pytest.mark.parametrize("shape", _SHAPES, ids=list(_SHAPES))
def test_invariant_predict_replicates_train(shape, test_dataframe):
    model = _SHAPES[shape]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(test_dataframe)
    assert_predict_replicates_train(model, test_dataframe)


@pytest.mark.parametrize("shape", _SHAPES, ids=list(_SHAPES))
def test_invariant_labels_match_columns(shape, test_dataframe):
    model = _SHAPES[shape]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(test_dataframe)
    assert_labels_match_columns(model, test_dataframe)


# ══════════════════════════════════════════════════════════════════════
# Anchors
#
# Absolute, hand-computed values the invariants above cannot express. Each pins one
# thing: without at least one anchor, the invariants only prove the tree is *self
# consistent* (e.g. training data matches the collected mask) - not that the mask,
# label, or aggregated value is itself correct.
# ══════════════════════════════════════════════════════════════════════


def test_anchor_mask_conjunction(test_dataframe):
    # Pins the exact mask arithmetic: Split's filter conjoined with each Lift value's.
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


def test_anchor_nested_lift_labels(test_dataframe):
    # Pins the exact label strings a double-Lift produces (outer dimension first).
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

    assert model.fitted["m[parity=0, category=a]"].seen == [2]
    assert model.fitted["m[parity=0, category=b]"].seen == [4, 6]
    assert model.fitted["m[parity=0, category=c]"].seen == [8]
    assert model.fitted["m[parity=1, category=a]"].seen == [1, 3]
    assert model.fitted["m[parity=1, category=b]"].seen == [5]
    assert model.fitted["m[parity=1, category=c]"].seen == [7, 9]

    pred = model.predict(test_dataframe)
    assert pred["m[parity=0, category=b]"].drop_nulls().to_list() == [[4, 6]] * 2
    assert pred["m[parity=1, category=c]"].drop_nulls().to_list() == [[7, 9]] * 2


def test_anchor_nested_aggregation_values(test_dataframe):
    # Pins the aggregated output: sum the inner ensemble, then the outer -> 3x the input.
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


def test_anchor_feed_crossfit_dataflow(test_dataframe):
    # Pins the Feed data-flow: the student actually receives the teacher's OOF predictions.
    # CV cross-fitting via Lift inside source - each teacher fold trains on `fold != v` and
    # predicts on `fold == v`; the coalesce produces OOF predictions covering all rows.
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


def test_anchor_tune_propagates_train_mask_and_hyperparameter(test_dataframe):
    # Named regression for the Tune train-mask leak, and pins the tuned hyperparameter
    # arithmetic: the enclosing Split restricts training to x < 5, so the source sees
    # only [1, 2, 3, 4]; before the fix the Tune re-rooted and trained on all nine rows.
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


def test_anchor_tune_the_teacher(test_dataframe):
    # Pins the canonical "tune the teacher" integration: a tuned teacher feeds a student.
    t_src = leaf(lambda: MockModel(x_column="x"), "t_src")
    teacher = leaf(lambda offset=0.0: OffsetModel(offset=offset), "teacher")
    tuned = tune("tn", consumer=teacher, source=t_src, logic=_mean_of_t_src)
    student = leaf(lambda: ConsumerModel(source_col="teacher"), "student")
    model = feed("d", source=tuned, consumer=student)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(test_dataframe)

    assert model.fitted["t_src"].seen == FULL
    assert model.hyperparameters["offset"] == 5.0  # mean([1..9])
    assert model.fitted["teacher"].value == 10.0  # mean([1..9]) + 5.0
    assert model.fitted["student"].seen == [10.0]

    pred = model.predict(test_dataframe)
    assert pred["student"].drop_nulls().to_list() == [10.0] * 9


def test_anchor_lift_over_feed_is_rejected(test_dataframe):
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


def test_anchor_lift_over_tune_is_rejected(test_dataframe):
    # Lift-above-Tune is rejected for the same reason as Lift-above-Feed: the Tune re-runs
    # its source inside `logic` with an un-decorated label context, which the Lift breaks.
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
