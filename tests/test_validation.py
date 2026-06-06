import pytest
from dataclasses import dataclass
import polars as pl

from pomap.nodes import Lift, Leaf, Split, Ensemble, Feed, LearnsFrom
from pomap.validation import _validate
from pomap.interface import leaf, lift, feed, ensemble


@dataclass
class MockModel:
    value: float = None

    def fit(self, training_set: pl.DataFrame):
        self.value = 0.0

    def predict(self, df: pl.DataFrame):
        return [self.value] * len(df)


def _leaf(label: str) -> Leaf:
    return Leaf(label=label, factory=lambda: MockModel())


def _trivial_lift(child, name="fold", values=("v",)) -> Lift:
    return Lift(
        name=name,
        child=child,
        values=list(values),
        train_filter=lambda v: pl.lit(True),
        test_filter=lambda v: pl.lit(True),
    )


def test_valid_tree_passes():
    # Split → Ensemble → (Feed with Split inside its consumer, lift over leaf)
    consumer_split = Split(
        name="inner_tt",
        child=_leaf("cons"),
        train_filter=pl.lit(True),
        test_filter=pl.lit(True),
    )
    inner_feed = Feed(name="d", source=_leaf("src"), consumer=consumer_split)
    lifted = _trivial_lift(_leaf("other"), name="cat", values=["a", "b"])
    ens = Ensemble(name="e", models=[inner_feed, lifted])
    root = Split(
        name="outer_tt",
        child=ens,
        train_filter=pl.lit(True),
        test_filter=pl.lit(True),
    )
    # Outer Split has the Feed as a descendant via the Ensemble — should raise.
    with pytest.raises(ValueError, match="has a Split as an ancestor"):
        _validate(root)

    # Removing the outer Split makes it valid.
    _validate(ens)


def test_duplicate_sibling_leaves_raises():
    ens = Ensemble(name="e", models=[_leaf("dup"), _leaf("dup")])
    with pytest.raises(ValueError, match="Duplicate decorated leaf labels"):
        _validate(ens)


def test_duplicate_lift_names_in_ancestor_chain_raises():
    inner = _trivial_lift(_leaf("m"), name="fold")
    outer = _trivial_lift(inner, name="fold")
    with pytest.raises(ValueError, match="reused along an ancestor chain"):
        _validate(outer)


def test_sibling_lifts_can_share_name():
    """Two Lifts at the same depth (not in an ancestor chain) may reuse a name."""
    a = _trivial_lift(_leaf("a"), name="fold")
    b = _trivial_lift(_leaf("b"), name="fold")
    ens = Ensemble(name="e", models=[a, b])
    _validate(ens)  # no raise


def test_lift_above_feed_raises():
    inner_feed = Feed(name="d", source=_leaf("src"), consumer=_leaf("cons"))
    root = _trivial_lift(inner_feed, name="tt")
    with pytest.raises(ValueError, match="has a Lift as an ancestor"):
        _validate(root)


def test_lift_above_feed_via_ensemble_raises():
    """Lift doesn't have to be a direct parent — any ancestor counts."""
    inner_feed = Feed(name="d", source=_leaf("src"), consumer=_leaf("cons"))
    ens = Ensemble(name="e", models=[inner_feed, _leaf("other")])
    root = _trivial_lift(ens, name="tt")
    with pytest.raises(ValueError, match="has a Lift as an ancestor"):
        _validate(root)


def test_lift_inside_feed_source_passes():
    """Lift downstream of Feed is fine — only upstream is dangerous."""
    lifted_source = _trivial_lift(_leaf("src"), name="fold", values=["a", "b"])
    node = Feed(name="d", source=lifted_source, consumer=_leaf("cons"))
    _validate(node)  # no raise


def test_split_above_feed_raises():
    inner_feed = Feed(name="d", source=_leaf("src"), consumer=_leaf("cons"))
    node = Split(
        name="tt",
        child=inner_feed,
        train_filter=pl.lit(True),
        test_filter=pl.lit(True),
    )
    with pytest.raises(ValueError, match="has a Split as an ancestor"):
        _validate(node)


def test_lift_above_split_above_feed_reports_lift():
    """When both Lift and Split are ancestors, the error reports the worse one."""
    inner_feed = Feed(name="d", source=_leaf("src"), consumer=_leaf("cons"))
    inner_split = Split(
        name="tt",
        child=inner_feed,
        train_filter=pl.lit(True),
        test_filter=pl.lit(True),
    )
    root = _trivial_lift(inner_split, name="fold")
    with pytest.raises(ValueError, match="has a Lift as an ancestor"):
        _validate(root)


def test_split_inside_feed_consumer_passes():
    """Split downstream of Feed is fine — it's only upstream that leaks."""
    node = Feed(
        name="d",
        source=_leaf("src"),
        consumer=Split(
            name="tt",
            child=_leaf("cons"),
            train_filter=pl.lit(True),
            test_filter=pl.lit(True),
        ),
    )
    _validate(node)  # no raise


def test_reserved_leaf_label_raises():
    bad = Leaf(label="__pomap_row_index", factory=lambda: MockModel())
    with pytest.raises(ValueError, match="reserved column name"):
        _validate(bad)


def test_non_callable_learn_logic_raises():
    bad = LearnsFrom(
        name="lf",
        learner=_leaf("l"),
        learns_from=_leaf("s"),
        learn_logic="not a function",
    )
    with pytest.raises(ValueError, match="non-callable learn_logic"):
        _validate(bad)


def test_interface_helpers_invoke_validation():
    """Sanity check: validation fires at Model construction via interface helpers."""
    src = leaf(lambda: MockModel(), "src")
    cons = leaf(lambda: MockModel(), "cons")
    distillation = feed("d", source=src, consumer=cons)

    with pytest.raises(ValueError, match="has a Lift as an ancestor"):
        lift(
            distillation,
            values=["v"],
            name="tt",
            train_filter=lambda v: pl.lit(True),
            test_filter=lambda v: pl.lit(True),
        )


def test_interface_helper_catches_duplicate_labels():
    a = leaf(lambda: MockModel(), "dup")
    b = leaf(lambda: MockModel(), "dup")
    with pytest.raises(ValueError, match="Duplicate decorated leaf labels"):
        ensemble("e", a, b)