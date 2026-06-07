import pytest
from dataclasses import dataclass
import polars as pl

from pomap.nodes import Lift, Leaf, Split, Ensemble, Feed
from pomap.validation import _validate


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
    # Outer Split → Ensemble → (Feed with Split inside its consumer, lift over leaf)
    inner_feed = Feed(name="d", source=_leaf("src"), consumer=_leaf("cons"))
    lifted = _trivial_lift(_leaf("other"), name="cat", values=["a", "b"])
    ens = Ensemble(name="e", models=[inner_feed, lifted])
    root = Split(
        name="outer_tt",
        child=ens,
        train_filter=pl.lit(True),
        test_filter=pl.lit(True),
    )
    _validate(root)  # no raise


def test_duplicate_sibling_leaves_raises():
    ens = Ensemble(name="e", models=[_leaf("dup"), _leaf("dup")])
    with pytest.raises(ValueError, match="Duplicate node names"):
        _validate(ens)


def test_parent_child_duplicates_raise():
    a = _leaf("a")
    ens = Ensemble(name="a", models=[a])
    with pytest.raises(ValueError, match="Duplicate node names"):
        _validate(ens)


def test_grandparent_child_duplicates_raise():
    inner = Ensemble(name="e", models=[_leaf("a")])
    grand = _trivial_lift(inner, name="a")
    with pytest.raises(ValueError, match="Duplicate node names"):
        _validate(grand)


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


def test_split_above_feed_passes():
    inner_feed = Feed(name="d", source=_leaf("src"), consumer=_leaf("cons"))
    node = Split(
        name="tt",
        child=inner_feed,
        train_filter=pl.lit(True),
        test_filter=pl.lit(True),
    )
    _validate(node)  # no raise
