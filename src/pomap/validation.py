from typing import Iterator

from .nodes import PomapNode, Leaf, Lift, Feed


_RESERVED_COLUMN_NAMES = {"__pomap_row_index"}


def _validate(root: PomapNode) -> None:
    _check_unique_node_names(root)
    _check_no_lift_above_feed(root, under_lift=False)
    _check_no_reserved_leaf_labels(root)


def _check_unique_node_names(root: PomapNode) -> None:
    """Every node's identifying name must be globally unique within the tree."""
    seen = set()
    duplicates = set()
    for name in _collect_node_names(root):
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        raise ValueError(
            f"Duplicate node names: {sorted(duplicates)}. Every Leaf label and "
            "every non-leaf node name must be globally unique within the tree."
        )


def _collect_node_names(node: PomapNode) -> Iterator[str]:
    yield node.label if isinstance(node, Leaf) else node.name
    for child in node.children:
        yield from _collect_node_names(child)


def _check_no_lift_above_feed(node: PomapNode, under_lift: bool) -> None:
    """Reject Lift as an ancestor of Feed.

    The source leaf's prediction column would be named with the Lift's decoration
    (e.g. `teacher[fold=v]`) while the consumer's leaf code references the bare
    label literal — a column-name mismatch. Until output-channel stability lands,
    this case is blocked. Split-above-Feed is permitted: it raises NaN-augmentation
    warnings at fit time but is structurally sound.
    """
    if isinstance(node, Feed) and under_lift:
        raise ValueError(
            f"Feed {node.name!r} has a Lift as an ancestor. This is currently "
            "unsupported because the source's decorated leaf labels would not "
            "match the bare-name references in the consumer. Workaround: express "
            "the row split as a CV `lift` inside source with "
            "`aggregate_with=pl.coalesce`, rather than wrapping the Feed."
        )
    next_under_lift = under_lift or isinstance(node, Lift)
    for child in node.children:
        _check_no_lift_above_feed(child, next_under_lift)


def _check_no_reserved_leaf_labels(node: PomapNode) -> None:
    if isinstance(node, Leaf) and node.label in _RESERVED_COLUMN_NAMES:
        raise ValueError(
            f"Leaf label {node.label!r} collides with a reserved column name."
        )
    for child in node.children:
        _check_no_reserved_leaf_labels(child)
