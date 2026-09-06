from typing import Iterator

from .nodes import LeftsNode, Leaf, Lift, Feed, Tune


_RESERVED_COLUMN_NAMES = {"__lefts_row_index"}

def _validate(root: LeftsNode) -> None:
    _check_unique_node_names(root)
    _check_no_lift_above_incompatible_nodes(root, under_lift=False)
    _check_no_reserved_leaf_labels(root)


def _check_unique_node_names(root: LeftsNode) -> None:
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


def _collect_node_names(node: LeftsNode) -> Iterator[str]:
    yield node.label if isinstance(node, Leaf) else node.name
    for child in node.children:
        yield from _collect_node_names(child)


def _check_no_lift_above_incompatible_nodes(node: LeftsNode, under_lift: bool) -> None:
    """
    Reject Lift as an ancestor of Feed or Tune.

    We don't allow this because the behaviour is confusing. A lift
    may spread the output of .predict across multiple output columns,
    but feed and tune cannot reference these appropriately.

    The user will usually not detect this themselves until fit time,
    so we guard it during workflow construction
    """
    # TODO Ideally we would allow this behaviour if we could make it safe
    unliftable_nodes = (Feed, Tune)

    if isinstance(node, unliftable_nodes) and under_lift:
        kind = type(node).__name__
        raise ValueError(
            f"{kind} {node.name!r} has a Lift as an ancestor. This is currently "
            "unsupported. Re-express by Lifting inside the source subtree instead."
        )
    next_under_lift = under_lift or isinstance(node, Lift)
    for child in node.children:
        _check_no_lift_above_incompatible_nodes(child, next_under_lift)


def _check_no_reserved_leaf_labels(node: LeftsNode) -> None:
    if isinstance(node, Leaf) and node.label in _RESERVED_COLUMN_NAMES:
        raise ValueError(
            f"Leaf label {node.label!r} collides with a reserved column name."
        )
    for child in node.children:
        _check_no_reserved_leaf_labels(child)
