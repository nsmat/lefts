from typing import Iterator

from .nodes import PomapNode, Leaf, Lift, Split, Feed


_RESERVED_COLUMN_NAMES = {"__pomap_row_index"}


def _validate(root: PomapNode) -> None:
    _check_unique_node_names(root)
    _check_no_row_filter_above_feed(root, ancestor=None)
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


def _check_no_row_filter_above_feed(node: PomapNode, ancestor: str | None) -> None:
    "Reject Lift or Split as an ancestor of Feed."
    if isinstance(node, Feed) and ancestor is not None:
        raise ValueError(
            f"Feed {node.name!r} has a {ancestor} as an ancestor. This is currently unsupported"
        )
    # Lift dominates Split in severity, so prefer it in the error message.
    if isinstance(node, Lift):
        next_ancestor = "Lift"
    elif isinstance(node, Split):
        next_ancestor = ancestor if ancestor == "Lift" else "Split"
    else:
        next_ancestor = ancestor
    for child in node.children:
        _check_no_row_filter_above_feed(child, next_ancestor)


def _check_no_reserved_leaf_labels(node: PomapNode) -> None:
    if isinstance(node, Leaf) and node.label in _RESERVED_COLUMN_NAMES:
        raise ValueError(
            f"Leaf label {node.label!r} collides with a reserved column name."
        )
    for child in node.children:
        _check_no_reserved_leaf_labels(child)