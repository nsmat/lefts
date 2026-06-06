from .nodes import PomapNode, Leaf, Lift, Split, Feed, LearnsFrom
from .interpreter import _collect_labels


_RESERVED_COLUMN_NAMES = {"__pomap_row_index"}


def _validate(root: PomapNode) -> None:
    """Validate a Pomap tree. Raises ValueError on failure, returns None on success."""
    _check_unique_decorated_labels(root)
    _check_unique_lift_names(root, set())
    _check_no_row_filter_above_feed(root, ancestor=None)
    _check_no_reserved_leaf_labels(root)
    _check_learn_logic_callable(root)


def _check_unique_decorated_labels(root: PomapNode) -> None:
    labels = list(_collect_labels(root))
    seen: set[str] = set()
    duplicates: set[str] = set()
    for label in labels:
        if label in seen:
            duplicates.add(label)
        seen.add(label)
    if duplicates:
        raise ValueError(
            f"Duplicate decorated leaf labels: {sorted(duplicates)}. "
            "Each leaf must have a unique label after Lift decoration."
        )


def _check_unique_lift_names(node: PomapNode, ancestor_lifts: set[str]) -> None:
    next_ancestors = ancestor_lifts
    if isinstance(node, Lift):
        if node.name in ancestor_lifts:
            raise ValueError(
                f"Lift name {node.name!r} is reused along an ancestor chain. "
                "This would produce malformed labels."
            )
        next_ancestors = ancestor_lifts | {node.name}
    for child in node.children:
        _check_unique_lift_names(child, next_ancestors)


def _check_no_row_filter_above_feed(node: PomapNode, ancestor: str | None) -> None:
    """Reject Lift or Split as an ancestor of Feed.

    `_fit`'s Feed case re-roots the source's fit, silently dropping any outer
    row-filter masks — causing source leakage. Lift additionally causes a
    label-decoration mismatch that crashes outer predict. The Feed-contract fix
    tracked in CLAUDE.md will resolve both.
    """
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


def _check_learn_logic_callable(node: PomapNode) -> None:
    if isinstance(node, LearnsFrom) and not callable(node.learn_logic):
        raise ValueError(
            f"LearnsFrom {node.name!r} has a non-callable learn_logic of type "
            f"{type(node.learn_logic).__name__}."
        )
    for child in node.children:
        _check_learn_logic_callable(child)