from .nodes import PomapNode, Leaf, Lift, Feed, LearnsFrom
from .interpreter import _collect_labels


_RESERVED_COLUMN_NAMES = {"__pomap_row_index"}


def _validate(root: PomapNode) -> None:
    """Validate a Pomap tree. Raises ValueError on failure, returns None on success."""
    _check_unique_decorated_labels(root)
    _check_unique_lift_names(root, set())
    _check_no_lift_above_feed(root, under_lift=False)
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


def _check_no_lift_above_feed(node: PomapNode, under_lift: bool) -> None:
    if isinstance(node, Feed) and under_lift:
        raise ValueError(
            f"Feed {node.name!r} has a Lift as an ancestor. This is currently "
            "unsupported — the Feed re-roots its source's fit, dropping the "
            "outer Lift's filters (causing CV leakage) and the source's stored "
            "key won't match the lift-decorated key the outer predict will look "
            "up. See CLAUDE.md 'Known design issues' for the planned Feed-contract "
            "fix. Workaround: move row filters into the source/consumer subtrees, "
            "or use Split when you don't need fan-out."
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


def _check_learn_logic_callable(node: PomapNode) -> None:
    if isinstance(node, LearnsFrom) and not callable(node.learn_logic):
        raise ValueError(
            f"LearnsFrom {node.name!r} has a non-callable learn_logic of type "
            f"{type(node.learn_logic).__name__}."
        )
    for child in node.children:
        _check_learn_logic_callable(child)