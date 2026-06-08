from polars import Expr, lit

from ..nodes import LeftsNode, Leaf, Lift, Split
from .labels import _make_label


def _collect_masks(
    node: LeftsNode,
    label_context: dict | None = None,
    train_mask: Expr | None = None,
    validation_mask: Expr | None = None,
    test_mask: Expr | None = None,
) -> dict[str, dict[str, Expr | None]]:
    """
    Returns a dictionary of label -> {train: train_mask, validation: validation_mask, test: test_mask}
    """
    label_context = label_context or dict()
    train_mask = train_mask if train_mask is not None else lit(True)
    test_mask = test_mask if test_mask is not None else lit(True)

    result = {}

    match node:
        case Leaf(label=leaf_label):
            full_label = _make_label(leaf_label, label_context)
            result[full_label] = {
                "train": train_mask,
                "validation": validation_mask,
                "test": test_mask,
            }

        case Lift(
            child=child,
            name=name,
            values=values,
            train_filter=train_filter,
            validation_filter=validation_filter,
            test_filter=test_filter,
        ):
            for value in values:
                next_label_context = {**label_context, name: value}
                next_train_mask = train_filter(value) & train_mask
                next_test_mask = test_filter(value) & test_mask

                if validation_filter is None:
                    next_validation_mask = validation_mask
                else:
                    next_validation_mask = validation_filter(value)
                    if validation_mask is not None:
                        next_validation_mask = next_validation_mask & validation_mask

                result.update(
                    _collect_masks(
                        child,
                        next_label_context,
                        next_train_mask,
                        next_validation_mask,
                        next_test_mask,
                    )
                )

        case Split(
            child=child,
            train_filter=train_filter,
            validation_filter=validation_filter,
            test_filter=test_filter,
        ):
            next_train_mask = train_filter & train_mask
            next_test_mask = test_filter & test_mask

            if validation_filter is None:
                next_validation_mask = validation_mask
            else:
                next_validation_mask = validation_filter
                if validation_mask is not None:
                    next_validation_mask = next_validation_mask & validation_mask

            result.update(
                _collect_masks(
                    child,
                    label_context,
                    next_train_mask,
                    next_validation_mask,
                    next_test_mask,
                )
            )

        case LeftsNode():
            for child in node.children:
                result.update(
                    _collect_masks(
                        child, label_context, train_mask, validation_mask, test_mask
                    )
                )

    return result
