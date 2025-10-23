from core.nodes import PomapNode, Leaf, Lift, Ensemble
from typing import Iterable, Optional, Callable, List, Any, Generator
from core.label import Label
from polars import DataFrame


def _print_tree(node: PomapNode, prefix='', is_root=True) -> str:
    # Leaf formatting
    if not hasattr(node, "children") or not node.children:
        label = getattr(node, "label", str(node))
        if is_root:
            return label
        return f"{prefix}└── {label}"

    # Internal node
    if is_root:
        lines = [f"{node.name}"]
    else:
        lines = [f"{prefix}└── {node.name}"]

    for i, child in enumerate(node.children):
        next_prefix = prefix + "    "
        lines.append(_print_tree(child, next_prefix, is_root=False))

    return "\n".join(lines)


def _validate_tree(node: PomapNode):
    # TODO Add namespace checking - note there's something a bit funny where
    # 'Ensemble' is used as the name for all Ensembles, but that doesn't actually
    # feed into the labels, so won't cause any clashes. I think we just need to disambiguate
    # The namespace name from the _print_tree name.
    ...


def _collect_labels(root: PomapNode) -> Generator[Label]:
    match root:
        case Leaf(label=l):
            yield Label(leaf=l)

        case Lift(child=child, atomics=atomics, name=name):
            # Under a lift, we will take the cartesian product
            # Of the existing labels with the lift atomics
            for child_label in _collect_labels(child):
                for atomic in atomics:
                    yield Label(**{**child_label, name: atomic})

        case Ensemble(models=models):
            for model in models:
                yield from _collect_labels(model)

        case _:
            return


def _collect_leaves(node: PomapNode) -> Generator[Leaf]:
    match node.children:
        case []:
            yield node
        case children:
            for child in children:
                yield from _collect_leaves(child)


def _get_train_df_for_label(node: PomapNode, df: DataFrame, label: Label) -> DataFrame:
    match node:
        case Leaf():
            return df

        case Lift(child=child, name=name, train_mask_for_label=train_mask_for_label):
            # In a lift, we apply the mask specified in the lift
            # To the train df returned by the child

            # First, we plit the label into the part that's relevant for the lift
            # and the part that is relevant for the rest of the tree.
            lift_label, child_label = label[name], label.drop(name)
            mask_expr = train_mask_for_label(lift_label)
            return _get_train_df_for_label(child, df, label=child_label).filter(mask_expr)

        case Ensemble(models):
            # In an ensemble, we just pass through the dataframe
            # from the appropriate child. The correct child is the one
            # That matches the label.
            for child in models:
                if label in _collect_labels(child):
                    return _get_train_df_for_label(child, df, label=label)
                else:
                    raise ValueError(f"Label {label} not present in leaves")


        case _:
            raise NotImplementedError(f"Not implemented for node type {node.__name__}")


def _get_test_df_for_label(node: PomapNode, df: DataFrame, label: Label) -> DataFrame:
    match node:
        case Leaf():
            return df

        case Lift(child=child, name=name, test_mask_for_label=test_mask_for_label):
            # In a lift, we apply the mask specified in the lift
            # To the train df returned by the child

            # First, we plit the label into the part that's relevant for the lift
            # and the part that is relevant for the rest of the tree.
            lift_label, child_label = label[name], label.drop(name)
            mask_expr = test_mask_for_label(lift_label)
            return _get_test_df_for_label(child, df, label=child_label).filter(mask_expr)

        case Ensemble(models):
            # In an ensemble, we just pass through the dataframe
            # from the appropriate child. The correct child is the one
            # That matches the label.
            for child in models:
                if label in _collect_labels(child):
                    return _get_test_df_for_label(child, df, label=label)

        case _:
            raise NotImplementedError(f"Not implemented for node type {node.__name__}")


def _fit(node: PomapNode, df:DataFrame) -> dict:
    models = {}
    labels = _collect_labels(node)

    # Create a dictionary of leaves, indexed by their atomic labels
    leaves = _collect_leaves(node)
    leaves = {leaf.label: leaf for leaf in leaves}

    for label in labels:
        leaf_label = label['leaf']
        model = leaves[leaf_label].factory()

        train_df = _get_train_df_for_label(node, df, label)
        model.fit(train_df)

        models[label] = model

    return models


def _predict(node: PomapNode, models: dict, df: DataFrame):
    if '__pomap_row_index' in df.columns:
        raise ValueError('Trying to create column __pomap_row_index but it already exists')

    df = df.with_row_index(name='__pomap_row_index')

    labels = _collect_labels(node)
    for label in labels:
        test_df = _get_test_df_for_label(node, df, label)
        predictions = models[label].predict(test_df).rename(label.column())

        test_df = test_df.with_columns(predictions)

        df = df.join(
            test_df.select('__pomap_row_index', label.column()),
            on='__pomap_row_index',
            coalesce=True,
            how='left'
        )

    return df