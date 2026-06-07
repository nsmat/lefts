from dataclasses import dataclass
from typing import Iterator, Optional

from polars import DataFrame, Series

from ..nodes import PomapNode, Lift, Ensemble
from .labels import _make_label, _collect_labels
from .masks import _collect_masks


@dataclass
class _Model:
    root: PomapNode
    models: Optional[dict] = None
    hyperparameters: Optional[dict] = None

    def predict(self, df: DataFrame):
        return _predict(self.root, self.models, df)

    def fit(self, df: DataFrame):
        raise NotImplementedError()


def _get_aggregation_input_columns(
    node: PomapNode, label_context: dict
) -> Iterator[str]:
    match node:
        case Lift(child=child, name=name, values=values):
            for value in values:
                yield from _collect_labels(child, label_context | {name: value})
        case Ensemble():
            for child in node.children:
                yield from _collect_labels(child, label_context)


def _aggregate(node: PomapNode, df: DataFrame, label_context: dict) -> DataFrame:
    input_cols = list(_get_aggregation_input_columns(node, label_context))
    full_col = _make_label(node.name, label_context)
    df = df.with_columns(node.aggregate_with(*input_cols).alias(full_col))
    return df.drop(*input_cols)


def _apply_aggregations(
    node: PomapNode,
    df: DataFrame,
    label_context: dict | None = None,
) -> DataFrame:
    label_context = label_context or {}

    match node:
        case Lift(name=name, values=values, child=child, aggregate_with=fn):
            for value in values:
                df = _apply_aggregations(child, df, label_context | {name: value})
            if fn is not None:
                df = _aggregate(node, df, label_context)
        case Ensemble(aggregate_with=fn):
            for child in node.children:
                df = _apply_aggregations(child, df, label_context)
            if fn is not None:
                df = _aggregate(node, df, label_context)
        case PomapNode():
            for child in node.children:
                df = _apply_aggregations(child, df, label_context)

    return df


def _predict(
    node: PomapNode,
    models: dict,
    df: DataFrame,
    precomputed_masks: dict | None = None,
    label_context: dict | None = None,
):
    """Predict every leaf in `precomputed_masks` and apply aggregations.

    `precomputed_masks` and `label_context` are passthroughs for callers that
    already hold an outer context (e.g. `Feed`'s `_fit` augmentation, which
    passes a scoped subset of the outer dict and the current label context so
    decorated leaf labels and aggregated column names match the surrounding
    tree). When omitted, both default to root-level values: masks are computed
    from `node`'s subtree and the label context is empty.
    """
    label_context = label_context or {}

    if "__pomap_row_index" in df.columns:
        raise ValueError(
            "Trying to create column __pomap_row_index but it already exists"
        )

    df = df.with_row_index(name="__pomap_row_index")

    if precomputed_masks is None:
        precomputed_masks = _collect_masks(node)

    # We compute the full space of output columns first, then apply aggregation post-hoc
    for label, masks in precomputed_masks.items():
        test_df = df.filter(masks["test"])
        predictions = models[label].predict(test_df)
        predictions = Series(name=label, values=predictions)

        test_df = test_df.with_columns(predictions)

        df = df.join(
            test_df.select("__pomap_row_index", label),
            on="__pomap_row_index",
            coalesce=True,
            how="left",
        )

    df = _apply_aggregations(node, df, label_context)
    df = df.drop("__pomap_row_index")

    return df
