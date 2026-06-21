from dataclasses import dataclass
from typing import Iterator, Optional

from .params import PredictErrors, _check_literal

import polars as pl
from polars import DataFrame, Series

from ..nodes import LeftsNode, Leaf, Lift, Split, Ensemble, Tune, Feed
from .labels import _make_label, _collect_labels
from .masks import _collect_masks


@dataclass
class _Model:
    root: LeftsNode
    models: Optional[dict] = None
    hyperparameters: Optional[dict] = None
    logs: Optional[dict] = None
    exceptions: Optional[dict] = None

    def predict(self, df: DataFrame, errors: str = "raise"):
        return _predict(self.root, self.models, df, errors=errors)

    def fit(self, df: DataFrame):
        raise NotImplementedError()


def _get_aggregation_input_columns(
    node: LeftsNode, label_context: dict
) -> Iterator[str]:
    match node:
        case Lift(child=child, name=name, values=values):
            for value in values:
                yield from _collect_labels(child, label_context | {name: value})
        case Ensemble():
            for child in node.children:
                yield from _collect_labels(child, label_context)


def _aggregate(node: LeftsNode, df: DataFrame, label_context: dict) -> DataFrame:
    input_cols = list(_get_aggregation_input_columns(node, label_context))
    full_col = _make_label(node.name, label_context)
    df = df.with_columns(node.aggregate_with(*input_cols).alias(full_col))
    return df.drop(*input_cols)


def _predict(
    node: LeftsNode,
    models: dict,
    df: DataFrame,
    precomputed_masks: dict | None = None,
    label_context: dict | None = None,
    is_root: bool = True,
    errors: PredictErrors = "raise",
) -> DataFrame:

    label_context = label_context or {}

    if is_root:
        _check_literal(errors, PredictErrors, "errors")
        if "__lefts_row_index" in df.columns:
            raise ValueError(
                "Trying to create column __lefts_row_index but it already exists"
            )
        df = df.with_row_index(name="__lefts_row_index")
        if precomputed_masks is None:
            precomputed_masks = _collect_masks(node)

    match node:
        case Leaf(label=label):
            full_label = _make_label(label, label_context)

            if full_label not in models:
                match errors:
                    case "raise":
                        raise RuntimeError(
                            f"Model {full_label!r} was not fit. Call .fit() before .predict(), "
                            "or use errors='skip_unfit_models' / errors='output_nan'."
                        )
                    case "output_nan":
                        df = df.with_columns(pl.lit(None).alias(full_label))
                    case "skip_unfit_models":
                        pass
                    case _:
                        raise ValueError(f"Unhandled errors value {errors!r}")
            else:
                test_mask = precomputed_masks[full_label]["test"]
                test_df = df.filter(test_mask)
                predictions = models[full_label].predict(test_df)
                predictions = Series(name=full_label, values=predictions)
                test_df = test_df.with_columns(predictions)
                df = df.join(
                    test_df.select("__lefts_row_index", full_label),
                    on="__lefts_row_index",
                    coalesce=True,
                    how="left",
                )

        case Lift(
            child=child, name=name, values=values, aggregate_with=aggregation_function
        ):
            for value in values:
                df = _predict(
                    child,
                    models,
                    df,
                    precomputed_masks,
                    label_context | {name: value},
                    is_root=False,
                    errors=errors,
                )
            if aggregation_function is not None:
                df = _aggregate(node, df, label_context)

        case Split(child=child):
            df = _predict(
                child,
                models,
                df,
                precomputed_masks,
                label_context,
                is_root=False,
                errors=errors,
            )

        case Ensemble(aggregate_with=aggregation_function):
            for child in node.children:
                df = _predict(
                    child,
                    models,
                    df,
                    precomputed_masks,
                    label_context,
                    is_root=False,
                    errors=errors,
                )
            if aggregation_function is not None:
                df = _aggregate(node, df, label_context)

        case Feed(source=source, consumer=consumer):
            df = _predict(
                source,
                models,
                df,
                precomputed_masks,
                label_context,
                is_root=False,
                errors=errors,
            )
            df = _predict(
                consumer,
                models,
                df,
                precomputed_masks,
                label_context,
                is_root=False,
                errors=errors,
            )

        case Tune(consumer=consumer, source=source):
            df = _predict(
                source,
                models,
                df,
                precomputed_masks,
                label_context,
                is_root=False,
                errors=errors,
            )
            df = _predict(
                consumer,
                models,
                df,
                precomputed_masks,
                label_context,
                is_root=False,
                errors=errors,
            )

    if is_root:
        df = df.drop("__lefts_row_index")

    return df
