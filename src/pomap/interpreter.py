import warnings

from .nodes import PomapNode, Leaf, Lift, Split, Ensemble, LearnsFrom, Feed
from typing import Iterator, Tuple, Any, Optional
from polars import DataFrame, Series, Expr, lit
from dataclasses import dataclass
from inspect import signature


def _make_label(leaf_name: str, label_context: dict) -> str:
    if not label_context:
        return leaf_name
    dims = ", ".join(f"{k}={v}" for k, v in label_context.items())
    return f"{leaf_name}[{dims}]"


def _print_tree(node: PomapNode, prefix="", is_root=True) -> str:
    # Leaf formatting
    if not hasattr(node, "children") or not node.children:
        label = getattr(node, "label", str(node))
        if is_root:
            return label
        return f"{prefix}└── {label}"

    # Internal node
    if is_root:
        lines = [f"{node.tree_repr}"]
    else:
        lines = [f"{prefix}└── {node.tree_repr}"]

    for child in node.children:
        next_prefix = prefix + "    "
        lines.append(_print_tree(child, next_prefix, is_root=False))

    return "\n".join(lines)


def _collect_labels(
    node: PomapNode, label_context: dict | None = None
) -> Iterator[str]:
    label_context = label_context or {}

    match node:
        case Lift(aggregate_with=fn) | Ensemble(aggregate_with=fn) if fn is not None:
            # In the case where we have an aggregation function, we
            # halt because all child labels will be pulled into the aggregated column
            yield _make_label(node.name, label_context)
        case Leaf(label=label):
            yield _make_label(label, label_context)
        case Lift(child=child, name=name, values=values):
            for value in values:
                yield from _collect_labels(child, label_context | {name: value})
        case PomapNode():
            for child in node.children:
                yield from _collect_labels(child, label_context)


@dataclass
class _Model:
    root: PomapNode
    models: Optional[dict] = None
    hyperparameters: Optional[dict] = None

    def predict(self, df: DataFrame):
        return _predict(self.root, self.models, df)

    def fit(self, df: DataFrame):
        raise NotImplementedError()


def _collect_masks(
    node: PomapNode,
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

        case PomapNode():
            for child in node.children:
                result.update(
                    _collect_masks(
                        child, label_context, train_mask, validation_mask, test_mask
                    )
                )

    return result


def _fit(
    node: PomapNode,
    df: DataFrame,
    hyperparameters: dict = None,
    label_context: dict = None,
    is_root=True,
    precomputed_masks: dict = None,
) -> Tuple[dict[str, Any], dict[str, Any]]:

    label_context = label_context or dict()
    hyperparameters = hyperparameters or dict()

    # Define output types
    fitted_models: dict[str, Any] = {}
    output_hyperparameters: dict[str, Any] = {}

    if is_root:
        precomputed_masks = _collect_masks(node)

    match node:
        case Leaf(label=label, factory=factory):
            # Note: it is safe to use the passed hyperparameters
            # Without further filtering on label, because the hyperparameters
            # Are passed from a LearnsFrom to every node beneath them in the tree.

            model = factory(**hyperparameters)
            model_label = _make_label(label, label_context)

            train_mask = precomputed_masks[model_label]["train"]
            validation_mask = precomputed_masks[model_label]["validation"]

            train_df = df.filter(train_mask)

            fit_signature = signature(model.fit)

            # Now we inspect the signature of fit to determine whether
            # It expects a validation set, and also to perform runtime checking
            # That it's signature conforms to expectations.

            allowable_fit_parameters = {"training_set", "validation_set"}
            excess_parameters = set(fit_signature.parameters) - allowable_fit_parameters
            assert excess_parameters == set(), (
                f"Model {label} .fit(...) should only have arguments {allowable_fit_parameters} "
                f" but has unexpected parameters {excess_parameters}"
            )

            # Now, if required, we compute the validation set and pass it through to the
            # fit function as a kwarg.
            fit_kwargs = dict()
            if (
                "validation_set" in fit_signature.parameters
                and validation_mask is not None
            ):
                fit_kwargs["validation_set"] = df.filter(validation_mask)

            elif ("validation_set" not in fit_signature.parameters) and (
                validation_mask is not None
            ):
                raise ValueError(
                    f"Validation set created but model {label} .fit has no validation_set argument"
                )

            model.fit(train_df, **fit_kwargs)

            fitted_models[model_label] = model

        case Lift(child=child, values=values, name=name):
            # Under a lift, we will take the cartesian product
            # Of the existing labels with the lift values
            # Filtering appropriately based on each value.
            for value in values:
                extended_label_context = {**label_context, name: value}

                child_models, child_hyperparameters = _fit(
                    child,
                    df,
                    hyperparameters,
                    extended_label_context,
                    False,
                    precomputed_masks,
                )

                fitted_models |= child_models
                output_hyperparameters |= child_hyperparameters

        case Split(child=child):
            child_models, child_hyperparameters = _fit(
                child,
                df,
                hyperparameters,
                label_context,
                False,
                precomputed_masks,
            )
            fitted_models |= child_models
            output_hyperparameters |= child_hyperparameters

        case Ensemble():
            for child in node.children:
                child_fitted_models, child_learned_hyperparameters = _fit(
                    child, df, hyperparameters, label_context, False, precomputed_masks
                )
                fitted_models |= child_fitted_models
                output_hyperparameters |= child_learned_hyperparameters

        case LearnsFrom(
            learner=learner, learns_from=learns_from, learn_logic=learn_logic
        ):
            source_models, learned_hyperparameters = _fit(
                learns_from,
                df,
            )

            learn_from_model = _Model(
                learns_from, source_models, learned_hyperparameters
            )
            learned_hyperparameters |= learn_logic(learn_from_model, df)

            learner_models, learner_hyperparameters = _fit(
                learner,
                df,
                learned_hyperparameters,
                label_context,
                False,
                precomputed_masks,
            )

            fitted_models |= source_models | learner_models
            output_hyperparameters |= learner_hyperparameters | learned_hyperparameters

        case Feed(source=source, consumer=consumer):
            _check_feed_row_compatibility(
                source, consumer, df, precomputed_masks, label_context
            )
            source_models, source_hyperparameters = _fit(
                source,
                df,
                hyperparameters,
                label_context,
                False,
                precomputed_masks,
            )

            # Predict will loop through every label which keys the precomputed_masks
            # So before running predict on the source we need to subset down to
            # Just the columns we need
            source_labels = _collect_masks(source, label_context).keys()
            source_precomputed_masks = {label: precomputed_masks[label] for label in source_labels}

            # We run predict with the source model, so the consumer has predictions available.
            augmented_df = _predict(
                source,
                source_models,
                df,
                source_precomputed_masks,
                label_context,
            )
            consumer_models, consumer_hyperparameters = _fit(
                consumer,
                augmented_df,
                hyperparameters,
                label_context,
                False,
                precomputed_masks,
            )
            fitted_models |= source_models | consumer_models
            output_hyperparameters |= source_hyperparameters | consumer_hyperparameters

        case _:
            raise ValueError(f"Unknown node type {type(node)}")

    return fitted_models, output_hyperparameters


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


def _check_feed_row_compatibility(
    source_root: PomapNode,
    consumer_root: PomapNode,
    df: DataFrame,
    precomputed_masks: dict,
    label_context: dict,
) -> None:
    """
    Emits warnings if there is suspicious behaviour in the train/test
    overlap of the source and consumer of a Feed node.

    Specifically it will warn if:
    - The test set of the source is not a subset of the consumer. This may indicate a data leak.
    - The test set of the source is a strict subset of the consumer, since this will cause NaNs in the fed column.
    """

    # It's possible to have multiple leaves with separate train/test
    # Specification as one source (i.e. an ensemble with an aggregate)
    # Hence, we take the union of all child masks.
    def union_mask(node: PomapNode, mask_kind: str) -> Expr:
        result = lit(False)
        for label in _collect_masks(node, label_context):
            result = result | precomputed_masks[label][mask_kind]
        return result

    source_train = union_mask(source_root, "train")
    source_test = union_mask(source_root, "test")
    consumer_train = union_mask(consumer_root, "train")

    leak_rows = df.filter(source_train & ~consumer_train).height
    if leak_rows > 0:
        warnings.warn(
            f"Feed: source's train set contains {leak_rows} rows not in consumer's "
            "train set. This may signal a data leak (if those rows are part of "
            "consumer's test set) or be intentional (if source legitimately trains "
            "on extra data outside consumer's scope).",
            UserWarning,
            stacklevel=4,
        )

    nan_rows = df.filter(consumer_train & ~source_test).height
    if nan_rows > 0:
        warnings.warn(
            f"Feed: {nan_rows} rows in consumer's train set are not in source's "
            "test set. Source's augmentation column will be NaN on those rows "
            "during consumer fit. If your consumer does not handle NaN features, "
            "consider expressing source as a CV `lift` with `aggregate_with=pl.coalesce` "
            "so source's test_mask covers consumer's train rows.",
            UserWarning,
            stacklevel=4,
        )


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
