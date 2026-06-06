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
):
    label_context = label_context or dict()

    train_mask = train_mask if train_mask is not None else lit(True)
    test_mask = test_mask if test_mask is not None else lit(True)

    match node:
        case Leaf(label=leaf_label):
            full_label = _make_label(leaf_label, label_context)
            yield full_label, train_mask, validation_mask, test_mask

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

                lift_train_mask = train_filter(value)
                next_train_mask = lift_train_mask & train_mask

                lift_test_mask = test_filter(value)
                next_test_mask = lift_test_mask & test_mask

                if validation_filter is None:
                    next_validation_mask = validation_mask
                else:
                    next_validation_mask = validation_filter(value)
                    if validation_mask is not None:
                        next_validation_mask = next_validation_mask & validation_mask

                yield from _collect_masks(
                    child,
                    next_label_context,
                    next_train_mask,
                    next_validation_mask,
                    next_test_mask,
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

            yield from _collect_masks(
                child,
                label_context,
                next_train_mask,
                next_validation_mask,
                next_test_mask,
            )

        case PomapNode():
            for child in node.children:
                yield from _collect_masks(
                    child, label_context, train_mask, validation_mask, test_mask
                )


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
        precomputed_masks = {p[0]: (p[1], p[2], p[3]) for p in precomputed_masks}

    match node:
        case Leaf(label=label, factory=factory):
            # Note: it is safe to use the passed hyperparameters
            # Without further filtering on label, because the hyperparameters
            # Are passed from a LearnsFrom to every node beneath them in the tree.

            model = factory(**hyperparameters)
            model_label = _make_label(label, label_context)

            train_mask = precomputed_masks[model_label][0]
            validation_mask = precomputed_masks[model_label][1]

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
            source_models, source_hyperparameters = _fit(source, df)
            source_model = _Model(source, source_models, source_hyperparameters)
            augmented_df = source_model.predict(df)

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


def _predict(node: PomapNode, models: dict, df: DataFrame):
    if "__pomap_row_index" in df.columns:
        raise ValueError(
            "Trying to create column __pomap_row_index but it already exists"
        )

    df = df.with_row_index(name="__pomap_row_index")

    precomputed_masks = {p[0]: (p[1], p[2], p[3]) for p in _collect_masks(node)}

    # We compute the full space of output columns first, then apply aggregation post-hoc
    for label, (_train_mask, _validation_mask, test_mask) in precomputed_masks.items():
        test_df = df.filter(test_mask)
        predictions = models[label].predict(test_df)
        predictions = Series(name=label, values=predictions)

        test_df = test_df.with_columns(predictions)

        df = df.join(
            test_df.select("__pomap_row_index", label),
            on="__pomap_row_index",
            coalesce=True,
            how="left",
        )

    df = _apply_aggregations(node, df)
    df = df.drop("__pomap_row_index")

    return df
