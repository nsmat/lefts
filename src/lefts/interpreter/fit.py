import warnings

from ..nodes import LeftsNode, Leaf, Lift, Split, Ensemble, Tune, Feed
from .labels import _make_label
from .masks import _collect_masks
from .predict import _Model, _predict
from typing import Tuple, Any
from polars import DataFrame, Expr, lit
from inspect import signature


def _fit(
    node: LeftsNode,
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
            # Are passed from a Tune to every node beneath them in the tree.
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

        case Tune(consumer=consumer, source=source, logic=logic):
            source_models, learned_hyperparameters = _fit(
                source,
                df,
            )

            tune_model = _Model(
                source, source_models, learned_hyperparameters
            )
            learned_hyperparameters |= logic(tune_model, df)

            consumer_models, consumer_hyperparameters = _fit(
                consumer,
                df,
                learned_hyperparameters,
                label_context,
                False,
                precomputed_masks,
            )

            fitted_models |= source_models | consumer_models
            output_hyperparameters |= consumer_hyperparameters | learned_hyperparameters

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

            augmented_df = _predict(
                source,
                source_models,
                df,
                precomputed_masks,
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


def _check_feed_row_compatibility(
    source_root: LeftsNode,
    consumer_root: LeftsNode,
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
    def union_mask(node: LeftsNode, mask_kind: str) -> Expr:
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
            "train set. This may signal a data leak - validate with Model.mark_train_validation_test_rows"
            "if you are unsure",
            UserWarning,
            stacklevel=4,
        )

    nan_rows = df.filter(consumer_train & ~source_test).height
    if nan_rows > 0:
        warnings.warn(
            f"Feed: {nan_rows} rows in consumer's train set are not in source's "
            "test set. Source's augmentation column will be NaN on those rows "
            "during consumer fit. If this is unexpected, validate your train and test filter"
            "behaviour with Model.mark_train_validation_test_rows",
            UserWarning,
            stacklevel=4,
        )
