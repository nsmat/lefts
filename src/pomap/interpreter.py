from .nodes import PomapNode, Leaf, Lift, Ensemble, LearnsFrom
from typing import Iterator, Tuple, Any, Optional
from .label import Label
from polars import DataFrame, Series, Expr, lit
from dataclasses import dataclass
from inspect import signature


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

    for i, child in enumerate(node.children):
        next_prefix = prefix + "    "
        lines.append(_print_tree(child, next_prefix, is_root=False))

    return "\n".join(lines)


def _mark_in_train_data_for_label(node: PomapNode, df: DataFrame, label: Label):
    # TODO
    ...


def _mark_in_test_data_for_label(node: PomapNode, df: DataFrame, label: Label):
    # TODO
    ...


def _validate_tree(node: PomapNode, observed_names=None):
    # TODO add namespace checking. Need to check both types and namespaces
    ...


def _collect_labels(node: PomapNode, label_context=None) -> Iterator[Label]:
    label_context = label_context or {}

    match node:
        case Leaf(label=l):
            yield Label(leaf=l, **label_context)

        case Lift(child=child, values=values, name=name):
            # Under a lift, we will take the cartesian product
            # Of the existing labels with the lift values
            for atomic in values:
                extended_label_context = label_context | {name: atomic}
                yield from _collect_labels(child, extended_label_context)

        case Ensemble() | LearnsFrom():
            for child in node.children:
                yield from _collect_labels(child, label_context)

        case _:
            raise NotImplementedError(
                f"Not implemented for node type {node.__class__.__name__}"
            )


def _collect_leaves(node: PomapNode) -> Iterator[Leaf]:
    match node.children:
        case []:
            yield node
        case children:
            for child in children:
                yield from _collect_leaves(child)


def _get_train_df_for_label(node: PomapNode, df: DataFrame, label: Label) -> DataFrame:
    match node:
        case Leaf() | LearnsFrom():
            return df

        case Lift(child=child, name=name, train_filter=train_mask_for_label):
            # In a lift, we apply the mask specified in the lift
            # To the train df returned by the child

            # First, we split the label into the part that's relevant for the lift
            # and the part that is relevant for the rest of the tree.
            lift_label, child_label = label[name], label.drop(name)
            mask_expr = train_mask_for_label(lift_label)
            return _get_train_df_for_label(child, df, label=child_label).filter(
                mask_expr
            )

        case Ensemble(models):
            # In an ensemble, we just pass through the dataframe
            # from the appropriate child. The correct child is the one
            # That matches the label.
            for child in models:
                if label in _collect_labels(child):
                    return _get_train_df_for_label(child, df, label=label)
            raise ValueError(f"Label {label} not present in model labels")

        case _:
            raise NotImplementedError(
                f"Not implemented for node type {node.__class__.__name__}"
            )


def _get_test_df_for_label(node: PomapNode, df: DataFrame, label: Label) -> DataFrame:
    match node:
        case Leaf():
            return df

        case Leaf() | LearnsFrom():
            return df

        case Lift(child=child, name=name, test_filter=test_mask_for_label):
            # In a lift, we apply the mask specified in the lift
            # To the train df returned by the child

            # First, we plit the label into the part that's relevant for the lift
            # and the part that is relevant for the rest of the tree.
            lift_label, child_label = label[name], label.drop(name)
            mask_expr = test_mask_for_label(lift_label)
            return _get_test_df_for_label(child, df, label=child_label).filter(
                mask_expr
            )

        case Ensemble(models=models):
            # In an ensemble, we just pass through the dataframe
            # from the appropriate child. The correct child is the one
            # That matches the label.
            for child in models:
                if label in _collect_labels(child):
                    return _get_test_df_for_label(child, df, label=label)

        case _:
            raise NotImplementedError(
                f"Not implemented for node type {node.__class__.__name__}"
            )


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
            full_label = Label(leaf=leaf_label, **label_context)
            yield full_label, train_mask, validation_mask, test_mask

        case Lift(
            child=child,
            name=name,
            values=values,
            train_filter=train_mask_for_label,
            validation_filter=validation_mask_for_label,
            test_filter=test_mask_for_label,
        ):
            for atomic in values:
                next_label_context = {name: atomic, **label_context}

                lift_train_mask = train_mask_for_label(atomic)
                next_train_mask = lift_train_mask & train_mask

                lift_test_mask = test_mask_for_label(atomic)
                next_test_mask = lift_test_mask & test_mask

                if validation_mask_for_label is None:
                    validation_mask_for_label = validation_mask_for_label
                    next_validation_mask = validation_mask
                else:
                    next_validation_mask = validation_mask_for_label(atomic)
                    if validation_mask is not None:
                        next_validation_mask = next_validation_mask & validation_mask

                yield from _collect_masks(
                    child,
                    next_label_context,
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
) -> Tuple[dict[Label, Any], dict[str, Any]]:

    label_context = label_context or dict()
    hyperparameters = hyperparameters or dict()

    # Define output types
    fitted_models: dict[Label, Any] = {}
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
            model_label = Label(leaf=label, **label_context)

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
            for atomic in values:
                extended_label_context = {**label_context, name: atomic}

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

        case _:
            raise ValueError(f"Unknown node type {type(node)}")

    return fitted_models, output_hyperparameters


def _predict(node: PomapNode, models: dict, df: DataFrame):
    if "__pomap_row_index" in df.columns:
        raise ValueError(
            "Trying to create column __pomap_row_index but it already exists"
        )

    df = df.with_row_index(name="__pomap_row_index")

    labels = _collect_labels(node)
    for label in labels:
        test_df = _get_test_df_for_label(node, df, label)
        predictions = models[label].predict(test_df)
        predictions = Series(name=label.column(), values=predictions)

        test_df = test_df.with_columns(predictions)

        df = df.join(
            test_df.select("__pomap_row_index", label.column()),
            on="__pomap_row_index",
            coalesce=True,
            how="left",
        )

    df = df.drop("__pomap_row_index")

    return df
