from typing import Iterator

from ..nodes import LeftsNode, Leaf, Lift, Split, Ensemble, Feed, Tune


def _make_label(leaf_name: str, label_context: dict) -> str:
    if not label_context:
        return leaf_name
    dims = ", ".join(f"{k}={v}" for k, v in label_context.items())
    return f"{leaf_name}[{dims}]"


def _format_list(items, print_all_labels: bool) -> str:
    items = list(items)
    if print_all_labels or len(items) <= 6:
        body = ", ".join(str(i) for i in items)
    else:
        body = f"{items[0]}, ..., {items[-1]}"
    return f"[{body}]"


def _aggregation_suffix(node: LeftsNode) -> str:
    fn = getattr(node, "aggregate_with", None)
    if fn is None:
        return ""
    fn_name = getattr(fn, "__name__", None) or repr(fn)
    return f'  ⇒ {fn_name} → "{node.name}"'


def _count_models(node: LeftsNode) -> int:
    match node:
        case Leaf():
            return 1
        case Lift(child=child, values=values):
            return len(values) * _count_models(child)
        case _:
            return sum(_count_models(child) for child in node.children)


def _collect_unaggregated_labels(
    node: LeftsNode, label_context: dict | None = None
) -> Iterator[str]:
    """Collects the full set of labels from the leaves, ignoring aggregation"""
    label_context = label_context or {}
    match node:
        case Leaf(label=label):
            yield _make_label(label, label_context)
        case Lift(child=child, name=name, values=values):
            for value in values:
                yield from _collect_unaggregated_labels(
                    child, label_context | {name: value}
                )
        case LeftsNode():
            for child in node.children:
                yield from _collect_unaggregated_labels(child, label_context)


def _count_fit_models(node: LeftsNode, label_context: dict, models: dict) -> int:
    return len(set(_collect_unaggregated_labels(node, label_context)) & set(models))


def _node_header(
    node: LeftsNode,
    print_all_labels: bool,
    models: dict | None = None,
    label_context: dict | None = None,
    show_count: bool = True,
) -> str:
    label_context = label_context or {}

    model_str = ""
    if show_count: # Will be False for all children beneath a Lift
        count = _count_models(node)
        failed_str = ""
        if models is not None:
            fit_count = _count_fit_models(node, label_context, models)
            failed = count - fit_count
            if failed > 0:
                failed_str = f", {failed} failed"
        model_str = f" ({count} model{'' if count == 1 else 's'}{failed_str})"

    match node:
        case Leaf(label=label):
            return f"Leaf '{label}'{model_str}"
        case Lift(name=name, values=values):
            vals = _format_list(values, print_all_labels)
            return f"Lift '{name}'{model_str}: {vals}{_aggregation_suffix(node)}"
        case Split(name=name):
            return f"Split '{name}'{model_str}"
        case Ensemble(name=name):
            return f"Ensemble '{name}'{model_str}{_aggregation_suffix(node)}"
        case Tune(name=name):
            return f"Tune '{name}'{model_str}"
        case Feed(name=name):
            return f"Feed '{name}'{model_str}"
        case _:
            return getattr(node, "name", repr(node))


def _print_tree(
    node: LeftsNode,
    print_all_labels: bool = False,
    prefix: str = "",
    is_root: bool = True,
    is_last: bool = True,
    models: dict | None = None,
    label_context: dict | None = None,
    show_count: bool = True,
) -> str:
    label_context = label_context or {}
    header = _node_header(
        node, print_all_labels, models=models, label_context=label_context, show_count=show_count,
    )

    if is_root:
        outputs = _format_list(_collect_labels(node), print_all_labels)
        lines = [f"{header}  → outputs: {outputs}"]
        child_prefix = "    "
    else:
        connector = "└── " if is_last else "├── "
        lines = [f"{prefix}{connector}{header}"]
        child_prefix = prefix + ("    " if is_last else "│   ")

    children = list(node.children)
    child_models = None if isinstance(node, Lift) else models
    child_show_count = not isinstance(node, Lift)
    for i, child in enumerate(children):
        lines.append(
            _print_tree(
                child,
                print_all_labels,
                prefix=child_prefix,
                is_root=False,
                is_last=(i == len(children) - 1),
                models=child_models,
                label_context=label_context,
                show_count=child_show_count,
            )
        )

    return "\n".join(lines)


def _collect_labels(
    node: LeftsNode, label_context: dict | None = None
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
        case LeftsNode():
            for child in node.children:
                yield from _collect_labels(child, label_context)
