# leaf

`leaf` is the entry point for any Lefts workflow. It wraps a **model factory** — a callable that returns an object with `.fit` and `.predict` — into a `Model` that can be composed with other Lefts commands.

## Signature

```python
leaf(model_constructor, label) -> Model
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_constructor` | `Callable[..., Any]` | A zero-argument (or keyword-argument) callable that returns a model with `.fit` and `.predict` |
| `label` | `str` | A unique name for this leaf within the workflow |

## Model interface contract

The object returned by `model_constructor` must implement:

```python
def fit(self, training_set: pl.DataFrame, validation_set: pl.DataFrame = None): ...
def predict(self, df: pl.DataFrame) -> Iterable: ...
```

`validation_set` is optional — omit it if your model does not use a validation set.

## Example

```python
from lefts import leaf
from lefts.helpers import tabular_model
from sklearn.linear_model import Ridge

model = leaf(
    tabular_model(Ridge(), features=["x1", "x2"], target="y"),
    label="ridge",
)
```

## Notes

- All labels in a workflow must be **globally unique**.
- Use `lefts.helpers.tabular_model` to adapt any sklearn-compatible estimator without writing a custom class.
