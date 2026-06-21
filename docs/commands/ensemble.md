# ensemble

`ensemble` binds multiple models into a single model that fits and predicts all of them in parallel. The result of `.predict()` includes one column per child model (or a single aggregated column if `aggregate_with` is set).

## Signature

```python
ensemble(name, *models, aggregate_with=None) -> Model
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique name for this ensemble |
| `*models` | `Model` | Any number of `Model` objects to run in parallel |
| `aggregate_with` | `Callable \| None` | Post-processes the output columns into a single column |

## Example

```python
from lefts import leaf, ensemble
from lefts.helpers import tabular_model
from sklearn.linear_model import Ridge, Lasso

ridge = leaf(tabular_model(Ridge(), features=["x"], target="y"), label="ridge")
lasso = leaf(tabular_model(Lasso(), features=["x"], target="y"), label="lasso")

model = ensemble("linear_ensemble", ridge, lasso)
```

After `model.predict(df)`, the returned DataFrame has columns `ridge` and `lasso`.
