# lift

`lift` replicates a model across a list of values, giving each copy its own train/test/validation filter derived from that value. Useful for rolling retrains, cross-validation, or any scenario where you want per-slice models.

## Signature

```python
lift(model, values, name, train_filter, test_filter, validation_filter=None, aggregate_with=None) -> Model
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `Model` | The model to replicate |
| `values` | `Iterable` | One copy of the model is trained per value |
| `name` | `str` | Unique name for this lift; used to label each child as `"<leaf>[<name>=<value>]"` |
| `train_filter` | `Callable[[value], Expr]` | Maps each value to a boolean Polars expression selecting train rows |
| `test_filter` | `Callable[[value], Expr]` | Maps each value to a boolean Polars expression selecting test rows |
| `validation_filter` | `Callable[[value], Expr] \| None` | Optional validation filter |
| `aggregate_with` | `Callable \| None` | Post-processes the per-value output columns (e.g. `pl.coalesce`) into a single column named `name` |

## Example

```python
import polars as pl
import datetime as dt
from lefts import leaf, lift
from lefts.helpers import tabular_model
from sklearn.linear_model import Ridge

dates = pl.datetime_range(
    start=dt.datetime(2020, 1, 1),
    end=dt.datetime(2020, 6, 1),
    interval="1mo",
    eager=True,
).to_list()

base = leaf(tabular_model(Ridge(), features=["x"], target="y"), label="ridge")

rolling = lift(
    base,
    name="rolling",
    values=dates,
    train_filter=lambda d: pl.col("date") < d,
    test_filter=lambda d: pl.col("date").dt.month() == d.month,
    aggregate_with=pl.coalesce,
)
```

## Notes

- `Lift` cannot be an ancestor of `Feed` — lift the inner model first, then feed.
- Without `aggregate_with`, `.predict()` returns one column per value; with it, a single column named `name`.
