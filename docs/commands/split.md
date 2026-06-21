# split

`split` restricts a model to fixed train, test, and (optionally) validation subsets defined by Polars expressions. Unlike `lift`, `split` takes a single filter per split rather than a function over a list of values.

## Signature

```python
split(name, model, train_filter, test_filter, validation_filter=None) -> Model
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique name for this split |
| `model` | `Model` | The model to apply the split to |
| `train_filter` | `pl.Expr` | Boolean expression selecting train rows |
| `test_filter` | `pl.Expr` | Boolean expression selecting test rows |
| `validation_filter` | `pl.Expr \| None` | Optional boolean expression selecting validation rows |

## Example

```python
import polars as pl
from lefts import leaf, split
from lefts.helpers import tabular_model
from sklearn.linear_model import Ridge

model = split(
    name="time_split",
    model=leaf(tabular_model(Ridge(), features=["x"], target="y"), label="ridge"),
    train_filter=pl.col("date") < pl.lit("2023-01-01").str.to_date(),
    test_filter=pl.col("date") >= pl.lit("2023-01-01").str.to_date(),
)
```

## Notes

- Train and test sets may overlap — Lefts does not enforce disjoint splits.
- Use `lift` instead when you want multiple splits derived from a list of values.
