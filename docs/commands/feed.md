# feed

`feed` chains two models: the **source** is fitted and its predictions are appended to the DataFrame, then the **consumer** is fitted on this augmented DataFrame. The same augmentation happens at `.predict` time.

## Signature

```python
feed(name, source, consumer) -> Model
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique name for this feed operation |
| `source` | `Model` | Fitted first; its predictions are added as a new column |
| `consumer` | `Model` | Fitted on the augmented DataFrame |

## Example

```python
from lefts import leaf, feed
from lefts.helpers import tabular_model
from sklearn.linear_model import Ridge

source = leaf(
    tabular_model(Ridge(), features=["x1"], target="y"),
    label="stage1",
)
consumer = leaf(
    tabular_model(Ridge(), features=["x1", "stage1"], target="y"),  # uses source's output
    label="stage2",
)

model = feed("two_stage", source=source, consumer=consumer)
```

## Notes

- The source's prediction column is named after its leaf label and is available to the consumer under that name.
- `Lift` cannot be an ancestor of `Feed` — lift the inner model first, then feed.
