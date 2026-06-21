# tune

`tune` fits a **source** model first, then calls a user-supplied **logic** function to derive hyperparameters, and finally passes those hyperparameters to the **consumer** model's factory. The consumer is instantiated with the derived hyperparameters as keyword arguments.

## Signature

```python
tune(name, consumer, source, logic) -> Model
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique name for this tune operation |
| `consumer` | `Model` | The model to be instantiated with learned hyperparameters |
| `source` | `Model` | The model fitted first to inform hyperparameter selection |
| `logic` | `Callable[[fitted_source, df], dict]` | Maps the fitted source and full DataFrame to a hyperparameter dict |

## Example

```python
from lefts import leaf, tune
from lefts.helpers import tabular_model
from sklearn.linear_model import Ridge

source = leaf(tabular_model(Ridge(), features=["x"], target="y"), label="source_ridge")
consumer = leaf(tabular_model(Ridge(), features=["x"], target="y"), label="consumer_ridge")

def derive_alpha(fitted_source_model, df):
    # Inspect the fitted source to pick an alpha for the consumer
    return {"alpha": 0.5}

model = tune("tuned", consumer=consumer, source=source, logic=derive_alpha)
```

## Notes

- The `logic` function receives the fitted source `Model` object and the full DataFrame.
- The returned dict is unpacked as keyword arguments to the consumer's leaf factory.
