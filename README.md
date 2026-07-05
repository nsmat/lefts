# Lefts: composable machine-learning model transformations

Lefts is a very simple domain specific language for building complex machine learning workflows from simple ones. Starting with your favourite machine learning models, you can use Lefts operations to:
- Build complex ensembles.
- Build complex cross validation and hyper-parametrisation procedures.
- Allow a model to create features or targets for another model.
- And any creative combination of the above.

Without making subsequent model fitting, evaluation or experimentation any more complex than it was with the original model. This implementation is built on top of the excellent Polars DataFrame library.


# Installation

```bash
pip install lefts
```

# Commands
Lefts has five commands, which give it its name:
- **L**ift: trains multiple copies of a model across different subsets of data.
- **E**nsemble: Takes a set of models and makes them evaluate as one.
- **T**une: Allows a model to learn its hyperparameters from another.
- **F**eed: Allows the output of one model to be used as a feature or target by another.
- **S**plit: Trains a model on a given train/test/validation split.

These commands can be applied to any model that is defined by:
- a fit method, which maps from training data into the model parameters
- a predict method, which maps from model parameters and test data into predictions.

A Lefts command creates a new model by transforming these functions into a new .fit and .predict. Because this new model also has a .fit and .predict, it can be transformed with further Lefts commands.

# An example

The following code creates a rolling monthly retrain workflow: twelve copies of a Ridge regression, each trained on all data before its month and evaluated on that month only.

```python
from lefts import leaf, lift
from lefts.helpers import tabular_model

import polars as pl
from sklearn.linear_model import Ridge
import datetime as dt

features = ["temp", "humidity", "hour"]
target = "demand"

dates = [dt.date(2024, m, 1) for m in range(1, 13)]

model = lift(
    leaf(tabular_model(Ridge, features=features, target=target), label='ridge'),
    name='monthly_retrain',
    values=dates,
    train_filter=lambda d: pl.col('date') < d,
    test_filter=lambda d: pl.col('date').dt.month() == d.month,
    aggregate_with=pl.coalesce,
)

# Fits 12 independent models
model.fit(df)

# Each row gets the prediction from the model trained up to its month
predictions = model.predict(df)
```

You can inspect the workflow before running it with `model.print_tree()`:

```
Lift 'monthly_retrain' (12 models): [2024-01-01, ..., 2024-12-01]  ⇒ coalesce → "monthly_retrain"
    └── Leaf 'ridge'
```

For a more complex example combining lift, ensemble and other commands, see `notebooks/quantile_ensemble.ipynb`.
