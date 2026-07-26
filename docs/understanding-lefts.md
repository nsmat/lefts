## What is lefts for?
Lefts is a tiny domain specific language for machine learning scientists, data analysts and quants that are applying machine learning models to real data. It is useful whenever you need to manage a large number of models simultaneously. For example, cross validation, ensembling or scaling an ML workflow to new targets usually explode the complexity as you need to manage train and test pipelines and prevent data leakage for every model at once. 

Lefts takes care of that for you, so you can focus on the modelling. By making orchestration simple, it allows you to experiment freely with new model architectures and explore previously intractible methods.

The core design features of lefts are:
* A wide range of behaviours is captured by a small number of commands. 
* Interacting with a workflow doesn't get more complicated as you experiment with it.
* Lefts commands always compose, so you can easily build up arbitrarily complex behaviour.
* Train/test splits are transparent and enforced by lefts.
* Lefts workflows are lazily evaluated, allowing issues to be caught before touching the training data.


## Lefts Models and the 'leaf' command

Any lefts workflow must start with the leaf command. The first argument of leaf is a constructor of an 'estimator', that is, an object with two methods:

* `fit(training_set: pl.DataFrame, validation_set: pl.DataFrame) -> None`, which trains your model and stores the parameters. The presence of a validation set argument is optional.
* `predict(test_set: pl.DataFrame) -> Iterable`: which evaluates your model and returns the predictions.

For example, this is a valid estimator:

```python
import polars as pl
import scipy.stats

class LinearRegression:
    def __init__(self, x = 'x', y = 'y'):
        self.x = x
        self.y = y
        self.slope = None
        self.intercept = None

    def fit(self, training_set: pl.DataFrame):
        result = scipy.stats.linregress(training_set[self.x].to_numpy(), training_set[self.y].to_numpy())
        self.slope = result.slope
        self.intercept = result.intercept

    def predict(self, df: pl.DataFrame):
        return self.slope * df[self.x].to_numpy() + self.intercept
```

Which we would pass to leaf like so:
```python
from lefts import leaf

model = leaf(LinearRegression, label='lr')
```

leaf returns a lefts Model type. Like an estimator, it has a .fit and .predict methods, but it also has additional properties which allow it to be modified and manipulated by subsequent lefts commands.

When a Model calls .fit, it will store all the fitted estimators it trains in a dictionary '.fitted'. If you want access to any of the methods of the original estimator class, they can be accessed there.

### Using sklearn estimators

Writing your own estimator class is often unnecessary - lefts includes a function that wraps an sklearn estimator to be ready for feeding into leaf:

```python
from lefts.helpers import tabular_model
from sklearn.linear_model import Lasso

model = leaf(
    tabular_model(Lasso, features=['x'], target='y'),
    label='lasso',
)
```

## Lefts commands

### Split

Every Model has functions ('filters') that determine whether a row of a given dataframe is part of the train, test and validation periods. These filters are used to ensure only the right data is seen when we call .fit and .predict. During fitting, we only see train and validation data, during prediction, we only see test data.

Let's illustrate with the simple command 'split':

```python
from lefts import split
import datetime as dt

train_test_split = split(
    name='split_linear_regression',
    model=model,
    train_filter=pl.col('date') < dt.date(2026, 1, 1),
    test_filter=pl.col('date') >= dt.date(2026, 1, 1),
)
```

The resulting model will only fit on data before 2026-01-01, and only generate predictions on data from that date onwards.

The behaviour can be checked using `mark_train_validation_test_rows`:

```python
train_test_split.mark_train_validation_test_rows(df)
```

```
shape: (4, 3)
┌────────────┬───────────┬──────────┐
│ date       ┆ lr__train ┆ lr__test │
│ ---        ┆ ---       ┆ ---      │
│ date       ┆ bool      ┆ bool     │
╞════════════╪═══════════╪══════════╡
│ 2025-11-01 ┆ true      ┆ false    │
│ 2025-12-01 ┆ true      ┆ false    │
│ 2026-01-01 ┆ false     ┆ true     │
│ 2026-02-01 ┆ false     ┆ true     │
└────────────┴───────────┴──────────┘
```

 
A leaf considers every row eligible for both training and testing. When a command modifies filters, it does so by taking the intersection of its filters with those already on the model. This means that applying a command can only make the filters more restrictive.

### Ensemble

Ensemble combines multiple Models so that they always fit and predict in parallel:

```python
from lefts import ensemble

comparison = ensemble(
    'linear+lasso',
    leaf(LinearRegression, label='lr'),
    leaf(LassoRegression, label='lasso'),
)

comparison.fit(df)
comparison.predict(df)
```

```
shape: (4, 3)
┌────────────┬──────┬──────────────────┐
│ date       ┆ lr   ┆ lasso_regression │
│ ---        ┆ ---  ┆ ---              │
│ date       ┆ f64  ┆ f64              │
╞════════════╪══════╪══════════════════╡
│ 2025-11-01 ┆ 2.01 ┆ 1.99             │
│ 2025-12-01 ┆ 3.98 ┆ 4.02             │
│ 2026-01-01 ┆ 6.03 ┆ 5.97             │
│ 2026-02-01 ┆ 7.99 ┆ 8.01             │
└────────────┴──────┴──────────────────┘
```

By default each model gets its own output column, named after the model label.

The 'aggregate_with' parameter takes a function which operates on a set of dataframe columns and returns a single column. For example, you could take the mean of all the models in the ensemble:

```python
comparison = ensemble(
    'mean(linear+lasso)',
    leaf(LinearRegression, label='lr'),
    leaf(LassoRegression, label='lasso'),
    aggregate_with=pl.mean_horizontal,
)
```

```
shape: (4, 2)
┌────────────┬────────────────────┐
│ date       ┆ mean(linear+lasso) │
│ ---        ┆ ---                │
│ date       ┆ f64                │
╞════════════╪════════════════════╡
│ 2025-11-01 ┆ 2.0                │
│ 2025-12-01 ┆ 4.0                │
│ 2026-01-01 ┆ 6.0                │
│ 2026-02-01 ┆ 8.0                │
└────────────┴────────────────────┘
```

The output column takes the name of the ensemble operation.

### Lift

Lift will train multiple copies of a model on different, potentially overlapping subsets of the data. For example, suppose you want to create a 'rolling retrain' workflow, where the same model is retrained every month.

```python
from lefts import lift

dates = [dt.date(2025, 11, 1), dt.date(2025, 12, 1), dt.date(2026, 1, 1)]
lr = leaf(LinearRegression, label='lr')

rolling = lift(
    lr,
    name='monthly',
    values=dates,
    train_filter=lambda d: pl.col('date') < d,
    test_filter=lambda d: pl.col('date').dt.month() == d.month,
    aggregate_with=None
)
```

When fit is called, three separate instances of LinearRegression will be created - one for each of the listed dates. Each one will be trained on data from before that model's cutoff. When predict is called, each of the models will only be evaluated on the data for the subsequent month.

```
shape: (3, 4)
┌────────────┬────────────────────────┬────────────────────────┬────────────────────────┐
│ date       ┆ lr[monthly=2025-11-01] ┆ lr[monthly=2025-12-01] ┆ lr[monthly=2026-01-01] │
│ ---        ┆ ---                    ┆ ---                    ┆ ---                    │
│ date       ┆ f64                    ┆ f64                    ┆ f64                    │
╞════════════╪════════════════════════╪════════════════════════╪════════════════════════╡
│ 2025-11-01 ┆ 2.01                   ┆ null                   ┆ null                   │
│ 2025-12-01 ┆ null                   ┆ 3.98                   ┆ null                   │
│ 2026-01-01 ┆ null                   ┆ null                   ┆ 6.02                   │
└────────────┴────────────────────────┴────────────────────────┴────────────────────────┘
```

Each of the per-subset models will output its own column, as with ensemble. However, they will now be labelled with the values you lifted over.

Lift is a very powerful operation. It can be used for:

- Rolling retrain workflows.
- K-fold cross validation procedures.
- Doing simultaneous predictions across many targets.
- Injecting strong categorical features into your models.
- Dealing with Simpson's paradox.
- And others!

### Feed

Feed is used when one model relies on the output of another for either training or prediction. The 'source' is fitted first, its predictions are added to the dataframe as a new column, and then the consumer is fitted on this augmented dataframe. The same augmentation happens at predict time.

```python
from lefts import feed
from functools import partial

stage1 = leaf(LinearRegression, label='stage1')

stage_2_estimator = partial(LinearRegression, x='stage1') # Uses the column 'stage1' as its feature
stage2 = leaf(stage_2_estimator, label='stage2')

chained = feed('two_stage', source=stage1, consumer=stage2)
```

### Tune

Tune is used for when you want to derive model hyperparameters by training and evaluating another model. 

Lefts places a convention around hyperparameters - any modifiable hyperparameters should be arguments in the constructor of the estimator. The free variables that Tune can optimise are the parameters of the estimator `__init__`.

In the slightly artificial example below, we imagine a linear regression model that allows for a learned offset on top of the prediction. 

```python
class LinearRegressionWithOffset:
    def __init__(self, x='x', y='y', offset=0.0):
        self.x = x
        self.y = y
        self.offset = offset
        self.slope = None
        self.intercept = None

    def fit(self, training_set: pl.DataFrame) -> None:
        result = scipy.stats.linregress(training_set[self.x].to_numpy(), training_set[self.y].to_numpy())
        self.slope = result.slope
        self.intercept = result.intercept

    def predict(self, df: pl.DataFrame) -> Iterable:
        return self.slope * df[self.x].to_numpy() + self.intercept + self.offset
```

Tune requires that we specify how we will derive the hyperparameter from the source model. This is done by passing a Python function that has access to the trained source_model and the training data. 

```python
def derive_offset(source_model, df):
    df = source_model.predict(df)
    # The source model will be labelled 'source', so that is the name of the column that will contain the predictions.
    df = df.with_columns(residual=pl.col('source') - pl.col('y'))
    bias = df['residual'].mean()
    return {'offset': bias}
```

Then the hyperparameter tuning workflow provided by using Tune:

```python
from lefts import tune

source   = leaf(LinearRegression, label='source')
consumer = leaf(LinearRegressionWithOffset, label='consumer')

tuned = tune('bias_correction', source=source, consumer=consumer, logic=derive_offset)

tuned.fit(df)
```

Is roughly equivalent to:

```python
source.fit(df)

hyperparameters = derive_offset(source, df)

parameterised_model = partial(LinearRegressionWithOffset, **hyperparameters)

consumer = leaf(parameterised_model, label='consumer')
consumer.fit(df)
```

## Composing lefts commands

All the commands listed in the previous section operate on Model objects and return Model objects. This means that the commands compose - you can string them together to build up complicated behaviour.

For example, in the following we ensemble two different lifted estimators.

```python
dates = [dt.date(2025, 11, 1), dt.date(2025, 12, 1), dt.date(2026, 1, 1)]

rolling_linear = lift(
    leaf(LinearRegression, label='linear'),
    name='monthly_linear',
    values=dates,
    train_filter=lambda d: pl.col('date') < d,
    test_filter=lambda d: pl.col('date').dt.month() == d.month,
    aggregate_with=pl.coalesce,
)

rolling_lasso = lift(
    leaf(LassoRegression, label='lasso'),
    name='monthly_lasso',
    values=dates,
    train_filter=lambda d: pl.col('date') < d,
    test_filter=lambda d: pl.col('date').dt.month() == d.month,
    aggregate_with=pl.coalesce,
)

comparison = ensemble('comparison', rolling_linear, rolling_lasso)
```

The resulting workflow is a tree of lefts commands. This illustrates why the first command was called 'leaf' - the basic estimators are always the leaves of this tree. 

lefts is always evaluated lazily. Before calling .fit and .predict, you could call `comparison.print_tree()` to see the resulting workflow:

```
Ensemble 'comparison' (6 models)  → outputs: [monthly_linear, monthly_lasso]
    ├── Lift 'monthly_linear' (3 models): [2025-11-01, 2025-12-01, 2026-01-01]  ⇒ coalesce → "monthly_linear"
    │   └── Leaf 'linear'
    └── Lift 'monthly_lasso' (3 models): [2025-11-01, 2025-12-01, 2026-01-01]  ⇒ coalesce → "monthly_lasso"
        └── Leaf 'lasso'
```


---

For a deeper look at how lefts is designed and why commands compose, see [Design philosophy](design-philosophy.md).



