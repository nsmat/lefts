## Lefts Models and the 'leaf' command

Any lefts workflow must start with the leaf command. The first argument of leaf is a constructor of an 'estimator', that is, an object with a .fit and .predict method which operate on polars dataframes.

For example, this is a valid estimator:

```python
class LinearRegression:
    def __init__(self, x = 'x', y = 'y'):
        self.x = x
        self.y = y
        self.slope = None
        self.intercept = None

    def fit(self, training_set: pl.DataFrame) -> None:
        result = scipy.stats.linregress(training_set[self.x].to_numpy(), training_set[self.y].to_numpy())
        self.slope = result.slope
        self.intercept = result.intercept

    def predict(self, df: pl.DataFrame) -> Iterable:
        return self.slope * df[self.x].to_numpy() + self.intercept
```

Which we would pass to leaf like so:
```python
model = leaf(LinearRegression, label='linear_regression')
```

leaf returns a lefts Model type. Like an estimator, it has a .fit and .predict methods, but it also has additional properties which allow it to be modified and manipulated by subsequent lefts commands. 

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

# Lefts commands

### Split

Let's illustrate this with the simple command 'split':

```python
train_test_split = split(
    name='split_linear_regression',
    model=model,
    train_filter=pl.col('date') < dt.date(2026, 1, 1),
    test_filter=pl.col('date') >= dt.date(2026, 1, 1),
)
```

The resulting model will only fit on data before 2026-01-01, and only generate predictions on data from that date onwards. Beneath the hood, we have taken the intersection of the new filters and the ones already existing on model.

The behaviour can be checked using `mark_train_validation_test_rows`:

```python
train_test_split.mark_train_validation_test_rows(df)
```

| date | linear_regression__train | linear_regression__test |
|------|--------------------------|-------------------------|
| 2025-11-01 | true | false |
| 2025-12-01 | true | false |
| 2026-01-01 | false | true |
| 2026-02-01 | false | true |

This illustrates an important invariant of lefts: any command that modifies the train, validate and test filters will only make them more restrictive.

### Ensemble

Ensemble combines multiple Models so that they always fit and predict in parallel:

```python
comparison = ensemble(
    'linear+lasso',
    leaf(LinearRegression, label='linear_regression'),
    leaf(LassoRegression, label='lasso_regression'),
)

comparison.fit(df)
comparison.predict(df)
```

| date | linear_regression | lasso_regression |
|------|------------------|-----------------|
| 2025-11-01 | 2.01 | 1.99 |
| 2025-12-01 | 3.98 | 4.02 |
| 2026-01-01 | 6.03 | 5.97 |
| 2026-02-01 | 7.99 | 8.01 |

By default each model gets its own output column, named after the model label.

The 'aggregate_with' parameter takes a function which operates on a set of dataframe columns and returns a single column. For example, you could take the mean of all the models in the ensemble:

```python
comparison = ensemble(
    'mean(linear+lasso)',
    leaf(LinearRegression, label='linear_regression'),
    leaf(LassoRegression, label='lasso_regression'),
    aggregate_with=pl.mean_horizontal,
)
```

| date | mean(linear+lasso) |
|------|--------------------|
| 2025-11-01 | 2.00 |
| 2025-12-01 | 4.00 |
| 2026-01-01 | 6.00 |
| 2026-02-01 | 8.00 |

The output column takes the name of the ensemble operation.

### Lift

Lift will train multiple copies of a model on different, potentially overlapping subsets of the data. For example, suppose you want to create a 'rolling retrain' workflow, where the same model is retrained every month.

```python
dates = [dt.date(2025, 11, 1), dt.date(2025, 12, 1), dt.date(2026, 1, 1)]
linear_regression = leaf(LinearRegression, label='linear_regression')

rolling = lift(
    linear_regression,
    name='monthly_retrain',
    values=dates,
    train_filter=lambda d: pl.col('date') < d,
    test_filter=lambda d: pl.col('date').dt.month() == d.month,
    aggregate_with=None
)
```

When fit is called, three separate instances of LinearRegression will be created - one for each of the listed dates. Each one will be trained on data from before that model's cutoff. When predict is called, each of the models will only be evaluated on the data for the subsequent month.

| date | linear_regression[monthly_retrain=2025-11-01] | linear_regression[monthly_retrain=2025-12-01] | linear_regression[monthly_retrain=2026-01-01] |
|------|----------------------------------------------|----------------------------------------------|----------------------------------------------|
| 2025-11-01 | 2.01 | null | null |
| 2025-12-01 | null | 3.98 | null |
| 2026-01-01 | null | null | 6.02 |

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

# Composing lefts commands

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



