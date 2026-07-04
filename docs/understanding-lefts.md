# Installing lefts

```
pip install lefts
```

# What does Lefts do?

- Introduce the idea that we are going to train big families of estimators.

# The leaf command and lefts Models

Any lefts workflow must start with the leaf command. The first argument of leaf is a constructor of an 'estimator', that is, an object with a .fit and .predict method which operate on polars dataframes.

For example, this is a valid estimator:

```python
class LinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, df: pl.DataFrame) -> None:
        result = scipy.stats.linregress(df["x"].to_numpy(), df["y"].to_numpy())
        self.slope = result.slope
        self.intercept = result.intercept

    def predict(self, df: pl.DataFrame) -> Iterable:
        return self.slope * df["x"].to_numpy() + self.intercept
```

Which we would pass to leaf like so:
```
model = leaf(LinearRegression, label='linear_regression')
```

leaf returns a lefts Model type, which can be thought of as a set of five functions:
- __fit__: creates estimator instances and optimises their parameters over the train set.
- __factory__: called by fit to instantiate all the estimators required by the workflow.
- __predict__: generates predictions on the test rows.
- __train_filter / test_filter / validation_filter__: Polars expressions which tell you whether each row of a dataframe is in the train/test/validation set.
- __labels__: which gives you unique labels for every estimator instance we create.

The Model output by 'leaf' will look a lot like the original estimator. Calling .fit and .predict will give you exactly the same result as if you just used LinearRegression. The filters will mark every row of the dataframe as being in the train, validation, or test sets. The only label is 'linear_regression'.

### Lefts commands are functors

However, as we apply further lefts commands, we will build up complex Models with heavily modified variants of the five functions. Each command is a functor that modifies those functions to yield a new, transformed model. 

For example, if we have a Lefts command T, we can create a new model by using it to transform each function.

```
● ┌─ Model ─────────────┐              ┌─ Model ────────────────┐                                                                                                                                                                   
  │ .fit:     fitter    │  ────T────►  │ .fit:     T(fitter)    │                                                                                                                                                                   
  │ .predict: predictor │              │ .predict: T(predictor) │                                                                                                                                                                   
  └─────────────────────┘              └────────────────────────┘
 ```

The transformed model has the same interface, so we can keep applying more lefts transformations to it to build up increasingly complex behaviour.

### The Split Command

Let's illustrate this with the simple command 'split':

```python
train_test_split = split(
    name='split_linear_regression',
    model=model,
    train_filter = pl.col('date') < dt.date(2026, 1, 1),
    test_filter = pl.col('dat') >= dt.date(2026, 1, 1)
)
```

The resulting model will only fit on data before 2026-01-01, and only generate predictions on data from that date onwards. 

### Invariants

Before describing the remaining commands, we should note that lefts is built around specific invariants




