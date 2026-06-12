# Lefts: composable machine-learning model transformations

Lefts is a very simple domain specific language for building complex machine learning workflows from simple ones. Starting with your favourite machine learning models, you can use Lefts operations to:
- Build complex ensembles.
- Build complex cross validation and hyper-parametrisation procedures.
- Allow a model to create features or targets for another model.
- And any creative combination of the above.

Without making subsequent model evaluation, storage, or experimentation any more complex than it was with the original model. This implementation is built on top of the excellent Polars DataFrame library.


# Commands
Lefts has five commands, which give it it's name:
- **L**ift: trains multiple copies of a model across different subsets of data.
- **E**nsemble: Takes a set of models and makes them evaluate as one.
- **T**une: Allows a model to learn its hyperparameters from another.
- **F**eeds: Allows the output of one model to be used as a feature or target by another.
- **S**plit: Trains a model on a given train/test/validation split.

# Models

Lefts can operate on any model that is defined by:
- a fit method, which maps from training data into the model parameters
- a predict method, which maps from model parameters and test data into predictions.

A Lefts command creates a new model by transforming these functions into a new .fit and .predict. Because this new model also has a .fit and .predict, it can be transformed with further Lefts commands.


### Conventions

Lefts imposes some constraints on model interfaces.
- All hyperparameters are passed as arguments to the fit method.
- We expect that data is passed to fit and predict as Polars dataframes.
- The predict method returns an iterable, with the order of predictions matching the order on the input training data frame.

See the example notebooks to understand how to adapt your models to the required format.

# A quick example - a rolling retrain of 10 quantile target lightgbms

See notebooks/quantile_ensemble.py for the full code.

```python
features = ["temp", "atemp", "hum", "windspeed", "hr", "weekday", "mnth"]
target = "cnt"
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
test_period_start_dates = pl.datetime_range(
    start = dt.datetime(2011, 3, 1),
    end = dt.datetime(2012, 12, 1),
    interval='1mo',
    eager=True
).to_list()

quantile_models = []
for q in quantiles:
    # Convert LGBMRegressor into the format required by lefts
    base_model = leaf(tabular_model(
                LGBMRegressor(objective="quantile", alpha=q),
                features=features,
                target=target,
            ),
            label=f"q{q}",
                     )

    # 'Lift' each per-quantile model into a family of models, each with a different train and test period
    rolling_retrain = lift(
        base_model,
        name=f"q{q}_rolling_retrain",
        values=test_period_start_dates,

        # A row is in a given train period if it is 
        train_filter=lambda test_period_start_date: pl.col("datetime") < test_period_start_date,
        test_filter=lambda test_period_start_date: pl.col("datetime").dt.month() == test_period_start_date.month,
        aggregate_with=pl.coalesce,
    )

    quantile_models.append(rolling_retrain)

model = ensemble("quantiles", *quantile_models)

# Fits |quantiles| x |test_period_start_dates| models
model.fit(df)

# Adds |quantiles| columns, each with the unique prediction associated with that test row. 
predictions = model.predict(df)
```
