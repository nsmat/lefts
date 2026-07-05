# Design philosophy of lefts

This page is not necessary to use lefts, but is here for people that want to understand how it was designed and what is going on in the background.

A Model can be thought of as a set of five functions:

- **fit**: creates estimator instances and optimises their parameters over the train set.
- **factory**: called by fit to instantiate all the estimators required by the workflow.
- **predict**: generates predictions on the test rows.
- **train_filter / test_filter / validation_filter**: Polars expressions which tell you whether each row of a dataframe is in the train/test/validation set.
- **labels**: which gives you unique labels for every estimator instance we create.

The Model output by 'leaf' will look a lot like the original estimator. Calling model.fit and model.predict will give you the same result as if you just used LinearRegression. The filters will mark every row of the dataframe as being in the train, validation, or test sets. The only label is 'linear_regression'.

However, as we apply further lefts commands, we will build up complex Models with heavily modified variants of the five functions. Each command is a functor that modifies those functions to yield a new, transformed model.

For example, if we have a Lefts command T, we can create a new model by using it to transform each function.

```
┌─ Model ─────────────┐              ┌─ Model ────────────────┐
│ .fit:     fitter    │  ────T────►  │ .fit:     T(fitter)    │
│ .predict: predictor │              │ .predict: T(predictor) │
└─────────────────────┘              └────────────────────────┘
```

The transformed model has the same interface, so we can keep applying more lefts transformations to it to build up increasingly complex behaviour.
