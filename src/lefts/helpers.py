import polars as pl
from typing import Any, Callable


def tabular_model(
    estimator_class,
    features: list[str],
    target: str,
) -> Callable[..., Any]:
    """
    Wraps a sklearn-compatible estimator constructor into a Lefts model factory.

    The returned factory can be passed directly to `leaf(model_constructor=...)`.

    Parameters
    ----------
    estimator_class:
        A callable that constructs a sklearn-compatible estimator (must implement .fit(X, y) and .predict(X)).
        Use ``functools.partial`` to pre-bind constructor arguments.
    features:
        Column names to use as model inputs.
    target:
        Column name of the target variable.
    """

    def factory(**hyperparameters):
        est = estimator_class(**hyperparameters)

        class _Model:
            def fit(self, training_set: pl.DataFrame):
                X = training_set.select(features).to_numpy()
                y = training_set[target].to_numpy()
                est.fit(X, y)

            def predict(self, df: pl.DataFrame):
                return est.predict(df.select(features).to_numpy())

        return _Model()

    return factory
