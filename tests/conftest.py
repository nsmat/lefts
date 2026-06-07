import pytest
from dataclasses import dataclass
import polars as pl

from pomap.nodes import Leaf, Lift, Ensemble


@dataclass
class MockModel:
    x_column: str
    value: float = None

    def fit(self, training_set: pl.DataFrame):
        self.value = training_set[self.x_column].mean()

    def predict(self, df: pl.DataFrame):
        return [self.value] * df.shape[0]


@dataclass
class ConsumerModel:
    source_col: str
    value: float = None

    def fit(self, training_set: pl.DataFrame):
        self.value = training_set[self.source_col].mean()

    def predict(self, df: pl.DataFrame):
        return [self.value] * len(df)


@pytest.fixture
def test_dataframe():
    df_a = pl.DataFrame({"x": [2.0, 3.0, 4.0], "category": ["a", "a", "a"]})
    df_b = pl.DataFrame({"x": [10.0, 15.0, 20.0], "category": ["b", "b", "b"]})
    df_c = pl.DataFrame({"x": [4.0, 6.0, 8.0], "category": ["c", "c", "c"]})
    return pl.concat([df_a, df_b, df_c]).with_columns(x2=-pl.col("x"))


@pytest.fixture
def leaf_factory():
    def _make_leaf(x_column: str):
        return Leaf(
            label=f"model-{x_column}", factory=lambda: MockModel(x_column=x_column)
        )

    return _make_leaf


@pytest.fixture
def model_x(leaf_factory):
    return leaf_factory("x")


@pytest.fixture
def model_x2(leaf_factory):
    return leaf_factory("x2")


@pytest.fixture
def lift_x(model_x):
    return Lift(
        name="category",
        child=model_x,
        values=["a", "b", "c"],
        train_filter=lambda v: pl.col("category") == pl.lit(v),
        test_filter=lambda v: pl.col("category") == pl.lit(v),
    )


@pytest.fixture
def ensemble_x1_x2(model_x, model_x2):
    return Ensemble(name="ensemble", models=[model_x, model_x2])
