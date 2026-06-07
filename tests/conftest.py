import pytest
from dataclasses import dataclass
import polars as pl

from pomap.nodes import Leaf, Lift


@dataclass
class MockModel:
    """
    A passthrough model whose prediction is all the training data in
    'x_column' packed into a list. This makes it easy to understand
    which data is being seen as 'training data' by model.
    """

    x_column: str
    seen: list = None

    def fit(self, training_set: pl.DataFrame):
        self.seen = training_set[self.x_column].to_list()

    def predict(self, df: pl.DataFrame):
        return [self.seen] * df.shape[0]


@dataclass
class ConsumerModel:
    """
    For use in Feed based setups - passes the exact values
    through from the teacher so we retain maximum visibility
    """

    source_col: str
    seen: list = None

    def fit(self, training_set: pl.DataFrame):
        self.seen = training_set[self.source_col].to_list()

    def predict(self, df: pl.DataFrame):
        return df[self.source_col].to_list()


@pytest.fixture
def test_dataframe():
    return pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "category": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "fold": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        }
    )


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
def lift_x(model_x):
    return Lift(
        name="category",
        child=model_x,
        values=["a", "b", "c"],
        train_filter=lambda v: pl.col("category") == pl.lit(v),
        test_filter=lambda v: pl.col("category") == pl.lit(v),
    )
