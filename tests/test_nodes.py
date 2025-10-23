import pytest
from dataclasses import dataclass
import polars as pl

from core.label import Label
from core.nodes import Lift, Leaf, Ensemble
from core.interpreter import _collect_labels, _collect_leaves, _fit, _predict


@pytest.fixture
def test_dataframe():
    df_a = pl.DataFrame(
        {
            'x': [2., 3., 4.],
            'category': ['a', 'a', 'a']
        }
    )

    df_b = pl.DataFrame(
        {
            'x': [1., 1.5, 2.],
            'category': ['b', 'b', 'b']
        }
    )

    df_c = pl.DataFrame(
        {
            'x': [4., 6., 8.],
            'category': ['c', 'c', 'c']
        }
    )

    df = pl.concat([df_a, df_b, df_c])
    df = df.with_columns(-pl.col('x'))

    return df


@dataclass
class TestModel:
    x_column: str
    value = None

    def fit(self, df: pl.DataFrame):
        self.value = df[self.x_column].mean()

    def predict(self, df: pl.DataFrame):
        return [self.value] * df.shape[0]


@pytest.fixture
def leaf_factory():
    """Fixture returning a helper that can make Leaf objects for any column."""
    def _make_leaf(x_column: str):
        return Leaf(
            label=f"model-{x_column}",
            factory=lambda: TestModel(x_column=x_column)
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
        model_x,
        atomics=["a", "b", "c"],
        train_mask_for_label=lambda atomic: pl.col("category") == pl.lit(atomic),
        test_mask_for_label=lambda atomic: pl.col("category") == pl.lit(atomic),
    )


@pytest.fixture
def ensemble_x1_x2(model_x, model_x2):
    return Ensemble([model_x, model_x2])

def test_labels_x(lift_x):
    expected_labels_x1 = {
        Label(leaf='model-x', category='a'),
        Label(leaf='model-x', category='b'),
        Label(leaf='model-x', category='c')
    }


    assert set(_collect_labels(lift_x)) == expected_labels_x1
