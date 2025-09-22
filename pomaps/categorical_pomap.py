import polars as pl
from pomap.core.pomap import Pomap


class CategoricalPomap(Pomap):

    def __init__(self, column: str, labels: list):
        super().__init__(name=f"Categorical {column}: {labels}")
        self._column = column
        self._labels = labels

    @property
    def labels(self) -> pl.DataFrame:
        return pl.Series(values=self._labels, name=self._column).to_frame()

    def train_label_expr(self, label, df: pl.DataFrame) -> pl.Expr:
        return pl.col(self._column) == label

    def test_label_expr(self, label, df: pl.DataFrame,) -> pl.Expr:
        return pl.col(self._column) == label

    def validate_label_expr(self, label, df: pl.DataFrame) -> pl.Expr:
        return pl.col(self._column) == label
