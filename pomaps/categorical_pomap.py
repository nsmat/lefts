import polars as pl
from pomap.core.pomap import Pomap


class CategoricalPomap(Pomap):

    def __init__(self, column: str, labels: list):
        super().__init__(name=f"Categorical {column}: {labels}", labels=labels)
        self._column = column

    def train_label_expr(self, label, df: pl.DataFrame) -> pl.Expr:
        return pl.col(self._column) == label

    def test_label_expr(self, label, df: pl.DataFrame,) -> pl.Expr:
        return pl.col(self._column) == label

    def validate_label_expr(self, label, df: pl.DataFrame) -> pl.Expr:
        return pl.col(self._column) == label
