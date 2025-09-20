import polars as pl
from pomap.pomap import  Pomap


class CategoricalPomap(Pomap):

    def __init__(self, column: str, labels: list):
        super().__init__(name=f"Categorical {column}: {labels}", reference_column=column)
        self._column = column
        self._labels = labels

    @property
    def labels(self) -> pl.DataFrame:
        return pl.Series(values=self._labels, name=self.reference_column).to_frame()

    def _label_rows_as(self, df: pl.DataFrame, label: dict, column_name: str) -> pl.DataFrame:
        df = df.with_columns(
            (pl.col(self._column) == label[self._column]).alias(column_name)
        )

        return df

    def label_rows_as_train(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        return self._label_rows_as(df, label, self._train_column_name(label))

    def label_rows_as_test(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        return self._label_rows_as(df, label, self._test_column_name(label))

    def label_rows_as_validate(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        return self._label_rows_as(df, label, self._validate_column_name(label))
