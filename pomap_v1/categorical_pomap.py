from pomap import PoMap
import polars as pl
from itertools import product


class CategoricalPoMap(PoMap):

    def __init__(self, dims_to_labels: dict):
        super().__init__(dims_to_labels)
        self._dims_to_labels = dims_to_labels

    @property
    def labels(self):
        # Get every possible combination of values
        combinations = product(*self._dims_to_labels.values())

        labels = []
        for values in combinations:
            # Associate each element in the product space with the appropriate key
            label = dict(zip(self._dims_to_labels.keys(), values))
            labels.append(label)

        return labels

    def _match_dims_expression(self, label):
        individual_matches = [(pl.col(d) == label[d]) for d in self.dims]
        expression = pl.all_horizontal(*individual_matches)
        return expression

    def label_rows_as_train(self, df: pl.DataFrame, label):
        df = df.with_columns(
            self._match_dims_expression(label).alias(self.train_column_name(label, self.dims))
        )

        return df

    def label_rows_as_test(self, df: pl.DataFrame, label):
        df = df.with_columns(
            self._match_dims_expression(label).alias(self.test_column_name(label, self.dims))
        )

        return df
