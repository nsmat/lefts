from pomap import PoMap
import polars as pl
import numpy as np


class RandomisedCrossValidationPoMap(PoMap):

    def __init__(self, dims, folds: int, index_column: 'str'):
        super().__init__(dims)
        self.folds = folds
        self.index_column = index_column
        self.fold_labels = [str(c) for c in range(folds)]

    def label_rows_as_test(self, df: pl.DataFrame, label: dict):
        assert df.unique(self.index_column).shape[0] == df.shape[0]
        index_to_test_fold = {}
        for index in df.select(self.index_column).unique():
            index_to_test_fold[index] = np.random.choice(self.fold_labels)

        df = df.with_columns(
            pl.col(self.index_column)
            .replace_strict(index_to_test_fold)
            .alias(self.train_column_name(label, self.dims))
        )

        return df

    def label_rows_as_train(self, df, label):
        df = self.label_rows_as_test(df, label)
        test_column = self.test_column_name(label, self.dims)
        df = df.with_columns(
            ~(pl.col(test_column))
            .alias(self.train_column_name(label, self.dims))
        )
        df = df.drop(test_column)

        return df
