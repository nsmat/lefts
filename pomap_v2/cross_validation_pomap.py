from pomap_v2.pomap import Pomap
import polars as pl
from random import choice

class RandomisedCrossValidationPoMap(Pomap):

    def __init__(self, num_folds: int, index_column: 'str'):
        super().__init__(reference_column=index_column, name=f'Randomised CV: {index_column}')
        self.num_folds = num_folds
        self.index_column = index_column
        self.fold_labels = [str(c) for c in range(num_folds)]
        self._test_label_mapping = {}

    def index_to_label(self, label):
        if label not in self._test_label_mapping:
            self._test_label_mapping[label] = choice(self.fold_labels)
        return self._test_label_mapping[label]

    def label_rows_as_test(self, df: pl.DataFrame, label: dict):
        assert df.unique(self.index_column).shape[0] == df.shape[0]

        # Every row is randomly assigned to one of the possible folds
        all_indexes = df.select(self.index_column).unique()
        index_to_label_dict = {index: self.index_to_label(index) for index in all_indexes}

        df = df.with_columns(
            pl.col(self.index_column)
            .replace_strict(index_to_label_dict)
            .alias(self._train_column_name(label))
        )

        return df

    def label_rows_as_train(self, df, label):
        df = self.label_rows_as_test(df, label)
        test_column = self._test_column_name(label)

        df = df.with_columns(
            ~(pl.col(test_column))
            .alias(self._train_column_name(label))
        )
        df = df.drop(test_column)

        return df
