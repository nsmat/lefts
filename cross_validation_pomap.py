from pomap.pomap import Pomap
import polars as pl
from random import choice

class RandomisedCrossValidationPoMap(Pomap):

    def __init__(self, num_folds: int, index_column: 'str'):
        super().__init__(name=f'Randomised CV: {index_column}')
        self.num_folds = num_folds
        self.index_column = index_column
        self.fold_labels = [str(c) for c in range(num_folds)]
        self._test_label_mapping = {}

    def labels(self) -> pl.DataFrame:
        return pl.Series(values=range(self.num_folds)).to_frame()

    def index_to_label(self, label):
        if label not in self._test_label_mapping:
            self._test_label_mapping[label] = choice(self.fold_labels)
        return self._test_label_mapping[label]

    def train_label_expr(self, df: pl.DataFrame, label):
        assert df.unique(self.index_column).shape[0] == df.shape[0]

        # Every row is randomly assigned to one of the possible folds
        all_indexes = df.select(self.index_column).unique()
        index_to_label_dict = {index: self.index_to_label(index) for index in all_indexes}

        expr = pl.col(self.index_column).replace_strict(index_to_label_dict) == label

        return expr




