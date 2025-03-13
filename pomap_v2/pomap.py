import polars as pl
from typing import Self
from itertools import product

from types import MethodType


class PoMap:

    def __init__(self, column, labels):
        self.column = column
        self.labels = labels

    def label_rows_as_train(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        raise NotImplementedError

    def label_rows_as_test(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        raise NotImplementedError

    @staticmethod
    def train_column_name(label: dict, dims):
        label = {d: label[d] for d in dims}  # Sort label for consistency
        return f'train({label})'

    @staticmethod
    def test_column_name(label: dict, dims):
        label = {d: label[d] for d in dims}  # Sort label for consistency
        return f'test({label})'

    def label_to_train(self, df: pl.DataFrame, label: dict):
        df = self.label_rows_as_train(df, label)
        df = df.filter(self.train_column_name(label, self.dims))
        df = df.drop(self.train_column_name(label, dims=self.dims))

        return df

    def label_to_validate(self, df: pl.DataFrame, label: dict):
        df = self.label_rows_as_train(df, label)
        df = df.filter(self.train_column_name(label, self.dims))
        df = df.drop(self.train_column_name(label, dims=self.dims))

        return df