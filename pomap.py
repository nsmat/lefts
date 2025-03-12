import polars as pl
from typing import Self
from itertools import product

from types import MethodType

# TODO right now we initialise via dims, but
# The 'constant' should be _dims_to_labels, since
# This should allow us to forget and add entire dimensions easily

class PoMap:

    def __init__(self, dims_to_labels: dict, *args, **kwargs):
        self._dims_to_labels = dims_to_labels

    @property
    def dims(self):
        return self._dims_to_labels.keys()

    @property
    def labels(self):
        # TODO implement consistent labelling here?
        # Can be a list of dict, also provide separate dataframe method
        raise NotImplementedError
    
    @property
    def labels_df(self):
        # TODO should fill this in as a polars readable df of labels
        raise NotImplementedError

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

    @staticmethod
    def validation_column_name(label: dict, dims):
        label = {d: label[d] for d in dims}  # Sort label for consistency
        return f'validation_column_name({label})'

    @staticmethod
    def _compose_label_functions(pomap_1, pomap_2, type='train'):
        if type == 'train':
            column_name_func = PoMap.train_column_name
            label_func = 'label_rows_as_train'
        elif type == 'test':
            column_name_func = PoMap.test_column_name
            label_func = 'label_rows_as_test'
        else:
            raise ValueError(f"Unknown type {type}")

        def composed(self, df, label):
            label_1 = {d: label[d] for d in pomap_1.dims}
            label_2 = {d: label[d] for d in pomap_2.dims}

            df = getattr(pomap_1, label_func)(df, label_1)
            df = getattr(pomap_2, label_func)(df, label_2)

            output_train_label = column_name_func(label,
                                                  pomap_1.dims + pomap_2.dims,
                                                  )
            df = df.with_columns(
                (
                pl.col(column_name_func(label_1, dims=pomap_1.dims)) &
                pl.col(column_name_func(label_2, dims=pomap_2.dims))
                )
                .alias(output_train_label)
            )

            df = df.drop(
                column_name_func(label_1, dims=pomap_1.dims),
                column_name_func(label_2, dims=pomap_2.dims)
            )

            return df

        return composed

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

    @staticmethod
    def composed_labels(pomap_1, pomap_2):
        def labels(self):
            composed_labels = []
            for label1 in pomap_1.labels:
                for label2 in pomap_2.labels:
                    composed_labels.append({**label1, **label2})

            return composed_labels
        return labels


    @staticmethod
    def product(pomap_1, pomap_2):
        product_map = PoMap(dims_to_labels={**pomap_1._dims_to_labels,
                                            **pomap_2._dims_to_labels
                                            })

        # Compose Functions and Patch them into the composed PoMap
        # Using MethodType is necessary to respect the special role played by 'self'
        label_rows_as_train = PoMap._compose_label_functions(pomap_1, pomap_2, type='train')
        product_map.label_rows_as_train = MethodType(label_rows_as_train, product_map)

        label_rows_as_test = PoMap._compose_label_functions(pomap_1, pomap_2, type='test')
        product_map.label_rows_as_test = MethodType(label_rows_as_test, product_map)

        labels = PoMap.composed_labels(pomap_1, pomap_2)
        product_map.labels = MethodType(labels, product_map)

        return product_map
