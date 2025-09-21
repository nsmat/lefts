import polars as pl
from typing import Self, Literal
from functools import reduce
from dataclasses import dataclass


@dataclass(frozen=True)
class LabelKey:
    pomap_name: str
    label: dict

    def __repr__(self):
        return f"{self.pomap_name}:{self.label}"


class _Pomap:
    __COMPOSITION_TYPES = Literal['leaf', 'product', 'sum']
    __LABEL_TYPES = Literal['train', 'test', 'validate']

    # A PoMap is defined by a 'dimension' and a set of labels belonging to that dimension
    def __init__(self,
                 children: list[Self],
                 name: str,
                 composition_type: __COMPOSITION_TYPES
                 ):

        self.name = name
        self._children = children
        self.composition_type = composition_type

        # Implement some standardised naming for the subclasses to use
        self._train_column_name = lambda label: f'train({label})'
        self._test_column_name = lambda label: f'test({label})'
        self._validate_column_name = lambda label: f'validate({label})'

    def product(self, other: "_Pomap", product_name=None) -> "_Pomap":
        product_name = product_name or f'{self.name} x {other.name}'
        return _Pomap(children=[self, other],
                      name=product_name,
                      composition_type='product'
                      )

    def sum(self, other: "_Pomap", sum_name=None) -> "_Pomap":
        sum_name = sum_name or f"{self.name} + {other.name}"
        return _Pomap(
            children=[self, other],
            name=sum_name,
            composition_type="sum"
        )

    @property
    def labels(self) -> pl.DataFrame:
        if self.composition_type == 'leaf':
            return self.labels

        elif self.composition_type == "product":
            # A product node should return the cross product of its children
            child_labels = [child.labels for child in self._children]
            return reduce(lambda left, right: left.join(right, how="cross"), child_labels)

        elif self.composition_type == "sum":
            child_labels = [child.labels for child in self._children]
            return pl.concat(child_labels, how='diagonal_relaxed')

        else:
            raise ValueError(f'Unknown composition type {self.composition_type} encountered')

    def label_rows_as_train(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self._label_rows_as(df, label, label_as='train')
        return df

    def label_rows_as_test(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self._label_rows_as(df, label, label_as='test')
        return df

    def label_rows_as_validate(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self._label_rows_as(df, label, label_as='validate')
        return df

    def _label_expr(self, label, label_as) -> pl.Expr:
        """
        Generates an expression which will evaluate to True if a row is belongs to
        period <label_rows_as> for <label>
        """

        if self.composition_type == 'leaf':

            leaf_label_method = {
                'train': self.label_rows_as_train,
                'test': self.label_rows_as_test,
                'validate': self.label_rows_as_validate
            }[label_as]

            return leaf_label_method(label, label_as)

        elif self.composition_type == 'product':
            return pl.all_horizontal([child._label_expr(label, label_as) for child in self._children])
        elif self.composition_type == 'sum':
            return pl.any_horizontal([child._label_expr(label, label_as) for child in self._children])
        else:
            raise ValueError(f'Unknown composition type {self.composition_type} encountered')

    def _label_rows_as(self, df: pl.DataFrame, label: dict, label_as: __LABEL_TYPES) -> pl.DataFrame:
        column_name_func = {'train': self._train_column_name,
                       'test': self._test_column_name,
                       'validate': self._validate_column_name,
                       }[label_as]
        column_name = column_name_func(label)

        expr = self._label_expr(label, label_as)
        return df.with_columns(expr.alias(column_name))

    def _label_to(self, df: pl.DataFrame, label: dict, label_to: str) -> pl.DataFrame:
        funcs = {
            'train': (self.label_rows_as_train, self._train_column_name),
            'test': (self.label_rows_as_test, self._test_column_name),
            'validate': (self.label_rows_as_validate, self._validate_column_name),
        }

        label_func, column_name_func = funcs[label_to]
        df = label_func(df, label)
        df = df.filter(column_name_func(label)).drop(column_name_func(label))

        return df

    #### Model Interface
    def label_to_train(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self._label_to(df, label, 'train')
        return df

    def label_to_test(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self._label_to(df, label, 'train')
        return df

    def label_to_validate(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self._label_to(df, label, 'validate')
        return df


class Pomap(_Pomap):

    def __init__(self, name: str):
        super().__init__(children=[self], name=name, composition_type='leaf')

    @property
    def labels(self) -> pl.DataFrame:
        raise NotImplementedError

    # These three (train, test, validate) functions define the behaviour of the PoMap.
    # E.g, is it a cross validation, is it categorical, etc.
    # See below for an example of a reasonably complex example.
    def label_rows_as_train(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        raise NotImplementedError

    # There has to be a separate one for test and validation, because
    # train and test data must be distinct.
    def label_rows_as_test(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        raise NotImplementedError

    # .... as above
    def label_rows_as_validate(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        raise NotImplementedError
