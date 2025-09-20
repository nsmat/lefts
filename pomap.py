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
    __COMPOSITION_TYPES = Literal["leaf", "product", "sum"]

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
            raise ValueError('Unknown composition type encountered')

    def label_rows_as_train(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self._label_rows_as(df, label, label_as='train')
        return df

    def label_rows_as_test(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self._label_rows_as(df, label, label_as='test')
        return df

    def label_rows_as_validate(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self._label_rows_as(df, label, label_as='validate')
        return df

    def _label_rows_as(self, df: pl.DataFrame, label: dict, label_as: str) -> pl.DataFrame:

        column_name_method = {
            'train': self._train_column_name,
            'test': self._test_column_name,
            'validate': self._validate_column_name
        }[label_as]

        node_columns = []
        for node in self._children:
            label_as_method_map = {
                'train': node.label_rows_as_train,
                'test': node.label_rows_as_test,
                'validate': node.label_rows_as_validate
            }

            label_as_method = label_as_method_map[label_as]

            df = label_as_method(df=df, label=label)
            node_columns.append(column_name_method(label))

        # We evaluate the match to a label differently depending on the
        # composition type of the root node.
        if self.composition_type == "product":
            agg = pl.col('__per_node_results').list.all()
        elif self.composition_type == "sum":
            agg = pl.col('__per_node_results').list.any()
        else:  # leaf
            agg = pl.col(node_columns[0])


        df = df.with_columns(__per_node_results=pl.concat_list(node_columns))
        df = df.with_columns(agg.alias(column_name_method(label)))
        df = df.drop('__per_node_results', *node_columns)

        return df

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
