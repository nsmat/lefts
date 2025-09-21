import polars as pl
from typing import Self, Optional
from functools import reduce


class _Pomap:

    # A PoMap is defined by a 'dimension' and a set of labels belonging to that dimension
    def __init__(self, nodes: list[Self], name: str, reference_column: Optional[str]):

        self.name = name
        self._nodes = nodes
        self.reference_column = reference_column

        # Implement some standardised naming for the subclasses to use
        self._train_column_name = lambda label: f'train({label})'
        self._test_column_name = lambda label: f'test({label})'
        self._validate_column_name = lambda label: f'validate({label})'

    def __repr__(self):
        return self.name

    def __getitem__(self, arg: str):
        for node in self._nodes:
            if node.name == 'arg':
                return node
        raise ValueError(f'PoMap has no node {arg}')

    # -=-=-=-=-=-=-=--=-=-==-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=--=-
    #   After this, things get interesting, since we start to deal with how the pomap actually behaves
    def product(self, other: "_Pomap", product_name=None) -> "_Pomap":

        # Reference columns or names?
        self_reference_columns = {n.reference_column for n in self._nodes}
        other_reference_columns = {n.reference_column for n in other._nodes}

        overlapping_reference_columns = self_reference_columns.intersection(other_reference_columns)
        assert overlapping_reference_columns == set(), f"Cannot compose two Pomaps with overlapping reference_columns. Found {overlapping_reference_columns} in common"

        # This composition assumes that ONLY the product operation is possible, not the sum.
        product_name = product_name if product_name else f'{self.name} x {other.name}'
        return _Pomap(nodes=self._nodes + other._nodes,
                      name=product_name,
                      reference_column=None
                      )

    @property
    def labels(self) -> pl.DataFrame:
        # TODO this is currently overkill, because our 'tree' is just a path.
        # However, it will be necessary if we add a product operation
        leaf_nodes = self._find_leaf_nodes(self)
        leaf_labels = [node.labels for node in leaf_nodes]

        df = reduce(lambda left, right: left.join(right, how='cross'), leaf_labels)

        return df

    @staticmethod
    def _find_leaf_nodes(node):
        if (len(node._children) == 1) and (node._children[0] is node):
            return [node]

        leaf_nodes = []
        for child in node._children:
            leaf_nodes.extend(_Pomap._find_leaf_nodes(child))

        return leaf_nodes

    # Need to implement these in terms of the composed logic
    def label_rows_as_train(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self._label_rows_as(df, label, label_as='train')
        return df

    def label_rows_as_test(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self._label_rows_as(df, label, label_as='test')
        return df

    def label_rows_as_validate(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self._label_rows_as(df, label, label_as='validate')
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

    def __init__(self, name: str, reference_column: str):
        super().__init__(nodes=[self], name=name, reference_column=reference_column)

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
