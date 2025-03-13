import polars as pl
from typing import Self
from functools import reduce

# TODO - allow better labels by adding a reference column instead
class _Pomap:

    # A PoMap is defined by a 'dimension' and a set of labels belonging to that dimension
    def __init__(self, nodes: list[Self], name: str):

        self.name = name
        self._nodes = nodes

        # Implement some standardised naming for the subclasses to use
        self._train_column_name = lambda label: f'train({self.name}={label})'
        self._test_column_name = lambda label: f'test({self.name}={label})'
        self._validate_column_name = lambda label: f'validate({self.name}={label})'

    def __repr__(self):
        return self.name

    def __getitem__(self, arg: str):
        for node in self._nodes:
            if node.name == 'arg':
                return node
        raise ValueError(f'Pomap has no node {arg}')

    # -=-=-=-=-=-=-=--=-=-==-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=--=-
    #   After this, things get interesting, since we start to deal with how the pomap actually behaves
    # COMPOSITION

    def product(self, other: "_Pomap") -> "_Pomap":
        # This composition assumes that ONLY the product operation is possible, not the sum.

        overlapping_names = set(self._nodes).intersection(other._nodes)
        assert overlapping_names == set(), f"Cannot compose two Pomaps with overlapping names. Found {overlapping_names} in common"

        return _Pomap(nodes=self._nodes + other._nodes, name=f'{self.name} + {other.name}')

    @property
    def labels(self) -> pl.DataFrame:
        # TODO this will have to change if we introduce a product operation
        # (or any other composition)
        # Since we will need to recurse through the syntax tree and
        # pluck out the labels of all the child nodes

        leaf_nodes = self._find_leaf_nodes(self)
        leaf_labels = [node.labels for node in leaf_nodes]

        df = reduce(lambda left, right: left.to_frame().join(right.to_frame(), how='cross'), leaf_labels)

        return df

    @staticmethod
    def _find_leaf_nodes(node):
        if (len(node._nodes) == 1) and (node._nodes[0] is node):
            return [node]

        leaf_nodes = []
        for child in node._nodes:
            leaf_nodes.extend(_Pomap._find_leaf_nodes(child))

        return leaf_nodes

    # Need to implement these in terms of the composed logic
    def label_rows_as_train(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:

        node_columns = []
        for node in self._nodes:
            node_sub_label = label[node.name]
            df = node.label_rows_as_train(df, node_sub_label)
            node_columns.append(node._train_column_name(node_sub_label))

        df = df.with_columns(node_trains=pl.concat_list(node_columns))
        df = df.with_columns(
            pl.col('node_trains').list.all()
            .alias(self._train_column_name(label))
        )
        df = df.drop('node_trains', *node_columns)

        return df

    def label_rows_as_test(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        pass

    def label_rows_as_validate(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        pass

    #### Model Interface
    def label_to_train(self, df: pl.DataFrame, label: dict):
        df = self.label_rows_as_train(df, label)
        df = df.filter(
            self.train_column_name(label)
        )
        df = df.drop(
            self.train_column_name(label)
        )

        return df



class Pomap(_Pomap):

    def __init__(self, name: str):
        super().__init__(nodes=[self], name=name)

    @property
    def labels(self, other: Self) -> pl.Series:
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