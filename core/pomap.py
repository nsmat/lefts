import polars as pl
from typing import Self, Literal
from pomap.core.label import Label
from typing import List
import itertools


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

    def _collect_labels(self) -> List[Label]:
        """
        Recursively build the list of Label objects that index the composed Pomap.
        - leaf: return Labels with single entry { self.name: row_dict }
        - product: cartesian product of child labels, merged
        - sum: union (concatenate) of child labels
        """
        # Leaf case: turn each row of self.labels (a DataFrame) into a Label
        if self.composition_type == "leaf":
            df = self.labels  # leaf class must implement .labels DataFrame
            labels = []
            for row in df.iter_rows(named=True):
                labels.append(Label.from_dict({self.name: row}))
            return labels

        # Product: cartesian product of children
        if self.composition_type == "product":
            child_lists = [child._collect_labels() for child in self._children]

            result = []
            for combo in itertools.product(*child_lists):
                merged = combo[0]
                for lbl in combo[1:]:
                    merged = merged.merged_with(lbl)
                result.append(merged)
            return result

        # Sum: concatenate child label lists (no merging)
        if self.composition_type == "sum":
            result = []
            for child in self._children:
                result.extend(child._collect_labels())
            # optionally deduplicate
            # result = list(dict.fromkeys(result))  # preserves order (if Label is hashable)
            # better: keep unique by set to remove duplicates
            seen = {}
            unique = []
            for lbl in result:
                if lbl not in seen:
                    seen[lbl] = True
                    unique.append(lbl)
            return unique

        raise ValueError(f"Unknown composition type {self.composition_type}")

    def labels_list(self) -> List[Label]:
        """Public accessor returning the list of Labels for this pomap composition."""
        return self._collect_labels()

    def view_labels(self) -> pl.DataFrame:
        """
        Produce a Polars DataFrame view of the composed Labels.
        Column names are namespaced as '<pomap_name>__<field>' to avoid collisions.
        Missing fields are filled with None (Polars will produce nulls).
        """
        labels = self._collect_labels()
        rows = []
        columns_set = set()
        for lbl in labels:
            d = {}
            mapping = lbl.to_dict()
            for pomap_name, sub in mapping.items():
                for k, v in sub.items():
                    colname = f"{pomap_name}__{k}"
                    d[colname] = v
                    columns_set.add(colname)
            rows.append(d)

        if not rows:
            return pl.DataFrame([])

        cols = sorted(columns_set)
        normalized_rows = [{c: r.get(c, None) for c in cols} for r in rows]
        return pl.DataFrame(normalized_rows)

    def label_rows_as_train(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self._label_rows_as(df, label, label_as='train')
        return df

    def label_rows_as_test(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self._label_rows_as(df, label, label_as='test')
        return df

    def label_rows_as_validate(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self._label_rows_as(df, label, label_as='validate')
        return df

    def _label_expr(self, df: pl.DataFrame, label, label_as: __LABEL_TYPES) -> pl.Expr:
        """
        Generates an expression which will evaluate to True if a row is belongs to
        period <label_rows_as> for <label>
        """

        if self.composition_type == 'leaf':

            leaf_label_method = {
                'train': self.train_label_expr,
                'test': self.test_label_expr,
                'validate': self.validate_label_expr
            }[label_as]

            return leaf_label_method(label, df)

        elif self.composition_type == 'product':
            return pl.all_horizontal([child._label_expr(df, label, label_as) for child in self._children])
        elif self.composition_type == 'sum':
            return pl.any_horizontal([child._label_expr(df, label, label_as) for child in self._children])
        else:
            raise ValueError(f'Unknown composition type {self.composition_type} encountered')

    def _label_rows_as(self, df: pl.DataFrame, label: dict, label_as: __LABEL_TYPES) -> pl.DataFrame:
        column_name_func = {'train': self._train_column_name,
                            'test': self._test_column_name,
                            'validate': self._validate_column_name,
                            }[label_as]
        column_name = column_name_func(label)

        expr = self._label_expr(df, label, label_as)
        return df.with_columns(expr.alias(column_name))

    # # #  Interface used to slice data during model training
    # # # Each function takes a label, and filters down to
    # # # The subset of the data that matches the train/test/validate
    # # # Condition for that label.
    def label_to_train(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self.label_rows_as_train(df, label)
        col = self._train_column_name(label)
        return df.filter(col).drop(col)

    def label_to_test(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self.label_rows_as_test(df, label)
        col = self._test_column_name(label)
        return df.filter(col).drop(col)

    def label_to_validate(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self.label_rows_as_validate(df, label)
        col = self._validate_column_name(label)
        return df.filter(col).drop(col)


class Pomap(_Pomap):

    def __init__(self, name: str):
        super().__init__(children=[], name=name, composition_type='leaf')

    @property
    def labels(self) -> pl.DataFrame:
        raise NotImplementedError

    def train_label_expr(self, label, df: pl.DataFrame) -> pl.Expr:
        raise NotImplementedError

    def test_label_expr(self, label, df: pl.DataFrame) -> pl.Expr:
        raise NotImplementedError

    def validate_label_expr(self, label, df: pl.DataFrame) -> pl.Expr:
        raise NotImplementedError
