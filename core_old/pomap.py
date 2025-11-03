import polars as pl
from typing import Self, Literal
from pomap.core.label import Label
from typing import Iterable
import itertools


class _Pomap:
    __COMPOSITION_TYPES = Literal["leaf", "product", "sum"]
    __LABEL_TYPES = Literal["train", "test", "validate"]

    # A PoMap is defined by a 'dimension' and a set of labels belonging to that dimension
    def __init__(
        self, children: list[Self], name: str, composition_type: __COMPOSITION_TYPES
    ):

        self.name = name
        self._children = children
        self.composition_type = composition_type

        # Implement some standardised naming for the subclasses to use
        self._train_column_name = lambda label: f"train({label})"
        self._test_column_name = lambda label: f"test({label})"
        self._validate_column_name = lambda label: f"validate({label})"

    def product(self, other: "_Pomap", product_name=None) -> "_Pomap":
        product_name = product_name or f"{self.name} x {other.name}"
        return _Pomap(
            children=[self, other], name=product_name, composition_type="product"
        )

    def sum(self, other: "_Pomap", sum_name=None) -> "_Pomap":
        sum_name = sum_name or f"{self.name} + {other.name}"
        return _Pomap(children=[self, other], name=sum_name, composition_type="sum")

    def view_labels(self) -> pl.DataFrame:
        labels = self._collect_labels()
        label_df = pl.from_dicts([label.as_dict() for label in labels])
        return label_df.select(*sorted(label_df.columns))

    def _collect_labels(self) -> list["Label"]:
        """
        Recursively collect all Labels for this PoMap composition.
        Returns a list of Label objects (immutable, hashable).
        """

        if self.composition_type == "leaf":
            return [Label({self.name: val}) for val in self._leaf_labels]

        # Resulting labels are cartesian product of children
        elif self.composition_type == "product":
            child_labels_lists = [child._collect_labels() for child in self._children]

            result = []
            for combination in itertools.product(*child_labels_lists):
                merged_mapping = {}
                for label in combination:
                    overlap = set(label.as_dict().keys()) & set(merged_mapping.keys())
                    if overlap:
                        raise ValueError(
                            f"Namespace collision in product: PoMap(s) {overlap} "
                            f"appear in multiple children"
                        )
                    merged_mapping.update(label.as_dict())
                result.append(Label(merged_mapping))

            return result

        # Resulting labels are disjoint union of children labels
        elif self.composition_type == "sum":
            result = []
            seen = set()
            for child in self._children:
                for label in child._collect_labels():
                    if label in seen:
                        raise ValueError(f"Label collision in sum composition: {label}")
                    seen.add(label)
                    result.append(label)

            return result

        else:
            raise ValueError(f"Unknown composition_type: {self.composition_type}")

    def label_rows_as_train(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self._label_rows_as(df, label, label_as="train")
        return df

    def label_rows_as_test(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self._label_rows_as(df, label, label_as="test")
        return df

    def label_rows_as_validate(self, df: pl.DataFrame, label: dict) -> pl.DataFrame:
        df = self._label_rows_as(df, label, label_as="validate")
        return df

    def _label_expr(self, df: pl.DataFrame, label, label_as: __LABEL_TYPES) -> pl.Expr:
        """
        Generates an expression which will evaluate to True if a row is belongs to
        period <label_rows_as> for <label>
        """

        if self.composition_type == "leaf":

            leaf_label_method = {
                "train": self.train_label_expr,
                "test": self.test_label_expr,
                "validate": self.validate_label_expr,
            }[label_as]

            return leaf_label_method(label[self.name], df)

        elif self.composition_type == "product":
            return pl.all_horizontal(
                [
                    child._label_expr(df=df, label=label, label_as=label_as)
                    for child in self._children
                ]
            )
        elif self.composition_type == "sum":
            return pl.any_horizontal(
                [
                    child._label_expr(df=df, label=label, label_as=label_as)
                    for child in self._children
                ]
            )
        else:
            raise ValueError(
                f"Unknown composition type {self.composition_type} encountered"
            )

    def _label_rows_as(
        self, df: pl.DataFrame, label: dict, label_as: __LABEL_TYPES
    ) -> pl.DataFrame:
        column_name_func = {
            "train": self._train_column_name,
            "test": self._test_column_name,
            "validate": self._validate_column_name,
        }[label_as]
        column_name = column_name_func(label)

        expr = self._label_expr(df=df, label=label, label_as=label_as)
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
    def __init__(self, name: str, labels: Iterable):
        super().__init__(children=[], name=name, composition_type="leaf")
        self._leaf_labels = labels

    @property
    def labels(self) -> list["Label"]:
        return [Label({self.name: v}) for v in self._leaf_labels]

    def train_label_expr(self, label, df: pl.DataFrame) -> pl.Expr:
        raise NotImplementedError

    def test_label_expr(self, label, df: pl.DataFrame) -> pl.Expr:
        raise NotImplementedError

    def validate_label_expr(self, label, df: pl.DataFrame) -> pl.Expr:
        raise NotImplementedError
