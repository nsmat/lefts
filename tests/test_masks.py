import polars as pl

from pomap.interface import leaf, lift, split


def _make_leaf(label):
    return leaf(lambda: None, label)


# ── Leaf ──────────────────────────────────────────────────────────


def test_masks_leaf_defaults_to_all_true(test_dataframe):
    marked = _make_leaf("m").mark_train_validation_test_rows(test_dataframe)
    assert {"m__train", "m__test"} <= set(marked.columns)
    assert "m__validation" not in marked.columns
    assert marked["m__train"].all()
    assert marked["m__test"].all()


# ── Lift ──────────────────────────────────────────────────────────


def test_masks_lift_per_value_filter(test_dataframe):
    model = lift(
        _make_leaf("m"),
        values=["a"],
        name="cat",
        train_filter=lambda v: pl.col("category") == pl.lit(v),
        test_filter=lambda v: pl.col("category") != pl.lit(v),
    )
    marked = model.mark_train_validation_test_rows(test_dataframe)
    train = marked.filter(pl.col("m[cat=a]__train"))
    test = marked.filter(pl.col("m[cat=a]__test"))
    assert set(train["category"].unique().to_list()) == {"a"}
    assert set(test["category"].unique().to_list()) == {"b", "c"}


# ── Split ─────────────────────────────────────────────────────────


def test_masks_split_filter(test_dataframe):
    model = split(
        "tt",
        _make_leaf("m"),
        train_filter=pl.col("x") < 10,
        test_filter=pl.col("x") >= 10,
    )
    marked = model.mark_train_validation_test_rows(test_dataframe)
    train = marked.filter(pl.col("m__train"))
    test = marked.filter(pl.col("m__test"))
    assert (train["x"] < 10).all()
    assert (test["x"] >= 10).all()


def test_masks_split_with_validation(test_dataframe):
    model = split(
        "tt",
        _make_leaf("m"),
        train_filter=pl.col("x") < 5,
        test_filter=pl.col("x") >= 10,
        validation_filter=pl.col("x").is_in([6, 8]),
    )
    marked = model.mark_train_validation_test_rows(test_dataframe)
    assert "m__validation" in marked.columns
    val = marked.filter(pl.col("m__validation"))
    assert set(val["x"].to_list()) == {6, 8}
