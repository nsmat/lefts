import polars as pl

from pomap.interface import leaf, lift, split, ensemble, feed, learn_from


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
        train_filter=pl.col("x") < 5,
        test_filter=pl.col("x") >= 5,
    )
    marked = model.mark_train_validation_test_rows(test_dataframe)
    train = marked.filter(pl.col("m__train"))
    test = marked.filter(pl.col("m__test"))
    assert (train["x"] < 5).all()
    assert (test["x"] >= 5).all()


def test_masks_split_with_validation(test_dataframe):
    model = split(
        "tt",
        _make_leaf("m"),
        train_filter=pl.col("x") < 5,
        test_filter=pl.col("x") >= 8,
        validation_filter=pl.col("x").is_between(5, 7, closed="both"),
    )
    marked = model.mark_train_validation_test_rows(test_dataframe)
    assert "m__validation" in marked.columns
    val = marked.filter(pl.col("m__validation"))
    assert set(val["x"].to_list()) == {5, 6, 7}


# ── Ensemble ──────────────────────────────────────────────────────


def test_masks_ensemble_passthrough(test_dataframe):
    # Ensemble has no row filter of its own; both children should inherit the
    # outer Split's mask unchanged.
    a = _make_leaf("model-a")
    b = _make_leaf("model-b")
    model = split(
        "tt",
        ensemble("ens", a, b),
        train_filter=pl.col("x") < 5,
        test_filter=pl.col("x") >= 5,
    )
    marked = model.mark_train_validation_test_rows(test_dataframe)
    for label in ("model-a", "model-b"):
        train = marked.filter(pl.col(f"{label}__train"))
        test = marked.filter(pl.col(f"{label}__test"))
        assert (train["x"] < 5).all()
        assert (test["x"] >= 5).all()


# ── Feed ──────────────────────────────────────────────────────────


def test_masks_feed_passthrough(test_dataframe):
    # Feed has no row filter of its own; source and consumer leaves should both
    # inherit the outer Split's mask.
    model = split(
        "tt",
        feed("d", source=_make_leaf("src"), consumer=_make_leaf("cons")),
        train_filter=pl.col("x") < 5,
        test_filter=pl.col("x") >= 5,
    )
    marked = model.mark_train_validation_test_rows(test_dataframe)
    for label in ("src", "cons"):
        train = marked.filter(pl.col(f"{label}__train"))
        test = marked.filter(pl.col(f"{label}__test"))
        assert (train["x"] < 5).all()
        assert (test["x"] >= 5).all()


# ── LearnsFrom ────────────────────────────────────────────────────


def test_masks_learns_from_passthrough(test_dataframe):
    # LearnsFrom has no row filter of its own; learns_from and learner subtrees
    # should both inherit the outer Split's mask.
    model = split(
        "tt",
        learn_from(
            "lf",
            learner=_make_leaf("learner"),
            learns_from=_make_leaf("source"),
            logic=lambda m, df: {},
        ),
        train_filter=pl.col("x") < 5,
        test_filter=pl.col("x") >= 5,
    )
    marked = model.mark_train_validation_test_rows(test_dataframe)
    for label in ("source", "learner"):
        train = marked.filter(pl.col(f"{label}__train"))
        test = marked.filter(pl.col(f"{label}__test"))
        assert (train["x"] < 5).all()
        assert (test["x"] >= 5).all()
