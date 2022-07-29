from __future__ import annotations

from typing import Literal, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn import preprocessing

import numpy as np
from sklearn.preprocessing import OrdinalEncoder

numeric_categorical_columns = [
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_63",
    "D_64",
    "D_66",
    "D_68",
]

string_categorical_column_prefix = "diff_cat"
no_diff_columns = ["Year", "Month", "Day", "date"]
kaggle_categorical_columns = numeric_categorical_columns + [
    string_categorical_column_prefix
]
categorical_columns_prefix = kaggle_categorical_columns + no_diff_columns


def get_categorical_columns(
    df: pl.DataFrame | pd.DataFrame, kind: Literal["both", "string", "numeric"] = "both"
) -> list[str]:
    if isinstance(df, pd.DataFrame):
        columns = df.columns.to_list()
    else:
        columns = df.columns

    if kind == "string":
        return [col for col in columns if col.startswith("diff_cat")]
    elif kind == "numeric":
        return [
            col
            for col in columns
            if any([col.startswith(start) for start in numeric_categorical_columns])
        ]
    else:  # both
        return [
            col
            for col in columns
            if any([col.startswith(start) for start in kaggle_categorical_columns])
        ]


def get_categorical_indices(
    df: pl.DataFrame | pd.DataFrame, kind: Literal["both", "string", "numeric"] = "both"
) -> list[int]:
    cols = get_categorical_columns(df, kind=kind)

    if isinstance(df, pd.DataFrame):
        columns: list[str] = df.columns.to_list()
    else:
        columns = df.columns

    indices = [columns.index(col) for col in cols]
    return indices


def get_categorical_xgboost(df: pl.DataFrame | pd.DataFrame):
    categorical_columns = get_categorical_columns(df)
    if isinstance(df, pd.DataFrame):
        columns: list[str] = df.columns.to_list()
    else:
        columns = df.columns
    return ["c" if col in categorical_columns else "q" for col in columns]


def load_old(kind: str = "train", lazy: bool = True) -> pl.DataFrame | pl.LazyFrame:
    path = f"artifacts/{kind}.parquet"
    if kind == "target":
        return pl.read_csv("artifacts/train_labels.csv")

    if lazy == True:
        return pl.scan_parquet(path)
    else:
        return pl.read_parquet(path)


# df_train = pl.read_parquet("data/train.parquet")
# df_target = pl.read_csv("data/train_labels.csv")

# df_test = pl.read_parquet("data/test.parquet")


def get_column_stats(df: pl.DataFrame | pl.LazyFrame):
    cat_col = pl.col(numeric_categorical_columns)
    num_col = pl.all().exclude(numeric_categorical_columns + no_diff_columns)

    return df.groupby(by="customer_ID").agg(
        [
            # descriptive stats of numerical columns
            num_col.min().prefix("min_"),
            num_col.max().prefix("max_"),
            num_col.std().prefix("std_"),
            num_col.var().prefix("var_"),
            num_col.sum().prefix("sum_"),
            # Only min and max of categorical columns
            cat_col.min().prefix("min_"),
            cat_col.max().prefix("max_"),
        ]
    )


def diff(col: pl.Expr) -> pl.Expr:
    return col.diff(n=1).cast(pl.Float32).prefix("diff_")


# def ratio(col: pl.Expr) -> pl.Expr:
#     col = col.cast(pl.Float64)
#     return ((1.0 + col - col.shift(periods=1)) / (1.0 + col.shift(periods=1))).prefix(
#         "ratio_"
#     )


def ratio(col: pl.Expr) -> pl.Expr:
    return (col / col.shift(periods=1)).prefix("ratio_")


# def ratio(column: str | list[str]) -> pl.Expr:
#     return (
#         (1 + pl.col(column) - pl.col(column).shift(1)) / (1 + pl.col(column).shift(1))
#     ).alias(f"ratio_{column}")


def diff_categorical(col: pl.Expr, fill_null=False) -> pl.Expr:
    new_col = col + "->" + col.shift(1)
    if fill_null:
        new_col = new_col.fill_null("")
    return new_col.prefix(f"diff_cat_")


def customer_id_to_int(x):
    return int(x[-16:], 16)


@overload
def casting(df: pl.DataFrame) -> pl.DataFrame:
    ...


@overload
def casting(df: pl.LazyFrame) -> pl.LazyFrame:
    ...


def casting(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    return (
        df.with_columns(
            [
                # pl.col("customer_ID").hash(seed=0),
                # pl.col("customer_ID").set_sorted(),
                pl.col("S_2").str.strptime(pl.Date, fmt="%Y-%m-%d"),
            ]
        )
        .with_columns(
            [
                pl.col("S_2").dt.year().cast(pl.UInt16).alias("Year"),
                pl.col("S_2").dt.month().cast(pl.UInt8).alias("Month"),
                pl.col("S_2").dt.day().cast(pl.UInt8).alias("Day"),
                pl.col("S_2").alias("date"),
            ]
        )
        .drop("S_2")
    )


def add_rank(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            [
                pl.count("customer_ID")
                .over(pl.col("customer_ID"))
                .cast(pl.UInt8)
                .alias("total_number"),
                pl.col("date")
                .rank(method="ordinal")
                .over(pl.col("customer_ID"))
                .cast(pl.UInt8)
                .alias("rank"),
            ]
        )
        .with_column((pl.col("rank") - pl.col("total_number") + 13))
        .drop("date")
    )


def add_index_and_count(
    df: pl.DataFrame | pl.LazyFrame, column: str = "customer_ID"
) -> pl.DataFrame | pl.LazyFrame:
    return (
        df.with_columns(
            [
                pl.col(column).count().over(column).alias("count"),
            ]
        ).with_column(pl.col("index").cumsum().alias("index"))
    ).sort(by=column)


def get_rank_df(df2: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    return (
        df2.groupby(by=["customer_ID"], maintain_order=True)
        .agg(pl.col("customer_ID").rank(method="ordinal").cast(pl.UInt8).alias("rank"))
        .explode("rank")
        # .sort(by=["customer_ID", "rank"])
        .with_column(pl.lit(1).cumsum().alias("index"))
        .with_column(pl.col("index").cumsum().alias("index"))
        .select([pl.col("index"), pl.all().exclude("index")])
    )


def rescale_rank(
    df2: pl.DataFrame | pl.LazyFrame, df_rank: pl.DataFrame | pl.LazyFrame
) -> pl.DataFrame | pl.LazyFrame:
    df3 = (
        df2.join(df_rank.select(pl.all().exclude("customer_ID")), on="index")
        .with_column(
            (pl.col("rank") - pl.col("count") + 13).cast(pl.UInt8).alias("rank")
        )
        .select(
            [
                "customer_ID",
                "count",
                "rank",
                pl.all().exclude(["customer_ID", "count", "rank", "index"]),
            ]
        )
        # .sort(by=["customer_ID", "rank"])
    )
    return df3


@overload
def process(df: pl.DataFrame) -> pl.DataFrame:
    ...


@overload
def process(df: pl.LazyFrame) -> pl.LazyFrame:
    ...


def process(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    df = add_rank(df)
    ranks = pl.Series("rank", range(1, 14), dtype=pl.UInt8).to_frame()
    if isinstance(df, pl.LazyFrame):
        ranks = ranks.lazy()

    customers = df.select("customer_ID").unique()

    df_full = (
        ranks.join(customers, how="cross")
        .sort(by=["customer_ID", "rank"])
        .select(["customer_ID", "rank"])
    )

    df_full = (
        df_full.join(df, on=["customer_ID", "rank"], how="outer")
        .select(pl.all().exclude(["count"]))
        .sort(
            by="customer_ID",
        )
    )

    return df_full


def diff_ratio(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    diffs = diff(pl.all().exclude(numeric_categorical_columns + no_diff_columns))
    ratios = ratio(pl.all().exclude(numeric_categorical_columns + no_diff_columns))

    diffs_categorical = diff_categorical(pl.col(numeric_categorical_columns))

    diff_ratio_names = [pl.col("^diff_.*$"), pl.col("^ratio_.*$")]
    df_groupby = (
        df.groupby("customer_ID", maintain_order=True)
        .agg([diffs, ratios, diffs_categorical])
        .explode(diff_ratio_names)
    )

    if isinstance(df, pl.LazyFrame):
        return hstack_lazy(df, df_groupby.select(pl.all().exclude("customer_ID")))
    else:
        return df.hstack(df_groupby.select(pl.all().exclude("customer_ID")))


def alternative_diff_ratio(df: pl.DataFrame) -> pl.DataFrame:
    diff_cat = pl.col(numeric_categorical_columns)
    diff_cat = diff_cat + "->" + diff_cat.shift(1)

    ratio = (
        pl.all()
        .exclude(numeric_categorical_columns + no_diff_columns + ["customer_ID"])
        .cast(pl.Float32)
    )
    ratio = ratio / ratio.shift(periods=1)

    diff = (
        pl.all()
        .exclude(numeric_categorical_columns + no_diff_columns + ["customer_ID"])
        .diff(n=1)
        .cast(pl.Float32)
    )

    # df.with_column(
    #         ratio.over("customer_ID").prefix("ratio_"),
    # )

    df = df.with_columns(
        [
            diff_cat.over("customer_ID").prefix("diff_cat_"),
            ratio.over("customer_ID").prefix("ratio_"),
            diff.over("customer_ID").prefix("diff_"),
        ]
    )
    return df


def filter_data(df: pl.DataFrame, fill_null=False) -> pl.DataFrame:
    # Exclude the first diff/ratio, as it is always null
    df = df.select(
        pl.all().exclude(r"^(?:diff_)(.*)(?:_1)$").exclude(r"^(?:ratio_)(.*)(?:_1)$")
    )

    # Set inf values to None
    ratio_columns = pl.col(r"^ratio_\w+$")
    df = df.with_column(
        pl.when(ratio_columns.is_infinite())
        .then(None)
        .otherwise(ratio_columns)
        .keep_name()
    )

    if fill_null:
        categorical_columns = get_categorical_columns(df)
        df = df.with_columns(
            [
                pl.col([pl.Int8, pl.Int16, pl.Int32, pl.Int64])
                .exclude(categorical_columns)
                .fill_null(-1),
                pl.col(categorical_columns)
                .cast(str)
                .cast(pl.Categorical)
                .to_physical()
                .cast(pl.Int16),
            ]
        )
    return df


def hash_columns(df) -> pl.DataFrame:
    categorical_columns = get_categorical_columns(df)
    return df.with_columns(
        pl.when(pl.col(categorical_columns).is_null())
        .then(None)
        .otherwise(pl.col(categorical_columns).hash())
    )


def encode_columns(df, dtype=pl.UInt8) -> pl.DataFrame:
    categorical_columns = get_categorical_columns(df, kind="string")
    return df.with_columns(
        pl.col(categorical_columns).cast(pl.Categorical).to_physical().cast(dtype),
    )


def pivot(df: pl.DataFrame) -> pl.DataFrame:
    column_names = df.columns[2:]

    ranks_list = list(range(1, 14))
    ranks = [str(entry) for entry in ranks_list]
    df = df.pivot(
        values=column_names, index="customer_ID", columns=["rank"], maintain_order=True
    )

    renamed_columns = df.columns[:1]
    for col in column_names:
        for rank in ranks:
            renamed_columns.append(f"{col}_{rank}")
    df.columns = renamed_columns

    return df


def split_df_into(
    df: pl.DataFrame,
    filepath: str = "artifacts/train_ratio_part_{:02d}.parquet",
) -> list[int]:

    raise ValueError("This code splits it randomly!")
    total_length = len(df)

    indices = [0]
    index = 0
    while index < total_length:
        index += 20000 * 13
        indices.append(index)

    for i, (i1, i2) in enumerate(zip(indices, indices[1:])):
        df[i1:i2].to_parquet(filepath.format(i))
    return indices


def add_index(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    return df.with_column(pl.lit(1).alias("index")).with_column(
        pl.col("index").cumsum()
    )


def hstack_lazy(
    df1: pl.DataFrame | pl.LazyFrame, df2: pl.DataFrame | pl.LazyFrame
) -> pl.DataFrame | pl.LazyFrame:
    df1 = add_index(df1)
    df2 = add_index(df2)
    return df1.join(df2, on="index").drop("index")


@overload
def run(
    df,
    target=None,
    fill_null: bool = True,
    encode: bool = True,
    use_hash: bool = False,
    train_test: bool = False,
) -> pl.DataFrame:
    ...


@overload
def run(
    df,
    target=None,
    fill_null: bool = True,
    encode: bool = True,
    use_hash: bool = False,
    train_test: bool = True,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    ...


def run(
    df: pl.DataFrame,
    target=None,
    fill_null: bool = True,
    encode: bool = True,
    use_hash: bool = False,
    train_test: bool = True,
) -> pl.DataFrame | Tuple[pl.DataFrame, pl.DataFrame]:
    df = casting(df)
    df_stats = get_column_stats(df)
    df = process(df)
    df = diff_ratio(df)
    assert not (encode and use_hash), "don't use encode and use_hash together"

    if use_hash:
        print("hashing")
        df = hash_columns(df)
    df = filter_data(df, fill_null=fill_null)

    if isinstance(df, pl.LazyFrame):
        print("Collecting before pivot!")
        df = df.collect()
        df_stats = df_stats.collect()

    print("Pivoting!")
    df = pivot(df)

    print("Joining")
    df = df.join(df_stats, on="customer_ID")
    if target:
        df = df.join(target, on="customer_ID", how="left")

    if encode:
        print("encoding categoricals as float")
        # df = encode_columns(df)
        df = sklearn_encoder(df)

    # Cast to float32
    df = df.with_column(pl.col(pl.Float64).cast(pl.Float32))

    print("Finished run!")
    if train_test:
        df_train, df_test = train_test_split(df)
        return df_train, df_test

    else:
        return df


def train_test_split(
    df: pl.DataFrame, train_fraction: float = 0.75
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Split polars dataframe into two sets.

    Args:
        df (pl.DataFrame): Dataframe to split
        train_fraction (float, optional): Fraction that goes to train. Defaults to 0.75.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: Tuple of train and test dataframes
    """
    df = df.with_column(pl.all().shuffle(seed=1))
    split_index = int(train_fraction * len(df))
    df_train = df[:split_index]
    df_test = df[split_index:]
    return (df_train, df_test)


def save(df: pl.DataFrame | pd.DataFrame, label: str):
    df.to_parquet(f"data/{label}.parquet")


@overload
def load(label: str, kind: Literal["polars"] = "polars") -> pl.DataFrame:
    ...


@overload
def load(label: str, kind: Literal["pandas"] = "pandas") -> pd.DataFrame:
    ...


def load(
    label: str, kind: Literal["polars", "pandas"] = "polars"
) -> pl.DataFrame | pd.DataFrame:
    if kind == "polars":
        df = pl.read_parquet(f"data/{label}.parquet")
    elif kind == "pandas":
        df = pd.read_parquet(f"data/{label}.parquet")
    return df


def split_frame(df: pl.DataFrame, n: int = 5):
    customer_id = df.select("customer_ID")
    step = len(df) // n

    i = 0
    previous_index = 0
    while True:
        current_index = previous_index + step
        next_index = current_index + 1

        if current_index > len(df):
            save(df[previous_index:], i)
            break

        if next_index > len(df):
            save(df[previous_index:], i)
            break

        current_customer_id = customer_id[current_index][0, 0]
        next_customer_id = customer_id[next_index][0, 0]

        while current_customer_id == next_customer_id:
            current_index += 1
            next_index += 1
            if next_index > len(df):
                next_index = len(df)
            current_customer_id = customer_id[current_index][0, 0]
            next_customer_id = customer_id[next_index][0, 0]

        save(df[previous_index : current_index + 1], i)
        i += 1
        previous_index = next_index


def lgbm_metric(
    preds: np.ndarray, train_data: "lgb.Dataset"
) -> Tuple[str, float, bool]:
    # Final boolean is whether or not to maximize metric
    target: np.ndarray = train_data.get_label()
    return ("amex", amex_metric_numpy(preds, target), True)


def xgboost_metric(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    return ("amex", amex_metric_np(preds=predt, target=dtrain.get_label()))


def amex_metric_numpy(y_pred: np.ndarray, y_true: np.ndarray) -> float:

    # count of positives and negatives
    n_pos = y_true.sum()
    n_neg = y_true.shape[0] - n_pos

    # sorting by descring prediction values
    indices = np.argsort(y_pred)[::-1]
    preds, target = y_pred[indices], y_true[indices]

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_filter = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = target[four_pct_filter].sum() / n_pos

    # weighted gini coefficient
    lorentz = (target / n_pos).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    # max weighted gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d)


def amex_metric_np(preds: np.ndarray, target: np.ndarray) -> float:
    n_pos = np.sum(target)
    n_neg = target.shape[0] - n_pos

    indices = np.argsort(preds)[::-1]
    preds, target = preds[indices], target[indices]

    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight * (1 / weight.sum())).cumsum()
    four_pct_mask = cum_norm_weight <= 0.04
    d = np.sum(target[four_pct_mask]) / n_pos

    lorentz = (target * (1 / n_pos)).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    g = gini / gini_max
    return 0.5 * (g + d)


def cv(dtrain, param: Optional[dict] = None):
    if param is None:
        param = {
            "objective": "binary:logistic",
        }

    bst = xgb.cv(
        param,
        dtrain,
        nfold=5,
        num_boost_round=10,
        early_stopping_rounds=20,
        custom_metric=xgboost_metric,
        maximize=True,
    )
    mean = bst["test-amex-mean"].mean()
    return mean


def encode_categoricals(df, astype_category: bool = False):
    categorical_columns = get_categorical_columns(df)

    categorical_columns_numeric = [
        col for col in categorical_columns if not col.startswith("diff_cat")
    ]
    categorical_columns_string = [
        col for col in categorical_columns if col.startswith("diff_cat")
    ]

    enc_numeric = preprocessing.OrdinalEncoder(dtype=np.int32, encoded_missing_value=-1)
    enc_string = preprocessing.OrdinalEncoder(dtype=np.int32, encoded_missing_value=-1)

    df.loc[:, categorical_columns_numeric] = enc_numeric.fit_transform(
        df[categorical_columns_numeric]
    )
    df.loc[:, categorical_columns_string] = enc_string.fit_transform(
        df[categorical_columns_string]
    )

    if astype_category:
        df = df.astype({col: "category" for col in categorical_columns})
    return df


def assert_equal(df1: pl.DataFrame, df2: pl.DataFrame):
    assert np.array_equal(df1.to_numpy(), df2.to_numpy())


def sklearn_encoder(df: pl.DataFrame) -> pl.DataFrame:
    enc = OrdinalEncoder(dtype=np.float32, encoded_missing_value=np.nan)
    df_categorical = df.select(get_categorical_columns(df))
    data = df_categorical.to_numpy()
    transformed = enc.fit_transform(data)

    series_transformed = [
        pl.Series(name=name, values=transformed[:, i], dtype=pl.Float32)
        for i, name in enumerate(df_categorical.columns)
    ]
    return df.with_columns(series_transformed)