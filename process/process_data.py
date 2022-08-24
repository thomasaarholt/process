from __future__ import annotations

from typing import Callable, Literal, Optional, Tuple, Union, overload

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import sympy as sp
import xgboost as xgb

from xgboost.callback import EarlyStopping

from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sympy.core import Expr, Symbol, symbols
from scipy import special
try:
    from rich import print
    blue = "\t[bold #00b8ff]"
    red = "\t[bold #ff0024]"
    green = "\t[bold #0ef139]"
    yellow = "[bold #fbf704]"
except ImportError:
    blue = "\t"
    red = "\t"
    green = "\t"
    yellow = ""

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
no_diff_columns = ["Year", "Month", "Day", "date",]
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
            (pl.col("date").max() - pl.col("date").min())
            .dt.days()
            .alias("days_from_first_to_last_payment"),
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
        .sort(by=["customer_ID", "date"])
    )

@overload
def add_rank(df: pl.DataFrame) -> pl.DataFrame:
    ...

@overload
def add_rank(df: pl.LazyFrame) -> pl.LazyFrame:
    ...

def add_rank(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    return (
        df.sort(by=["customer_ID", "date"]).with_columns(
            [
                pl.count("date")
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
        .with_column((pl.col("rank") - pl.col("total_number") + 13).alias("rank"))
        .sort(by=["customer_ID", "rank"])
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
def add_full_ranks(df: pl.DataFrame) -> pl.DataFrame:
    ...


@overload
def add_full_ranks(df: pl.LazyFrame) -> pl.LazyFrame:
    ...


def add_full_ranks(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
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
            by=["customer_ID", "rank"],
        )
    )

    return df_full

@overload
def diff_ratio(df: pl.DataFrame) -> pl.DataFrame:
    ...


@overload
def diff_ratio(df: pl.LazyFrame) -> pl.LazyFrame:
    ...


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

def diff_dates(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_column(
        pl.col("date")
        .diff()
        .dt.days()
        .over("customer_ID")
            .alias("days_since_last_payment"),
    ).drop("date")

def after_pay(df: pl.DataFrame) -> pl.DataFrame:
    bcols = ["B_11", "B_14", "B_17", "D_39", "D_131", "S_16", "S_23"]
    pcols = ['P_2','P_3']

    exprs = []
    for b in bcols:
        for p in pcols:
            exprs.append((pl.col(b) - pl.col(p)).alias(f"{b}-{p}"))
    return df.with_columns(exprs)

@overload
def filter_data(df: pl.DataFrame, fill_null: bool=False) -> pl.DataFrame:
    ...


@overload
def filter_data(df: pl.LazyFrame, fill_null: bool=False) -> pl.LazyFrame:
    ...

def filter_data(df: pl.DataFrame | pl.LazyFrame, fill_null: bool=False) -> pl.DataFrame | pl.LazyFrame:
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


def run(
    df: pl.DataFrame | pl.LazyFrame,
    target=None,
    fill_null: bool = False,
    encode: bool = True,
    use_hash: bool = False,
    train_test: bool = False,
    load_encoders: bool = False,
    handle_unknown_encodings: bool = False,
    drop_customer_ID: bool = True
) -> pl.DataFrame:
    print(yellow + "LET'S GO!")
    # 1. CAST
    df = casting(df)
    # 2. EXTRACT STATS
    df_stats = get_column_stats(df)
    # ADD 13 RANKS TO ALL
    df = add_full_ranks(df)

    # Add diff to date features
    df = diff_dates(df)

    # Add payment features
    df = after_pay(df)

    # ADD DIFF AND RATIO FEATURES
    df = diff_ratio(df)

    assert not (encode and use_hash), red + "don't use encode and use_hash together"

    if use_hash:
        print("hashing")
        df = hash_columns(df)
    # REMOVE BAD ROWS BEFORE PIVOT
    df = filter_data(df, fill_null=fill_null)
    
    if isinstance(df, pl.LazyFrame):
        print("Collecting before pivot!")
        df = df.collect()
        df_stats = df_stats.collect()

    # PIVOT DATA
    print(blue + "Pivoting!")
    df = pivot(df)

    # JOIN STATS ON PIVOTED DATA
    df = df.join(df_stats, on="customer_ID")
    if target:
        df = df.join(target, on="customer_ID", how="left")
    
    # ENCODE CATEGORICALS
    if encode:
        print(blue + "Encoding categoricals as float")
        # df = encode_columns(df)
        df = sklearn_encoder(df, load_encoders=load_encoders, handle_unknown_encodings = handle_unknown_encodings)

    # Cast to float32
    df = df.with_column(pl.col(pl.Float64).cast(pl.Float32))


    if drop_customer_ID:
        df = df.drop("customer_ID")

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
    df.write_parquet(f"data/{label}.parquet")


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
    # preds = np.rint(np.clip(preds, 0, 1))
    if isinstance(train_data, np.ndarray):
        target = train_data
    else:
        target: np.ndarray = train_data.get_label()
    return ("amex", amex_metric_numpy(preds, target), True)

def lgbm_metric_expit(
    preds: np.ndarray, train_data: "lgb.Dataset"
) -> Tuple[str, float, bool]:
    preds = special.expit(preds)
    return lgbm_metric(preds, train_data)

def predict(model, X):
    return special.expit(model.predict(X))

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


def encode_categoricals(df, astype_category: bool = False, load_encoders: bool = False):
    categorical_columns = get_categorical_columns(df)

    categorical_columns_numeric = [
        col for col in categorical_columns if not col.startswith("diff_cat")
    ]
    categorical_columns_string = [
        col for col in categorical_columns if col.startswith("diff_cat")
    ]

    if load_encoders:
        enc_numeric = joblib.load("enc_numeric")
        enc_string = joblib.load("enc_string")
    else:
        enc_numeric = preprocessing.OrdinalEncoder(dtype=np.int32, encoded_missing_value=-1)
        enc_string = preprocessing.OrdinalEncoder(dtype=np.int32, encoded_missing_value=-1)

    df.loc[:, categorical_columns_numeric] = enc_numeric.fit_transform(
        df[categorical_columns_numeric]
    )
    df.loc[:, categorical_columns_string] = enc_string.fit_transform(
        df[categorical_columns_string]
    )

    if not load_encoders:
        enc_numeric = joblib.dump(enc_numeric, "enc_numeric")
        enc_string = joblib.dump(enc_string, "enc_string")

    if astype_category:
        df = df.astype({col: "category" for col in categorical_columns})
    return df


def assert_equal(df1: pl.DataFrame, df2: pl.DataFrame):
    assert np.array_equal(df1.to_numpy(), df2.to_numpy())


def sklearn_encoder(df: pl.DataFrame, load_encoders: bool = False, handle_unknown_encodings: bool = False) -> pl.DataFrame:
    df_categorical = df.select(get_categorical_columns(df))
    data = df_categorical.to_numpy()
    
    if not load_encoders:
        print(red + "Creating new Categorical Feature Encodings!")
        enc = OrdinalEncoder(dtype=np.float32, encoded_missing_value=np.nan, )
        enc.fit(data)
        joblib.dump(enc, "categorical_feature_encodings")
    else:
        print(red + "Loading Encodings for Categorical Features!")
        enc = joblib.load("categorical_feature_encodings")
    
    if handle_unknown_encodings:
        enc.handle_unknown = 'use_encoded_value'
    transformed = enc.transform(data)
    
    series_transformed = [
        pl.Series(name=name, values=transformed[:, i], dtype=pl.Float32)
        for i, name in enumerate(df_categorical.columns)
    ]
    return df.with_columns(series_transformed)

def split_df_near_index(df: pl.DataFrame, near_index: int) -> int:
    customers = df.select(pl.col("customer_ID"))
    while customers[near_index, 0] == customers[near_index - 1, 0]:
        near_index = near_index + 1
    return near_index
    
def split_train_75_25():
    # split train data into 75 and 25 parts
    df = pl.read_parquet("artifacts/train.parquet")

    split_index = int(len(df) * 0.75)
    split_index = split_df_near_index(df, split_index)

    df_75 = df[:split_index]
    df_25 = df[split_index:]

    df_75.write_parquet("artifacts/train_75.parquet")
    df_25.write_parquet("artifacts/train_25.parquet")

    return df_75, df_25


def load_train_75():
    return pl.scan_parquet("artifacts/train_75.parquet")

def normalize_cumsum(df: pl.DataFrame, column: str) -> pl.DataFrame:
    return df.sort(by=column, reverse=True).with_column((pl.col(column) / pl.col(column).sum()).cumsum())


def get_cv_importance(cv_boosters: "lgb.CVBooster"):
    importance_gain = cv_boosters.feature_importance("gain")
    importance_split = cv_boosters.feature_importance("split")

    columns = cv_boosters.feature_name()[0]
    importance = (
        [pl.Series("columns", columns)]
        + [pl.Series(f"gain_cv_{i:02}", imp) for i, imp in enumerate(importance_gain)]
        + [pl.Series(f"split_cv_{i:02}", imp) for i, imp in enumerate(importance_split)]
        )

    df_importance = pl.DataFrame(importance)

    df_importance = df_importance.hstack(df_importance.select(pl.col(r"^gain_cv_.*$")).select(pl.fold(acc=pl.lit(0), f=lambda acc, x: acc + x, exprs=pl.col("*")).alias("gain_cumsum")))
    df_importance = df_importance.hstack(df_importance.select(pl.col(r"^split_cv_.*$")).select(pl.fold(acc=pl.lit(0), f=lambda acc, x: acc + x, exprs=pl.col("*")).alias("split_cumsum")))

    df_importance = normalize_cumsum(df_importance, "gain_cumsum")
    df_importance = normalize_cumsum(df_importance, "split_cumsum")

    df_importance = df_importance.with_columns([
        (pl.col("gain_cumsum") < 0.999).alias("gain_0.999"),
        (pl.col("split_cumsum") < 0.999).alias("split_0.999"),]
    )
    df_importance = df_importance.sort(by="gain_cumsum")
    return df_importance


# # polars
# if "target" in df.columns:
#     df_target = df.select(["customer_ID", "target"])
#     df_target.write_parquet(f"{FOLDER}/target.parquet")
#     target = df.select("target").to_series().to_pandas()
#     df = df.drop("target")




def binary_crossentropy_grad_hess() -> Tuple[Tuple[Symbol, Symbol, Symbol], Tuple[Expr, Expr, Expr]]:
    y, p, eps = symbols("y, p, epsilon")
    binary_cross = y*sp.log(p + eps) + (1-y+eps)*sp.log(1-p + eps)
    grad = sp.diff(binary_cross, p)
    hess = sp.diff(grad, p)
    return (y, p, eps), (binary_cross, grad, hess)
    
def binary_crossentropy_as_numpy() -> Tuple[Callable, Callable, Callable]:
    "All take arguments in the order `f(target, pred)`"
    (y, p, eps), (binary_cross, grad, hess) = binary_crossentropy_grad_hess()

    f_binary_cross = sp.lambdify((y, p, eps), binary_cross,)
    f_grad = sp.lambdify((y, p, eps), grad)
    f_hess = sp.lambdify((y, p, eps), hess)
    return (f_binary_cross, f_grad, f_hess)

_, f_grad, f_hess = binary_crossentropy_as_numpy()

def numpy_grad(target, pred):
    return (1 - target)/(1 - pred) + target / pred

def numpy_hess(target, pred):
    return (1-target) / (1-pred)**2 - target/pred**2

def lgbm_bin_cross_objective(y: np.ndarray, data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    target = data.get_label()
    pred = y
    EPS = 1e-7
    pred = np.clip(pred, EPS, 1 - EPS)
    # grad = numpy_grad(target, pred)
    # hess = numpy_hess(target, pred)
    grad = f_grad(target, pred, EPS)
    hess = f_hess(target, pred, EPS)
    return grad, hess

def lgbm_bin_cross_objective2(y: np.ndarray, data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    target = data.get_label()
    pred = y
    pred = special.expit(y)
    # EPS = 1e-7
    # pred = np.clip(pred, EPS, 1 - EPS)
    # grad = numpy_grad(target, pred)
    # hess = numpy_hess(target, pred)
    grad = pred - target
    hess = pred * (1-pred)
    return grad, hess




def lgb_cross_beta_func(beta: float):
    def grad(pred, target, beta):
        return -beta*pred*(target - 1) + target*(pred - 1)
    
    def hessian(pred, target, beta):
        return pred*(pred - 1) * (beta*(target-1)-target)

    def lgbm_bin_cross_objective_beta(pred: np.ndarray, data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        target: np.ndarray = data.get_label()
        pred = special.expit(pred)

        grad = pred*(beta -beta*target + target) - target
        hess = (target + beta - beta*target )*pred*(1-pred)
        return grad, hess
    return lgbm_bin_cross_objective_beta



def xgb_cross_validation(X: np.ndarray, y: np.ndarray, params: dict, early_stopping_rounds: int = 30):
    FOLDS = 5

    if isinstance(X, pd.DataFrame):
        feature_importance =  {col:0 for col in X.columns}
    else:
        feature_importance = {f'f{i}':0 for i in range(X.shape[1])}

    results = {"scores": [], "models": [], "feature_importances": []}
    for fold, (train_indices, test_indices) in enumerate(
        StratifiedKFold(FOLDS).split(X, y)
    ):
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(X_test, y_test)

        bst = xgb.train(
            params,
            dtrain=dtrain,
            evals=[(dtest, "valid")],
            custom_metric=xgboost_metric,
            callbacks=[EarlyStopping(rounds=early_stopping_rounds)],
        )
        preds = bst.predict(dtest)
        score = xgboost_metric(preds, dtest)[1]
        results["scores"].append(score)
        results["models"].append(bst)
        results["feature_importances"].append(feature_importance | bst.get_score(importance_type='gain'))

    results["score_mean"] = np.mean(results["scores"])
    results["score_std"] = np.std(results["scores"])
    results["feature_importances"] = [list(cols.values()) for cols in results["feature_importances"]]
    results["feature_importance"] = np.mean(results["feature_importances"], axis=0)

    return results


def lgb_cross_validation(X, y, params = {'importance_type':'gain'}, early_stopping_rounds=30):
    FOLDS = 5

    results = {"scores":[], "models":[], "feature_importances":[]}
    for fold, (train_indices, test_indices) in enumerate(StratifiedKFold(FOLDS).split(X, y)):
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        model = lgb.sklearn.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(-1)])

        results["scores"].append(model.score(X_test, y_test))
        results["models"].append(model)
        results["feature_importances"].append(model.feature_importances_)

    results['score_mean'] = np.mean(results["scores"])
    results['score_std'] = np.std(results["scores"])
    results['feature_importance'] = np.mean(results["feature_importances"], axis=0)

    return results


def feature_elimination_xgboost(df: pd.DataFrame, target: pd.Series, columns: list) -> Tuple[float, np.ndarray]:
        catcol = get_categorical_xgboost(df)

        train_data = xgb.DMatrix(df, label=target, enable_categorical=True, feature_types=catcol)

        param = {
            "objective": "binary",
            "gpu_device_id": 0,
            "device": "cuda_exp",
            "metric": "None",
            "custom_metric":xgboost_metric,
            "maximize":True,
        }
        results: pd.DataFrame = xgb.train(
            param, train_data, 
            num_boost_round=1000,
            folds=StratifiedKFold(5),
            early_stopping_rounds=20, 
            )

        score = results["test-amex-mean"][-1]

        feature_importances = np.sum(cv_boosters.feature_importance(), axis=0)
        return score, feature_importances

def feature_elimination_lightgbm(df: pd.DataFrame, target: pd.Series, columns: list) -> Tuple[float, np.ndarray]:
        df = df.loc[:, columns]
        print(green + f"Using {len(columns)} columns!")
        
        catcol = get_categorical_columns(df)

        train_data = lgb.Dataset(df, label=target, categorical_feature=catcol)

        param = {
            "objective": "binary",
            "gpu_device_id": 0,
            "device": "cuda_exp",
            "metric": "None",
        }
        results = lgb.cv(
            param,
            train_set=train_data,
            folds=StratifiedKFold(5),
            num_boost_round=3,
            feval=lgbm_metric,
            callbacks=[lgb.early_stopping(33)],
            return_cvbooster=True,
        )

        cv_boosters: lgb.CVBooster = results["cvbooster"]
        score = np.mean(results["valid amex-mean"])
        feature_importances = np.sum(cv_boosters.feature_importance(), axis=0)
        return score, feature_importances

def recursive_feature_elimination(df: pd.DataFrame) -> pl.DataFrame:
    target = df["target"]
    df = df.drop(columns="target")
    print(f"[bold green]DataFrame has shape: [bold red]{df.shape}[bold green]![/bold green]")
    score = 0
    best_score = 0

    SAVEPATH = "data"
    SAVENAME = "feature_importance"
    columns: list[str] = df.columns.to_list()

    for i in range(10):
        df = df.loc[:, columns]
        print(green + f"Using {len(columns)} columns!")
        score, feature_importances = feature_elimination_lightgbm(df, target, columns)
        if score > best_score:
            color = "bold green"
            best_score = score
        else:
            color = "bold red"

        print(blue + f"Mean score: {score:.4f}[/{color}]!")

        df_importance = pl.DataFrame({"columns": df.columns.to_list(), "feature_importance": feature_importances})
        df_importance = df_importance.sort(by="feature_importance", reverse=True)
        df_importance.write_csv(f"{SAVEPATH}/{SAVENAME}_{i:02}.csv")

        importances: list[str] = df_importance.select("columns").to_series().to_list()
        fraction = 0.7
        columns = importances[:int(fraction * df.shape[1])]

        print(f"[bold yellow]Best score: {best_score:.4f}[/bold yellow]!")
    print(f"[bold #1dfd02] saved in {SAVEPATH} as {SAVENAME} by iteration!")
    return df_importance

# # This cell tested a custom objective function. Takeaways are:
# # - no gain, even for small beta values
# # - use of custom objective in lightgbm and xgboost means that prediction values must
# # be transformed by f(x) = 1 / (1 + exp(-x)) sigmoid function
# # - we also had to pass `feval=lgbm_metric_expit,` to train!

# catcol = get_categorical_columns(df_train)
# train_data = lgb.Dataset(df_train, label=target, categorical_feature=catcol, free_raw_data=False)
# valid_data = lgb.Dataset(df_valid, label=valid_target, categorical_feature=catcol, reference=train_data, free_raw_data=False)

# scores = {}
# cms = {}
# numbers = np.linspace(1,4,8)[1:]
# betas = (1/numbers)[::-1].tolist() + [1] + numbers.tolist()
# for beta in betas:
#     param = {
#         "objective": lgb_cross_beta_func(beta),
#         "gpu_device_id": 0,
#         "device": "cuda_exp",
#         "metric": "None",
#     }
#     model = lgb.train(
#         param,
#         num_boost_round=30,
#         train_set=train_data,
#         valid_sets=[valid_data],
#         feval=lgbm_metric_expit,
#         callbacks=[lgb.early_stopping(10)],
#     )
#     scores[f"{beta}"] = model.best_score["valid_0"]["amex"]
#     cms[f"{beta}"] = confusion_matrix(valid_target,np.rint(model.predict(df_valid)))
#     print(scores[f"{beta}"])

# import matplotlib.pyplot as plt
# x = np.linspace(-(len(scores) - 1)//2, (len(scores) - 1)//2, len(scores))
# plt.plot(x, scores.values())

from sklearn.metrics import make_scorer

def make_amex_metric_sklearn():
    return make_scorer(amex_metric_np, greater_is_better=True)