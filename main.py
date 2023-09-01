import logging
import sys
import warnings

from decimal import Decimal

from hashlib import md5
from uuid import UUID

import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logging.getLogger()


# Data Preperation
def MSRP(df, purchasePriceName="PurchasePrice", discountPctName="DiscountPct"):
    """Create MSRP feature being very careful about prserving the correct precission."""
    disc_pct_dec = df[discountPctName].apply(
        lambda x: Decimal(x).quantize(Decimal("1.0000"))
    )
    purch_price_dec = df[purchasePriceName].apply(
        lambda x: Decimal(x).quantize(Decimal("1.00"))
    )
    return (purch_price_dec / (1 - disc_pct_dec)).apply(
        lambda x: x.quantize(Decimal("1.0"))
    )


def ProductID(df, product_id_cols):
    """Create a placehorder product id by hashing column that uniquely idenfity a product."""
    return df[product_id_cols].apply(
        lambda x: str(
            UUID(md5("".join([str(y) for y in x.to_list()]).encode()).hexdigest())
        ),
        axis=1,
    )


def grouped_feature(df, groupers, values, agg, offset, name):
    """Generic method for creating features applying a groupby then aggregating"""
    feature = df.groupby(groupers)[values].agg(agg) - offset
    feature.name = name
    return feature


def map_grouped_feature(df, grouped_feature, fks, pk, how="left"):
    """Map a feature from one granularity to another.
    For use primarly with mapping grouped features back to groupers from the core table.
    """
    return df[[pk] + fks].merge(grouped_feature, on=fks, how=how)[
        [pk, grouped_feature.name]
    ]


def order_date_features(df, orderDate="OrderDate"):
    """Derive date features"""
    df["OrderDate"] = pd.to_datetime(df[orderDate])
    df["OrderMonth"] = df[orderDate].dt.month
    df["OrderDayOfWeek"] = df[orderDate].dt.dayofweek
    df["OrderIsWeekend"] = df[orderDate].isin([5, 6])
    return df[["ID", "OrderMonth", "OrderDayOfWeek", "OrderIsWeekend"]].copy()


def data_prep(df, pk, target_name, product_id, train=True, avg_return_by_product_=None):
    """Combine all the data prep into one easy function."""
    if not train:
        if isinstance(avg_return_by_product_, pd.Series):
            pass
        else:
            raise ValueError(
                "If `train = False` then must pass a pd.Series as"
                " `avg_return_by_product_`"
            )

    df["MSRP"] = MSRP(df)

    df["ProductID"] = ProductID(df, product_id)

    order_date_feats = order_date_features(df)

    if train:
        avg_return_by_product_ = grouped_feature(
            df, ["ProductID"], target_name, "mean", 0, "avg_return_by_product"
        )

    dup_prod_per_order_ = grouped_feature(
        df, ["OrderID", "ProductID"], "ProductSize", "count", 1, "dup_prod_per_order"
    )
    dup_prod_per_order = map_grouped_feature(
        df, dup_prod_per_order_, ["OrderID", "ProductID"], pk
    )
    avg_return_by_product = map_grouped_feature(
        df, avg_return_by_product_, ["ProductID"], pk
    )

    X, y = (
        avg_return_by_product.merge(dup_prod_per_order, on=pk).merge(
            order_date_feats, on=pk
        ),
        df[target_name],
    )

    return X, y, avg_return_by_product_


# Modeling
def split(X, y, stratify_with_target=True, split_kwargs=None):
    """Train test split with option for stratification."""
    if stratify_with_target:
        return train_test_split(X, y, stratify=y, **split_kwargs)
    else:
        return train_test_split(X, y, **split_kwargs)


def train(X, y, model, model_kwargs=None):
    """Train a model."""
    if model_kwargs:
        M = model(**model_kwargs)
    else:
        M = model()
    M.fit(X, y)
    return M


def predict(X, model):
    return model.predict(X)


def predict_probability(X, model):
    return model.predict_proba(X)


def score_model(y, pred, score):
    """Evaluate for a given score function and predicted values"""
    return score(y, pred)


def all_in_one(X, y, model, score, kwargs={}):
    """Combine all of the model training and evaluation into one easy fuction."""
    X_train, X_test, y_train, y_test = split(
        X, y, True, {"random_state": 666, "test_size": 0.2}
    )
    M = train(X_train, y_train, model, model_kwargs=kwargs.get("model_kwargs"))
    pred_train = predict(X_train, M)
    pred_proba_train = predict_probability(X_train, M)[:, 1]
    pred = predict(X_test, M)
    pred_proba = predict_probability(X_test, M)[:, 1]
    score_train = score_model(y_train, pred_proba_train, score)
    score_test = score_model(y_test, pred_proba, score)
    logging.info(f"Train Score: {score_train}")
    logging.info(f"Test Score: {score_test}")
    return X_train, X_test, y_train, y_test, M, pred_train, pred, pred_proba


if __name__ == "__main__":
    """Prep the data, train the model, Infer results for test data.
    Optionally pass the directory containing the data as an arguement,
    otherwise put the data in the present working directory."""

    # Set parameters
    try:
        data_dir = sys.argv[1]  # "../data/Transactions/"
    except IndexError:
        data_dir = "./"
    product_id = ["ProductDepartment", "ProductCost", "MSRP"]
    target_name = "Returned"
    pk = "ID"

    # Load data
    df = pd.read_csv(f"{data_dir}train.csv")
    test = pd.read_csv(f"{data_dir}test.csv")

    # Create training data set and product return rate mapping for use in inference.  EDA and feature selection ommitted for brevity.
    X, y, avg_return_by_product_ = data_prep(df, pk, target_name, product_id)

    # Create test data for inference.
    X_inference, y_inference, _ = data_prep(
        test, pk, target_name, product_id, False, avg_return_by_product_
    )

    # Use mlflow to experiment with hyperparameter tuning and model selection ommited for brevity.
    best_xgboost = {
        "random_state": 666,
        "n_jobs": -1,
        "max_depth": 3,
        "n_estimators": 150,
        "lambda": 4,
        "eta": 0.2,
    }

    # Train model with optimal parameters.
    X_train, X_test, y_train, y_test, M, pred_train, pred, pred_proba = all_in_one(
        X.drop("ID", axis=1),
        y,
        XGBClassifier,
        roc_auc_score,
        kwargs={"model_kwargs": best_xgboost},
    )

    # Inference on test set.
    pred_proba_inference = M.predict_proba(X_inference.drop("ID", axis=1))[:, 1]

    # Prepare submission
    submission = X_inference["ID"].to_frame().copy()
    submission["Prediction"] = pred_proba_inference

    submission.to_csv(f"{data_dir}/submission.csv", index=False)
