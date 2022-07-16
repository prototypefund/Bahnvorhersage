import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from helpers import RtdRay
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import dask.dataframe as dd
from typing import Literal
from config import JSON_MODEL_PATH, ENCODER_PATH


def save_model(model: XGBClassifier, minute: int, ar_or_dp: Literal['ar', 'dp']):
    """Save trained XGBClassifier model to disk

    Parameters
    ----------
    minute : int
        The minute of delay that the model should predict
    ar_or_dp : str : `ar` | `dp`
        Whether the model should predict arrival or departure delays

    Raises
    ------
    ValueError
        `ar_or_dp` is neither `ar` nor `dp`
    """
    if ar_or_dp == 'ar':
        model.save_model(JSON_MODEL_PATH.format('ar_' + str(minute)))
    elif ar_or_dp == 'dp':
        model.save_model(JSON_MODEL_PATH.format('dp_' + str(minute)))
    else:
        raise ValueError(f'ar_or_dp has to be ar or dp not {ar_or_dp}')


def load_model(minute: int, ar_or_dp: Literal['ar', 'dp'], gpu=False) -> XGBClassifier:
    """Load trained XGBClassifier model for prediction form disk

    Parameters
    ----------
    minute : int
        The minute of delay that the model should predict
    ar_or_dp : str : `ar` | `dp`
        Whether the model should predict arrival or departure delays
    gpu : bool
        Load model for prediction on GPU, by default False

    Returns
    -------
    XGBClassifier
        The trained classifier model

    Raises
    ------
    ValueError
        `ar_or_dp` is neither `ar` nor `dp`
    """
    booster = XGBClassifier()
    if ar_or_dp == 'ar':
        booster.load_model(JSON_MODEL_PATH.format('ar_' + str(minute)))
    elif ar_or_dp == 'dp':
        booster.load_model(JSON_MODEL_PATH.format('dp_' + str(minute)))
    else:
        raise ValueError(f'ar_or_dp has to be ar or dp not {ar_or_dp}')

    if gpu:
        booster.set_params(predictor='gpu_predictor')

    return booster


def split_ar_dp(rtd: pd.DataFrame | dd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Some datapoints contain ar and dp information, some only ar or dp information. Split the data into clear ar and dp subsets.

    Parameters
    ----------
    rtd : pd.DataFrame | dd.DataFrame
        The dataframe to split. Loaded from RtdRay.load_for_ml_model()

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ar and dp subsets of the data. These subsets are likely not the same size.
    """
    status_encoder = {
        'ar': pickle.load(open(ENCODER_PATH.format(encoder="ar_cs"), "rb")),
        'dp': pickle.load(open(ENCODER_PATH.format(encoder="dp_cs"), "rb")),
    }

    ar = rtd.loc[~rtd["ar_delay"].isna() | (rtd["ar_cs"] == status_encoder["ar"]["c"])]
    dp = rtd.loc[~rtd["dp_delay"].isna() | (rtd["dp_cs"] == status_encoder["dp"]["c"])]

    return ar, dp


def train_model(train_x, train_y, **model_parameters):
    print("Majority Baseline during training:", majority_baseline(train_x, train_y))
    est = XGBClassifier(
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=0,
        tree_method="gpu_hist",
        use_label_encoder=False,
        **model_parameters,
    )
    est.fit(train_x, train_y)
    return est


def train_models(n_models=15, **load_parameters):
    train = RtdRay.load_for_ml_model(**load_parameters).compute()
    ar_train, dp_train = split_ar_dp(train)
    del train

    status_encoder = {
        'ar': pickle.load(open(ENCODER_PATH.format(encoder="ar_cs"), "rb")),
        'dp': pickle.load(open(ENCODER_PATH.format(encoder="dp_cs"), "rb")),
    }

    ar_labels = {}
    dp_labels = {}
    for minute in range(n_models):
        ar_labels[minute] = (ar_train["ar_delay"] <= minute) & (
            ar_train["ar_cs"] != status_encoder["ar"]["c"]
        )
        dp_labels[minute + 1] = (dp_train["dp_delay"] >= (minute + 1)) & (
            dp_train["dp_cs"] != status_encoder["dp"]["c"]
        )

    del ar_train["ar_delay"]
    del ar_train["dp_delay"]
    del ar_train["ar_cs"]
    del ar_train["dp_cs"]

    del dp_train["ar_delay"]
    del dp_train["dp_delay"]
    del dp_train["ar_cs"]
    del dp_train["dp_cs"]

    newpath = "cache/models"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # fmt: off
    parameters = {
        -1: {'learning_rate': 0.4, 'max_depth': 14, 'n_estimators': 100, 'gamma': 2.8,},
        0: {'learning_rate': 0.4, 'max_depth': 14, 'n_estimators': 100, 'gamma': 2.8,},
        1: {'learning_rate': 0.4, 'max_depth': 14, 'n_estimators': 100, 'gamma': 2.8,},
        2: {'learning_rate': 0.4, 'max_depth': 13, 'n_estimators': 100, 'gamma': 2.8,},
        3: {'learning_rate': 0.4, 'max_depth': 13, 'n_estimators': 100, 'gamma': 2.8,},
        4: {'learning_rate': 0.4, 'max_depth': 12, 'n_estimators': 100, 'gamma': 2.8,},
        5: {'learning_rate': 0.4, 'max_depth': 12, 'n_estimators': 100, 'gamma': 2.8,},
        6: {'learning_rate': 0.4, 'max_depth': 11, 'n_estimators': 100, 'gamma': 2.8,},
        7: {'learning_rate': 0.4, 'max_depth': 11, 'n_estimators': 100, 'gamma': 2.8,},
        8: {'learning_rate': 0.4, 'max_depth': 10, 'n_estimators': 100, 'gamma': 2.8,},
        9: {'learning_rate': 0.4, 'max_depth': 10, 'n_estimators': 100, 'gamma': 2.8,},
        10: {'learning_rate': 0.4, 'max_depth': 10, 'n_estimators': 100, 'gamma': 2.8,},
        11: {'learning_rate': 0.4, 'max_depth': 9, 'n_estimators': 100, 'gamma': 2.8,},
        12: {'learning_rate': 0.4, 'max_depth': 9, 'n_estimators': 100, 'gamma': 2.8,},
        13: {'learning_rate': 0.4, 'max_depth': 8, 'n_estimators': 100, 'gamma': 2.8,},
        14: {'learning_rate': 0.4, 'max_depth': 8, 'n_estimators': 100, 'gamma': 2.8,},
    }
    # fmt: on

    for minute in range(n_models):
        model_name = f"ar_{minute}"
        print("Training", model_name, '. . .')
        model = train_model(ar_train, ar_labels[minute], **parameters[minute])
        save_model(model, minute=minute, ar_or_dp='ar')
        print("Training", model_name, "done.")

        minute += 1
        model_name = f"dp_{minute}"
        print("Training", model_name, '. . .')
        model = train_model(dp_train, dp_labels[minute], **parameters[minute - 1])
        save_model(model, minute=minute, ar_or_dp='dp')
        print("Training", model_name, "done.")


def majority_baseline(x, y):
    clf = DummyClassifier(strategy="most_frequent", random_state=0)
    clf.fit(x, y)
    return round((clf.predict(x) == y.to_numpy()).sum() / len(x), 6)


def model_score(model, x, y):
    return model.score(x, y)


def test_model(model, x_test, y_test, model_name):
    baseline = majority_baseline(x_test, y_test)
    model_score = (model.predict(x_test) == y_test).sum() / len(y_test)

    print("Model:\t\t\t", model_name)
    print("Majority baseline:\t", round(baseline * 100, 6))
    print("Model accuracy:\t\t", round(model_score * 100, 6))
    print("Model improvement:\t", round((model_score - baseline) * 100, 6))


def test_models(n_models=15, **load_parameters):
    status_encoder = {
        'ar': pickle.load(open(ENCODER_PATH.format(encoder="ar_cs"), "rb")),
        'dp': pickle.load(open(ENCODER_PATH.format(encoder="dp_cs"), "rb")),
    }

    test = RtdRay.load_for_ml_model(**load_parameters).compute()
    ar_test, dp_test = split_ar_dp(test)

    ar_test_x = ar_test.drop(columns=["ar_delay", "dp_delay", "ar_cs", "dp_cs"])
    dp_test_x = dp_test.drop(columns=["ar_delay", "dp_delay", "ar_cs", "dp_cs"])
    del test

    for model_number in range(n_models):
        model_name = f"ar_{model_number}"
        print("test_results for model {}".format(model_name))
        test_y = (ar_test["ar_delay"] <= model_number) & (
            ar_test["ar_cs"] != status_encoder["ar"]["c"]
        )
        model = load_model(minute=model_number, ar_or_dp="ar", gpu=True)
        test_model(model, ar_test_x, test_y, model_name)

        model_number += 1
        model_name = f"dp_{model_number}"
        print("test_results for model {}".format(model_name))
        test_y = (dp_test["dp_delay"] >= model_number) & (
            dp_test["dp_cs"] != status_encoder["dp"]["c"]
        )
        model = load_model(minute=model_number, ar_or_dp="dp", gpu=True)
        test_model(model, dp_test_x, test_y, model_name)


def model_roc(model, x_test, y_test, model_name):
    from sklearn.metrics import (
        precision_recall_curve,
        plot_precision_recall_curve,
        auc,
        roc_curve,
    )

    prediction = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, prediction, pos_label=1)
    roc_auc = auc(fpr, tpr)

    lw = 2
    fig, ax = plt.subplots()
    ax.plot(
        fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver operating characteristic model {}".format(model_name))
    ax.legend(loc="lower right")

    # fig1, ax1 = plt.subplots()
    # ax1.set_title('Predictions')
    # ax1.boxplot(prediction)
    plt.show()

    # plt.scatter(prediction, y_test, color='red', label='prediction')
    # from sklearn import linear_model
    # # Create linear regression object
    # regr = linear_model.LinearRegression()

    # # Train the model using the training sets
    # regr.fit(prediction.reshape(-1, 1), y_test)
    # reg_y = regr.predict(prediction.reshape(-1, 1))
    # plt.plot(prediction, reg_y, color='blue', linewidth=3)

    # # plt.axis('tight')
    # plt.axis('scaled')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

    # from sklearn.metrics import precision_recall_curve
    # from sklearn.metrics import plot_precision_recall_curve

    # disp = plot_precision_recall_curve(model, x_test, y_test)
    # disp.ax_.set_title('2-class Precision-Recall curve')
    # plt.show()


if __name__ == "__main__":
    import helpers.bahn_vorhersage

    from dask.distributed import Client

    # Setting `threads_per_worker` is very important as Dask will otherwise
    # create as many threads as cpu cores which is to munch for big cpus with small RAM
    with Client(n_workers=min(10, os.cpu_count() // 4), threads_per_worker=2) as client:

        train_models(
            min_date=datetime.datetime.today() - datetime.timedelta(days=7 * 6),
            return_status=True,
            obstacles=False,
        )

        test_models(
            max_date=datetime.datetime.today() - datetime.timedelta(days=7 * 6),
            min_date=datetime.datetime.today() - datetime.timedelta(days=7 * 8),
            return_status=True,
            obstacles=False,
        )
