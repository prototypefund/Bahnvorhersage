import datetime
import os
import pickle
from typing import Literal

import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

from config import ENCODER_PATH, JSON_MODEL_PATH
from helpers import RtdRay
from ml_models.predictor import Predictor, load_model


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
        'ar': pickle.load(open(ENCODER_PATH.format(encoder='ar_cs'), 'rb')),
        'dp': pickle.load(open(ENCODER_PATH.format(encoder='dp_cs'), 'rb')),
    }

    ar = rtd.loc[~rtd['ar_delay'].isna() | (rtd['ar_cs'] == status_encoder['ar']['c'])]
    dp = rtd.loc[~rtd['dp_delay'].isna() | (rtd['dp_cs'] == status_encoder['dp']['c'])]

    return ar, dp


def train_model(train_x, train_y, **model_parameters):
    # print("Majority Baseline during training:", majority_baseline(train_x, train_y))
    est = XGBClassifier(
        n_jobs=-1,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=0,
        tree_method='gpu_hist',
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
        'ar': pickle.load(open(ENCODER_PATH.format(encoder='ar_cs'), 'rb')),
        'dp': pickle.load(open(ENCODER_PATH.format(encoder='dp_cs'), 'rb')),
    }

    ar_labels = {}
    dp_labels = {}
    for minute in range(n_models):
        ar_labels[minute] = (ar_train['ar_delay'] <= minute) & (
            ar_train['ar_cs'] != status_encoder['ar']['c']
        )
        dp_labels[minute] = (dp_train['dp_delay'] >= (minute)) & (
            dp_train['dp_cs'] != status_encoder['dp']['c']
        )

    del ar_train['ar_delay']
    del ar_train['dp_delay']
    del ar_train['ar_cs']
    del ar_train['dp_cs']

    del dp_train['ar_delay']
    del dp_train['dp_delay']
    del dp_train['ar_cs']
    del dp_train['dp_cs']

    newpath = 'cache/models'
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

    for minute in tqdm(range(n_models), desc='Training models'):
        model_name = f'ar_{minute}'
        print('Training', model_name, '. . .')
        model = train_model(ar_train, ar_labels[minute], **parameters[minute])
        save_model(model, minute=minute, ar_or_dp='ar')
        print('Training', model_name, 'done.')

        model_name = f'dp_{minute}'
        print('Training', model_name, '. . .')
        model = train_model(dp_train, dp_labels[minute], **parameters[minute - 1])
        save_model(model, minute=minute, ar_or_dp='dp')
        print('Training', model_name, 'done.')


def majority_baseline(x, y):
    clf = DummyClassifier(strategy='most_frequent', random_state=0)
    clf.fit(x, y)
    return clf.score(x, y.astype(np.uint8))


def test_model(model, x, y):
    baseline = majority_baseline(x, y)
    model_score = model.score(x, y.astype(np.uint8))

    return {
        'baseline': baseline,
        'accuracy': model_score,
        'improvement': model_score - baseline,
    }


def test_models(n_models=15, **load_parameters):
    status_encoder = {
        'ar': pickle.load(open(ENCODER_PATH.format(encoder='ar_cs'), 'rb')),
        'dp': pickle.load(open(ENCODER_PATH.format(encoder='dp_cs'), 'rb')),
    }

    test = RtdRay.load_for_ml_model(**load_parameters).compute()
    ar_test, dp_test = split_ar_dp(test)

    ar_test_x = ar_test.drop(columns=['ar_delay', 'dp_delay', 'ar_cs', 'dp_cs'])
    dp_test_x = dp_test.drop(columns=['ar_delay', 'dp_delay', 'ar_cs', 'dp_cs'])
    del test

    test_results = []

    for model_number in tqdm(range(n_models), desc='Testing models'):
        ar_test_y = (ar_test['ar_delay'] <= model_number) & (
            ar_test['ar_cs'] != status_encoder['ar']['c']
        )
        model = load_model(minute=model_number, ar_or_dp='ar', gpu=True)
        ar_result = test_model(model, ar_test_x, ar_test_y)
        test_results.append({'minute': model_number, 'ar_or_dp': 'ar', **ar_result})

        dp_test_y = (dp_test['dp_delay'] >= model_number) & (
            dp_test['dp_cs'] != status_encoder['dp']['c']
        )
        model = load_model(minute=model_number, ar_or_dp='dp', gpu=True)
        dp_results = test_model(model, dp_test_x, dp_test_y)
        test_results.append({'minute': model_number, 'ar_or_dp': 'dp', **dp_results})
    return test_results


def feature_importance(ar_or_dp: Literal['ar', 'dp']):
    import matplotlib.pyplot as plt
    import seaborn as sns

    importances = np.zeros((15, Predictor.model(0, ar_or_dp).n_features_in_))
    for i in range(15):
        importances[i, :] = Predictor.model(i, ar_or_dp).feature_importances_

    importances = pd.DataFrame(importances, columns=Predictor.FEATURES)
    importances.rename(columns=Predictor.FEATUER_NAMES, inplace=True)

    ax = sns.heatmap(importances, annot=True, linewidths=0.5)
    ax.set_ylabel('Model number', fontsize=30)
    ax.set_xlabel('Feature', fontsize=30)

    fig = ax.get_figure()
    fig.set_size_inches(13.6, 8.5)
    fig.savefig(f'feature_importance_{ar_or_dp}.png', dpi=300, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    from dask.distributed import Client

    from helpers.bahn_vorhersage import COLORFUL_ART

    print(COLORFUL_ART)

    # feature_importance('ar')
    # feature_importance('dp')
    # Setting `threads_per_worker` is very important as Dask will otherwise
    # create as many threads as cpu cores which is to munch for big cpus with small RAM
    with Client(n_workers=min(10, os.cpu_count() // 4), threads_per_worker=2) as client:
        train_models(
            min_date=datetime.datetime.today() - datetime.timedelta(days=7 * 6),
            return_status=True,
            obstacles=False,
        )

        test_result = test_models(
            max_date=datetime.datetime.today() - datetime.timedelta(days=7 * 6),
            min_date=datetime.datetime.today() - datetime.timedelta(days=7 * 8),
            return_status=True,
            obstacles=False,
        )
