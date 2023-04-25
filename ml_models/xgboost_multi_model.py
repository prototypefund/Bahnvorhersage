import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from helpers import RtdRay, ttl_lru_cache, StreckennetzSteffi
import datetime
import pandas as pd
import dask.dataframe as dd
from typing import Literal
import numpy as np
from tqdm import tqdm
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
    # print("Majority Baseline during training:", majority_baseline(train_x, train_y))
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

    for minute in tqdm(range(n_models), desc="Training models"):
        model_name = f"ar_{minute}"
        # print("Training", model_name, '. . .')
        model = train_model(ar_train, ar_labels[minute], **parameters[minute])
        save_model(model, minute=minute, ar_or_dp='ar')
        # print("Training", model_name, "done.")

        minute += 1
        model_name = f"dp_{minute}"
        # print("Training", model_name, '. . .')
        model = train_model(dp_train, dp_labels[minute], **parameters[minute - 1])
        save_model(model, minute=minute, ar_or_dp='dp')
        # print("Training", model_name, "done.")


def majority_baseline(x, y):
    clf = DummyClassifier(strategy="most_frequent", random_state=0)
    clf.fit(x, y)
    return clf.score(x, y.astype(np.uint8))


def test_model(model, x, y):
    baseline = majority_baseline(x, y)
    model_score = model.score(x, y.astype(np.uint8))

    return {
        "baseline": baseline,
        "accuracy": model_score,
        'improvement': model_score - baseline,
    }


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

    test_results = []

    for model_number in tqdm(range(n_models), desc="Testing models"):
        ar_test_y = (ar_test["ar_delay"] <= model_number) & (
            ar_test["ar_cs"] != status_encoder["ar"]["c"]
        )
        model = load_model(minute=model_number, ar_or_dp="ar", gpu=True)
        ar_result = test_model(model, ar_test_x, ar_test_y)
        test_results.append({'minute': model_number, 'ar_or_dp': 'ar', **ar_result})

        dp_test_y = (dp_test["dp_delay"] >= model_number) & (
            dp_test["dp_cs"] != status_encoder["dp"]["c"]
        )
        model = load_model(minute=model_number, ar_or_dp="dp", gpu=True)
        dp_results = test_model(model, dp_test_x, dp_test_y)
        test_results.append({'minute': model_number, 'ar_or_dp': 'dp', **dp_results})
    return test_results


def feature_importance(ar_or_dp: Literal['ar', 'dp']):
    import seaborn as sns
    import matplotlib.pyplot as plt

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
    fig.savefig(f"feature_importance_{ar_or_dp}.png", dpi=300, bbox_inches='tight')
    plt.clf()


class Predictor:
    CATEGORICALS = ['o', 'c', 'n', 'station', 'pp']
    FEATURES = [
        'station',
        'lat',
        'lon',
        'o',
        'c',
        'n',
        'distance_to_start',
        'distance_to_end',
        'pp',
        'stop_id',
        'minute',
        'day',
        'stay_time',
    ]

    FEATUER_NAMES = {
        'station': 'Station',
        'lat': 'Latitude',
        'lon': 'Longitude',
        'o': 'Operator',
        'c': 'Train type',
        'n': 'Train number',
        'distance_to_start': 'Distance to start',
        'distance_to_end': 'Distance to end',
        'pp': 'Platform',
        'stop_id': 'Stop number',
        'minute': 'Minute',
        'day': 'Day',
        'stay_time': 'Stay time',
    }

    FEATURE_DTYPES = {
        'station': 'int',
        'lat': 'float',
        'lon': 'float',
        'o': 'int',
        'c': 'int',
        'n': 'int',
        'distance_to_start': 'float',
        'distance_to_end': 'float',
        'pp': 'int',
        'stop_id': 'int',
        'minute': 'int',
        'day': 'int',
        'stay_time': 'float',
    }

    def __init__(self, n_models: int = 15):
        self.n_models = n_models

    @ttl_lru_cache(seconds_to_live=60 * 60 * 4)
    def categorical_encoder(
        self, category: Literal['o', 'c', 'n', 'station', 'pp']
    ) -> dict[str, int]:
        """Load categorical encoder either from disk or time sensitive cache.

        Parameters
        ----------
        category: str: `o` | `c` | `n` | `station` | `pp`
            The category that should be encoded by the encoder

        Returns
        -------
        dict[str, int]
            The encoder dict to map any str / category to a int

        Raises
        ------
        ValueError
            The supplied category does not exist
        """
        if category not in self.CATEGORICALS:
            raise ValueError(
                f'category {category} is unknown. Valid categories are: {self.CATEGORICALS}'
            )
        return pickle.load(open(ENCODER_PATH.format(encoder=category), 'rb'))

    @ttl_lru_cache(seconds_to_live=60 * 60 * 4)
    @staticmethod
    def model(minute: int, ar_or_dp: Literal['ar', 'dp']) -> XGBClassifier:
        """Load trained XGBClassifier model for prediction either form disk or time sensitive cache. See `load_model` for more details."""
        return load_model(minute, ar_or_dp)

    def predict_ar(self, features: pd.DataFrame) -> np.ndarray:
        features = features.to_numpy()
        prediction = np.empty((len(features), self.n_models))
        for model in range(self.n_models):
            prediction[:, model] = Predictor.model(model, 'ar').predict_proba(
                features, validate_features=False
            )[:, 1]
        return prediction

    def predict_dp(self, features: pd.DataFrame) -> np.ndarray:
        features = features.to_numpy()
        prediction = np.empty((len(features), self.n_models))
        for model in range(self.n_models):
            prediction[:, model] = Predictor.model(model, 'dp').predict_proba(
                features, validate_features=False
            )[:, 1]
        return prediction

    def predict_con(
        self,
        ar_prediction: np.ndarray,
        dp_prediction: np.ndarray,
        transfer_times: np.ndarray,
    ) -> np.ndarray:
        """Calculate connection score based on arrival predictions,
        departure predictions and transfer times.

        Parameters
        ----------
        ar_p : np.ndarray
            Arrival predictions
        dp_p : np.ndarray
            Departure predictions
        transfer_times : np.ndarray
            Transfer times between arrival and departure

        Returns
        -------
        np.ndarray
            Connections scores (0 <= con_score <= 1)
        """
        con_score = np.ones(len(transfer_times))

        for tra_time in range(self.n_models):
            # Mask
            m = transfer_times == tra_time
            if m.any():
                # If the transfer time is 5 minutes, the arriving train can be
                # delayed by up to 3 minutes, if the departing train is on time.
                # If the transfer time is 1 minute, even if the arriving train
                # is on time, the departing train musst be delayed by 1 minute.
                # Note: 0 <= tra_time < self.n_models
                max_ar_d = max(tra_time - 2, 0)
                min_dp_d = max(0, 2 - tra_time)

                # Calculate probability that the connection is made,
                # if the departing train is on time.
                con_score[m] = ar_prediction[m, max_ar_d] * dp_prediction[m, min_dp_d]

                # Calculate extra probability that the connection is made,
                # if the departing train is delayed
                con_score[m] = con_score[m] + np.sum(
                    (
                        ar_prediction[
                            m, max_ar_d + 1 : dp_prediction.shape[1] - min_dp_d
                        ]
                        - ar_prediction[
                            m, max_ar_d : dp_prediction.shape[1] - 1 - min_dp_d
                        ]
                    )
                    * dp_prediction[
                        m, min_dp_d + 1 : dp_prediction.shape[1] + min(2 - tra_time, 0)
                    ],
                    axis=1,
                )
        # Somtimes, due to inaccuracies in the prediction, the connection score
        # can be greater than 1. This is not possible, so we set it to 1.
        return np.minimum(con_score, np.ones(len(con_score)))

    def get_pred_data(
        self,
        segments: list[dict],
        streckennetz: StreckennetzSteffi,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Collect data needed for prediction.

        Parameters
        ----------
        segments : list[dict]
            The segments of a train journey
        streckennetz : StreckennetzSteffi
            An instance of the StreckennetzSteffi class

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ar_data, dp_data
        """
        dtypes = {
            'station': 'int',
            'lat': 'float',
            'lon': 'float',
            'o': 'int',
            'c': 'int',
            'n': 'int',
            'distance_to_start': 'float',
            'distance_to_end': 'float',
            'pp': 'int',
            'stop_id': 'int',
            'minute': 'int',
            'day': 'int',
            'stay_time': 'float',
        }
        ar_data = pd.DataFrame(
            columns=self.FEATURES,
            index=range(len(segments)),
        )
        dp_data = ar_data.copy()
        for i, segment in enumerate(segments):
            # Encode categoricals
            for cat in self.CATEGORICALS:
                try:
                    if cat == 'pp':
                        ar_data.at[i, cat] = self.categorical_encoder(cat)[
                            segment['ar_' + 'cp']
                        ]
                    else:
                        ar_data.at[i, cat] = self.categorical_encoder(cat)[
                            segment['ar_' + cat]
                        ]
                except KeyError:
                    ar_data.at[i, cat] = -1
                    print(
                        'unknown {cat}: {value}'.format(
                            cat=cat, value=segment['ar_' + cat]
                        )
                    )
                try:
                    if cat == 'pp':
                        dp_data.at[i, cat] = self.categorical_encoder(cat)[
                            segment['dp_' + 'cp']
                        ]
                    else:
                        dp_data.at[i, cat] = self.categorical_encoder(cat)[
                            segment['dp_' + cat]
                        ]
                except KeyError:
                    dp_data.at[i, cat] = -1
                    print(
                        'unknown {cat}: {value}'.format(
                            cat=cat, value=segment['dp_' + cat]
                        )
                    )

            ar_data.at[i, 'lat'] = segment['ar_lat']
            ar_data.at[i, 'lon'] = segment['ar_lon']
            dp_data.at[i, 'lat'] = segment['dp_lat']
            dp_data.at[i, 'lon'] = segment['dp_lon']

            ar_data.at[i, 'stop_id'] = segment['ar_stop_id']
            dp_data.at[i, 'stop_id'] = segment['dp_stop_id']

            ar_data.at[i, 'distance_to_start'] = streckennetz.route_length(
                segment['full_trip'][: ar_data.at[i, 'stop_id'] + 1],
                date=segment['dp_pt'],
            )
            ar_data.at[i, 'distance_to_end'] = streckennetz.route_length(
                segment['full_trip'][ar_data.at[i, 'stop_id'] :], date=segment['dp_pt']
            )
            dp_data.at[i, 'distance_to_start'] = streckennetz.route_length(
                segment['full_trip'][: dp_data.at[i, 'stop_id'] + 1],
                date=segment['dp_pt'],
            )
            dp_data.at[i, 'distance_to_end'] = streckennetz.route_length(
                segment['full_trip'][dp_data.at[i, 'stop_id'] :], date=segment['dp_pt']
            )

            ar_data.at[i, 'minute'] = (
                segment['ar_ct'].time().minute + segment['ar_ct'].time().hour * 60
            )
            ar_data.at[i, 'day'] = segment['ar_ct'].weekday()
            dp_data.at[i, 'minute'] = (
                segment['dp_ct'].time().minute + segment['dp_ct'].time().hour * 60
            )
            dp_data.at[i, 'day'] = segment['dp_ct'].weekday()

            ar_data.at[i, 'stay_time'] = segment['stay_times'][ar_data.at[i, 'stop_id']]
            dp_data.at[i, 'stay_time'] = segment['stay_times'][dp_data.at[i, 'stop_id']]

        return ar_data.astype(self.FEATURE_DTYPES), dp_data.astype(self.FEATURE_DTYPES)


if __name__ == "__main__":
    import helpers.bahn_vorhersage

    from dask.distributed import Client

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
