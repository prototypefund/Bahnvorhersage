import math
import pickle
from collections.abc import Iterable
from datetime import timedelta
from typing import Literal

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config import ENCODER_PATH, JSON_MODEL_PATH
from database.ris_transfer_time import TransferInfo
from helpers.cache import ttl_lru_cache
from helpers.StreckennetzSteffi import StreckennetzSteffi

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


class Predictor:
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
        if category not in CATEGORICALS:
            raise ValueError(
                f'category {category} is unknown. Valid categories are: {CATEGORICALS}'
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
        return np.sort(prediction, axis=1)

    def predict_dp(self, features: pd.DataFrame) -> np.ndarray:
        features = features.to_numpy()
        prediction = np.empty((len(features), self.n_models))
        for model in range(self.n_models):
            prediction[:, model] = Predictor.model(model, 'dp').predict_proba(
                features, validate_features=False
            )[:, 1]
        return -np.sort(-prediction, axis=1)

    def shift_predictions_by_transfer_time(
        self,
        ar_predictions: np.ndarray,
        dp_predictions: np.ndarray,
        transfer_times: np.ndarray,
        needed_transfer_times: Iterable[TransferInfo],
    ) -> tuple[np.ndarray, np.ndarray]:
        for i, (minutes, needed_transfer_time) in enumerate(
            zip(transfer_times, needed_transfer_times)
        ):
            transfer_time = timedelta(minutes=minutes.item())
            transfer_puffer = (
                transfer_time - needed_transfer_time.frequent_traveller.duration
            )

            total_minutes = math.ceil(transfer_puffer.total_seconds() / 60)

            if transfer_puffer == timedelta():
                continue
            elif transfer_puffer < timedelta():
                dp_predictions[i] = np.roll(dp_predictions[i], -total_minutes)
                dp_predictions[i, -total_minutes:] = 0
            elif transfer_puffer > timedelta(minutes=14):
                ar_predictions[i] = np.ones(ar_predictions[i].shape)
                dp_predictions[i] = np.ones(dp_predictions[i].shape)
            else:
                ar_predictions[i] = np.roll(ar_predictions[i], -total_minutes)
                ar_predictions[i, -total_minutes:] = 0

        return ar_predictions, dp_predictions

    def predict_con(
        self,
        ar_prediction: np.ndarray,
        dp_prediction: np.ndarray,
        transfer_times: np.ndarray,
        needed_transfer_times: Iterable[TransferInfo],
    ) -> np.ndarray:
        ar_prediction, dp_prediction = self.shift_predictions_by_transfer_time(
            ar_prediction, dp_prediction, transfer_times, needed_transfer_times
        )

        con_score = np.ones(len(ar_prediction))

        con_score = ar_prediction[:, 0] * dp_prediction[:, 0]

        con_score = con_score + np.sum(
            np.maximum(ar_prediction[:, 1:] - ar_prediction[:, :-1], 0)
            * dp_prediction[:, 1:],
            axis=1,
        )
        # Sometimes, due to inaccuracies in the prediction, the connection score
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
        # dtypes = {
        #     'station': 'int',
        #     'lat': 'float',
        #     'lon': 'float',
        #     'o': 'int',
        #     'c': 'int',
        #     'n': 'int',
        #     'distance_to_start': 'float',
        #     'distance_to_end': 'float',
        #     'pp': 'int',
        #     'stop_id': 'int',
        #     'minute': 'int',
        #     'day': 'int',
        #     'stay_time': 'float',
        # }
        ar_data = pd.DataFrame(
            columns=FEATURES,
            index=range(len(segments)),
        )
        dp_data = ar_data.copy()
        for i, segment in enumerate(segments):
            # Encode categoricals
            for cat in CATEGORICALS:
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

            is_bus = (
                True
                if (segment['ar_c'] == 'bus' or segment['dp_c'] == 'bus')
                else False
            )

            ar_data.at[i, 'distance_to_start'] = streckennetz.route_length(
                segment['full_trip'][: ar_data.at[i, 'stop_id'] + 1],
                is_bus=is_bus,
            )
            ar_data.at[i, 'distance_to_end'] = streckennetz.route_length(
                segment['full_trip'][ar_data.at[i, 'stop_id'] :],
                is_bus=is_bus,
            )
            dp_data.at[i, 'distance_to_start'] = streckennetz.route_length(
                segment['full_trip'][: dp_data.at[i, 'stop_id'] + 1],
                is_bus=is_bus,
            )
            dp_data.at[i, 'distance_to_end'] = streckennetz.route_length(
                segment['full_trip'][dp_data.at[i, 'stop_id'] :],
                is_bus=is_bus,
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

        return ar_data.astype(FEATURE_DTYPES), dp_data.astype(FEATURE_DTYPES)
