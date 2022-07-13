import pickle
from typing import Literal
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from config import ENCODER_PATH, JSON_MODEL_PATH
from helpers import ttl_lru_cache


class Predictor:
    CATEGORICALS = ['o', 'c', 'n', 'station', 'pp']

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
    def model(self, minute: int, ar_or_dp: Literal['ar', 'dp']) -> XGBClassifier:
        """Load trained XGBClassifier model for prediction either form disk or time sensitive cache

        Parameters
        ----------
        minute : int
            The minute of delay that the model should predict
        ar_or_dp : str : `ar` | `dp`
            Whether the model should predict arrival or departure delays

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
        return booster

    def predict_ar(self, features: pd.DataFrame) -> np.ndarray:
        features = features.to_numpy()
        prediction = np.empty((len(features), self.n_models))
        for model in range(self.n_models):
            prediction[:, model] = self.model(model, 'ar').predict_proba(
                features, validate_features=False
            )[:, 1]
        return prediction

    def predict_dp(self, features: pd.DataFrame) -> np.ndarray:
        features = features.to_numpy()
        prediction = np.empty((len(features), self.n_models))
        for model in range(self.n_models):
            prediction[:, model] = self.model(model, 'dp').predict_proba(
                features, validate_features=False
            )[:, 1]
        return prediction

    def predict_con(
        self,
        ar_prediction: np.ndarray,
        dp_prediction: np.ndarray,
        transfer_time: np.ndarray,
    ) -> np.ndarray:
        con_score = np.ones(len(transfer_time))

        for tra_time in range(self.n_models):
            mask = transfer_time == tra_time
            if mask.any():
                con_score[mask] = (
                    ar_prediction[mask, max(tra_time - 2, 0)]
                    * dp_prediction[mask, max(0, 2 - tra_time)]
                )
                con_score[mask] = con_score[mask] + np.sum(
                    (
                        ar_prediction[
                            mask,
                            max(tra_time - 2, 0)
                            + 1 : dp_prediction.shape[1]
                            - max(0, 2 - tra_time),
                        ]
                        - ar_prediction[
                            mask,
                            max(tra_time - 2, 0) : dp_prediction.shape[1]
                            - 1
                            - max(0, 2 - tra_time),
                        ]
                    )
                    * dp_prediction[
                        mask,
                        max(0, 2 - tra_time)
                        + 1 : dp_prediction.shape[1]
                        + min(2 - tra_time, 0),
                    ],
                    axis=1,
                )
        return np.minimum(con_score, np.ones(len(con_score)))

    def get_pred_data(self, segments: list, streckennetz) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            columns=[
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
            ],
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

        return ar_data.astype(dtypes), dp_data.astype(dtypes)


if __name__ == '__main__':
    pred = Predictor()
