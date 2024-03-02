from datetime import datetime, timedelta

import matplotlib.dates as mdates
import pandas as pd

from data_analysis.over_time import OverTime, add_rolling_mean
from helpers import RtdRay, groupby_index_to_flat
from ml_models.xgboost_multi_model import Predictor, split_ar_dp, train_models

date = datetime(2022, 9, 1)
predictor = Predictor()


class OverDayPrediction(OverTime):
    tablename = 'over_day_prediction'
    resolution = '10min'
    rolling_mean_window = 5
    dt_formatter = mdates.DateFormatter('%H:%M')
    dt_locator = mdates.HourLocator()
    plot_title = 'Stops over the day'
    plot_x_label = 'Time'

    def periodic_aggregator(self, rtd: pd.Series) -> pd.Series:
        return (
            datetime(year=2000, month=1, day=3)
            + pd.to_timedelta(rtd.dt.hour, unit='h')
            + pd.to_timedelta(rtd.dt.minute, unit='m')
        )

    def generate_data(self):
        ml_rtd = RtdRay.load_for_ml_model(
            max_date=date,
            min_date=date - timedelta(days=1),
            return_status=True,
            obstacles=False,
        ).compute()

        ar_ml_rtd, dp_ml_rtd = split_ar_dp(ml_rtd)

        del ar_ml_rtd['ar_delay']
        del ar_ml_rtd['dp_delay']
        del ar_ml_rtd['ar_cs']
        del ar_ml_rtd['dp_cs']

        del dp_ml_rtd['ar_delay']
        del dp_ml_rtd['dp_delay']
        del dp_ml_rtd['ar_cs']
        del dp_ml_rtd['dp_cs']

        ar_on_time = predictor.predict_ar(ar_ml_rtd)
        dp_on_time = predictor.predict_dp(dp_ml_rtd)

        rtd = RtdRay.load_data(
            columns=[
                'ar_pt',
                'dp_pt',
                'ar_delay',
                'ar_happened',
                'dp_delay',
                'dp_happened',
                'ar_cs',
                'dp_cs',
            ],
            max_date=date,
            min_date=date - timedelta(days=1),
        ).compute()

        rtd['ar_on_time'] = 0.0
        rtd['dp_on_time'] = 0.0

        rtd['ar_on_time'] = pd.Series(data=ar_on_time[:, 5], index=ar_ml_rtd.index).loc[
            :
        ]
        rtd['dp_on_time'] = pd.Series(
            data=1 - dp_on_time[:, 6] - 0.0274, index=dp_ml_rtd.index
        ).loc[:]

        rtd['ar_canceled'] = rtd['ar_cs'] == 'c'
        rtd['dp_canceled'] = rtd['dp_cs'] == 'c'

        rtd['agg_time'] = rtd['ar_pt'].fillna(value=rtd['dp_pt'])
        rtd['agg_time'] = rtd['agg_time'].dt.floor(self.resolution)
        rtd['agg_time'] = self.periodic_aggregator(rtd['agg_time'])

        rtd = rtd.groupby('agg_time').agg(
            {
                'ar_delay': ['count', 'mean'],
                'ar_happened': ['mean', 'sum'],
                'ar_on_time': ['mean'],
                'ar_canceled': ['mean'],
                'dp_delay': ['count', 'mean'],
                'dp_happened': ['mean', 'sum'],
                'dp_on_time': ['mean'],
                'dp_canceled': ['mean'],
            }
        )
        rtd = rtd.loc[~rtd.index.isna(), :]
        rtd = rtd.sort_index()
        rtd = groupby_index_to_flat(rtd)

        rtd = rtd.astype(
            {
                'ar_delay_mean': 'float64',
                'ar_delay_count': 'int',
                'ar_happened_mean': 'float64',
                'ar_happened_sum': 'int',
                'ar_on_time_mean': 'float64',
                'ar_canceled_mean': 'float64',
                'dp_delay_mean': 'float64',
                'dp_delay_count': 'int',
                'dp_happened_mean': 'float64',
                'dp_happened_sum': 'int',
                'dp_on_time_mean': 'float64',
                'dp_canceled_mean': 'float64',
            }
        )

        if self.rolling_mean_window > 0:
            rtd = add_rolling_mean(
                rtd,
                [
                    'ar_delay_mean',
                    'ar_delay_count',
                    'ar_happened_mean',
                    'ar_happened_sum',
                    'ar_on_time_mean',
                    'ar_canceled_mean',
                    'dp_delay_mean',
                    'dp_delay_count',
                    'dp_happened_mean',
                    'dp_happened_sum',
                    'dp_on_time_mean',
                    'dp_canceled_mean',
                ],
                window=self.rolling_mean_window,
            )

        return rtd


if __name__ == '__main__':
    from helpers.bahn_vorhersage import COLORFUL_ART

    print(COLORFUL_ART)

    print('Training new ml models...')
    train_models(
        min_date=date - timedelta(days=7 * 6),
        max_date=date,
        return_status=True,
        obstacles=False,
    )

    print('grouping over day')
    time = OverDayPrediction(generate=True)
    time.plot(save_as='over_day_predictions.png', kind='on_time_percentage')
