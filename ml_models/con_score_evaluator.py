from datetime import datetime, timedelta
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from dask.distributed import Client

from config import n_dask_workers
from helpers.ConnectionRay import ConnectionRay
from ml_models.xgboost_multi_model import Predictor


def scatterplot_trend_line(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    save_as: Optional[str] = None,
):
    fig, ax = plt.subplots()

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=50)

    ax.set_xlim(x.min(), x.max())
    ax.grid(True)

    ax.scatter(x=x, y=y, color="blue", alpha=0.05, edgecolor='none')
    ax.set_ylim(-10, 20)
    # calc the trendline
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(
        [x.min(), x.max()],
        p([x.min(), x.max()]),
        color='red',
        label="y=%.6fx+(%.6f)" % (z[0], z[1]),
    )
    ax.legend(loc='upper left', fontsize=20)
    # the line equation:
    print(title, 'line equation:', "y=%.6fx+(%.6f)" % (z[0], z[1]))

    if save_as:
        fig.set_size_inches(13.6, 7)
        fig.savefig(save_as, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def cor_test(x, y, name: str):
    result = scipy.stats.spearmanr(x, y)
    print('Spearman correlation test result:')
    print('---------------------------------')
    print(f'{name}: {result}')

    result = scipy.stats.pearsonr(x, y)
    print('Pearson correlation test result:')
    print('--------------------------------')
    print(f'{name}: {result}')


def con_score_stats(con_score: np.array, real_transfer_time: np.array):
    # calculate the percentage of connections with at least 2 minutes transfer time
    # within different con_score upper and lower bounds
    for upper, lower in zip(range(0, 100, 10), range(10, 110, 10)):
        upper = upper / 100
        lower = lower / 100
        fitting_transfers = real_transfer_time[
            (con_score >= upper) & (con_score < lower)
        ]
        accuracy = (fitting_transfers >= 2).mean()
        n_data = len(fitting_transfers)
        print(
            f'con_score between {upper} and {lower}: {accuracy * 100:.2f}% ({n_data} data points)'
        )


if __name__ == "__main__":
    with Client(n_workers=n_dask_workers, threads_per_worker=2) as client:
        pred = Predictor()
        con_ray = ConnectionRay()

        ar_rtd = con_ray.load_ar(
            return_times=True,
            obstacles=False,
            min_date=datetime(2022, 9, 1),
            max_date=datetime(2022, 9, 1) + timedelta(days=1),
        )

        fraction = 50_000 / len(ar_rtd)
        indecies_to_use = set(
            (ar_rtd.sample(frac=fraction, random_state=42).compute()).index
        )

        ar_rtd = con_ray.load_ar(
            return_times=True,
            obstacles=False,
        )

        ar_rtd = (
            ar_rtd.drop(columns=['dp_pt', 'dp_ct', 'ar_delay', 'dp_delay'], axis=0)
            .loc[ar_rtd.index.isin(indecies_to_use), :]
            .compute()
        )

        dp_rtd = con_ray.load_dp(return_times=True, obstacles=False)
        dp_rtd = (
            dp_rtd.drop(columns=['ar_pt', 'ar_ct', 'ar_delay', 'dp_delay'], axis=0)
            .loc[dp_rtd.index.isin(indecies_to_use), :]
            .compute()
        )

        transfer_time = (
            ((dp_rtd['dp_pt'] - ar_rtd['ar_pt']) / pd.Timedelta(minutes=1))
            .astype('int')
            .to_numpy()
        )
        real_transfer_time = (
            ((dp_rtd['dp_ct'] - ar_rtd['ar_ct']) / pd.Timedelta(minutes=1))
            .astype('int')
            .to_numpy()
        )

        ar_rtd = ar_rtd.drop(columns=['ar_pt', 'ar_ct'])
        dp_rtd = dp_rtd.drop(columns=['dp_pt', 'dp_ct'])

        ar_pred = pred.predict_ar(ar_rtd)
        dp_pred = pred.predict_dp(dp_rtd)

        con_score = pred.predict_con(ar_pred, dp_pred, transfer_time)

        five_minute_mask = (
            (transfer_time == 5)
            & (real_transfer_time < 60)
            & (real_transfer_time > -60)
        )
        n_indices = np.random.choice(
            np.arange(len(five_minute_mask)), five_minute_mask.sum(), replace=False
        )

        # Connections with 5 min transfer time
        five_minute_con_score = con_score[five_minute_mask]
        five_minute_transfer_time = transfer_time[five_minute_mask]
        five_minute_real_transfer_time = real_transfer_time[five_minute_mask]

        # Connections with 2 - 8 min transfer time, but as many as there are 5 min connections
        con_score = con_score[n_indices]
        transfer_time = transfer_time[n_indices]
        real_transfer_time = real_transfer_time[n_indices]

        print('5 min connections con_score stats:')
        con_score_stats(five_minute_con_score, five_minute_real_transfer_time)

        print('2 - 8 min connections con_score stats:')
        con_score_stats(con_score, real_transfer_time)

        scatterplot_trend_line(
            x=five_minute_con_score,
            y=five_minute_transfer_time,
            xlabel='Verbindungsscore',
            ylabel='Geplante Umsteigezeit in Min.',
            title='5 Minuten Umsteigezeit',
            save_as='con_score_5_min_planned_transfer_time.png',
        )

        scatterplot_trend_line(
            x=five_minute_con_score,
            y=five_minute_real_transfer_time,
            xlabel='Verbindungsscore',
            ylabel='Tatsächliche Umsteigezeit in Min.',
            title='5 Minuten Umsteigezeit',
            save_as='con_score_5_min_real_transfer_time.png',
        )
        cor_test(five_minute_con_score, five_minute_real_transfer_time, '5 min real')

        scatterplot_trend_line(
            x=con_score,
            y=transfer_time,
            xlabel='Verbindungsscore',
            ylabel='Geplante Umsteigezeit in Min.',
            title='2 - 10 Minuten Umsteigezeit',
            save_as='con_score_2_to_10_min_planned_transfer_time.png',
        )
        cor_test(con_score, transfer_time, '2-10 min planned')

        scatterplot_trend_line(
            x=con_score,
            y=real_transfer_time,
            xlabel='Verbindungsscore',
            ylabel='Tatsächliche Umsteigezeit in Min.',
            title='2 - 10 Minuten Umsteigezeit',
            save_as='con_score_2_to_10_min_real_transfer_time.png',
        )
        cor_test(con_score, real_transfer_time, '2-10 min real')
