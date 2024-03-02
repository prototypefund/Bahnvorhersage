import datetime
import os
from typing import Literal

import matplotlib.pyplot as plt
import tueplots.bundles
from dask.distributed import Client

from config import n_dask_workers
from database.cached_table_fetch import cached_table_fetch
from helpers import RtdRay, groupby_index_to_flat


def categories_to_str(categories: set) -> str:
    return '_'.join(list(sorted(categories)))


def load_data(train_categories: set):
    rtd = RtdRay.load_data(
        min_date=datetime.datetime(2021, 9, 1),
        columns=[
            'ar_pt',
            'dp_pt',
            'ar_delay',
            'ar_happened',
            'dp_delay',
            'dp_happened',
            'ar_cs',
            'dp_cs',
            'distance_to_start',
            'c',
        ],
    )

    rtd = rtd.loc[rtd['c'].isin(train_categories), :]

    rtd['ar_on_time'] = rtd['ar_delay'] <= 5
    rtd['dp_on_time'] = rtd['dp_delay'] <= 5
    rtd['ar_canceled'] = rtd['ar_cs'] == 'c'
    rtd['dp_canceled'] = rtd['dp_cs'] == 'c'

    return rtd


class OverDistance:
    tablename = 'over_distance_{train_categories}'

    plot_args = {
        'delay': {
            'ylabel': 'Delay in minutes',
            'title': 'Average delay over distance to train origin',
            'y_col': ['ar_delay_mean', 'dp_delay_mean'],
            'labels': {
                'ar_delay_mean': 'Arrival delay',
                'dp_delay_mean': 'Departure delay',
            },
        },
        'delay_var': {
            'ylabel': 'Delay variance ($minutes^2$)',
            'title': 'Average delay variance over distance to train origin',
            'y_col': ['ar_delay_var', 'dp_delay_var'],
            'labels': {
                'ar_delay_var': 'Arrival delay variance',
                'dp_delay_var': 'Departure delay variance',
            },
        },
        'cancellation': {
            'ylabel': 'Cancellation rate',
            'title': 'Average cancellation rate over distance to train origin',
            'y_col': ['cancellation_rate'],
            'labels': {
                'cancellation_rate': 'Cancellation rate',
            },
        },
    }

    def __init__(self, train_categories: set, **kwargs):
        self.tablename = self.tablename.format(
            train_categories=categories_to_str(train_categories)
        )
        self.data = cached_table_fetch(
            self.tablename,
            table_generator=lambda: self.data_generator(train_categories),
            index_col='index',
            **kwargs,
        )

    def data_generator(self, train_categories: set):
        rtd = load_data(train_categories)

        with Client(n_workers=n_dask_workers, threads_per_worker=2):
            rtd['distance_to_start'] = rtd['distance_to_start'].round(-4).astype(int)
            rtd = (
                rtd.groupby('distance_to_start')
                .agg(
                    {
                        'ar_delay': ['count', 'sum', 'var'],
                        'ar_happened': ['sum'],
                        'ar_on_time': ['sum'],
                        'ar_canceled': ['sum'],
                        'dp_delay': ['count', 'sum', 'var'],
                        'dp_happened': ['sum'],
                        'dp_on_time': ['sum'],
                        'dp_canceled': ['sum'],
                        'distance_to_start': ['first', 'count'],
                    }
                )
                .compute()
            )
            rtd = rtd.loc[~rtd.index.isna(), :]
            rtd = rtd.sort_index()
            rtd = groupby_index_to_flat(rtd)

            rtd['ar_delay_mean'] = rtd['ar_delay_sum'] / rtd['ar_delay_count']
            rtd['ar_happened_mean'] = rtd['ar_happened_sum'] / rtd['ar_delay_count']
            rtd['ar_on_time_mean'] = rtd['ar_on_time_sum'] / rtd['ar_delay_count']
            rtd['ar_canceled_mean'] = rtd['ar_canceled_sum'] / rtd['ar_delay_count']

            rtd['dp_delay_mean'] = rtd['dp_delay_sum'] / rtd['dp_delay_count']
            rtd['dp_happened_mean'] = rtd['dp_happened_sum'] / rtd['dp_delay_count']
            rtd['dp_on_time_mean'] = rtd['dp_on_time_sum'] / rtd['dp_delay_count']
            rtd['dp_canceled_mean'] = rtd['dp_canceled_sum'] / rtd['dp_delay_count']

            rtd['cancellation_rate'] = (
                rtd['ar_canceled_mean'] + rtd['dp_canceled_mean']
            ) / 2

            rtd.rename(columns={'distance_to_start_count': 'stop_count'}, inplace=True)

            return rtd

    def plot(
        self,
        plot_type: Literal['delay', 'delay_var', 'cancellation'] = 'delay',
        save_as: str = None,
    ):
        plt.rcParams.update(tueplots.bundles.icml2022())

        plot_data = self.data.copy()

        # Filter out buckets with less than 100 stops
        plot_data = plot_data.loc[plot_data['stop_count'] > 100, :]

        fig, ax = plt.subplots()
        count_ax = ax.twinx()

        ax.set_zorder(1)  # default zorder is 0 for ax1 and ax2
        ax.patch.set_visible(False)  # prevents ax1 from hiding ax2

        count_ax.stackplot(
            plot_data['distance_to_start'],
            plot_data['ar_delay_count'],
            labels=['Stop count'],
            color='lightgray',
        )

        for y_col in self.plot_args[plot_type]['y_col']:
            plot_data = plot_data.loc[~plot_data[y_col].isna(), :]

        for y_col in self.plot_args[plot_type]['y_col']:
            ax.plot(
                plot_data['distance_to_start'],
                plot_data[y_col],
                label=self.plot_args[plot_type]['labels'][y_col],
            )

        ax.set_xlim(
            plot_data['distance_to_start'].min(), plot_data['distance_to_start'].max()
        )
        ax.set_ylim(0, plot_data[self.plot_args[plot_type]['y_col']].max().max() * 1.1)
        count_ax.set_ylim(0, plot_data['ar_delay_count'].max() * 1.1)

        ax.set_xlabel('Distance to start station (m)')
        ax.set_ylabel(self.plot_args[plot_type]['ylabel'])
        count_ax.set_ylabel('Stop count')
        fig.legend(
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc='lower left',
            mode='expand',
            borderaxespad=0,
            ncol=3,
        )

        if save_as:
            fig.savefig(save_as, dpi=400)
        else:
            plt.show()


def correlation(train_categories: set):
    with Client(n_workers=n_dask_workers, threads_per_worker=2):
        rtd = load_data(train_categories)

        correlation = rtd[['ar_delay', 'dp_delay', 'distance_to_start']].corr(
            method='pearson'
        )
        correlation = correlation.compute()
        print('Pearson correlation for ', categories_to_str(train_categories), ':')
        print(
            'distance_to_start, ar_delay:',
            correlation.loc['distance_to_start', 'ar_delay'],
        )
        print(
            'distance_to_start, dp_delay:',
            correlation.loc['distance_to_start', 'dp_delay'],
        )


def main():
    if not os.path.exists('plots/over_distance'):
        os.makedirs('plots/over_distance')

    train_category_groups = [
        {'ICE'},
        {'IC', 'EC'},
        {'S'},
        {'RE'},
    ]

    for train_categories in train_category_groups:
        # correlation(train_categories)
        over_distance = OverDistance(generate=False, train_categories=train_categories)

        data = over_distance.data[['ar_happened_sum', 'ar_canceled_sum']]
        data['ar_canceled_sum'] = data['ar_canceled_sum'] * 20
        data.plot()

        plt.show()

        # over_distance.plot(
        #     plot_type='delay',
        #     save_as=f'plots/over_distance/delay_{categories_to_str(train_categories)}.png',
        # )
        # over_distance.plot(
        #     plot_type='delay_var',
        #     save_as=f'plots/over_distance/delay_var_{categories_to_str(train_categories)}.png',
        # )
        # over_distance.plot(
        #     plot_type='cancellation',
        #     save_as=f'plots/over_distance/cancellation_{categories_to_str(train_categories)}.png',
        # )


if __name__ == '__main__':
    main()
