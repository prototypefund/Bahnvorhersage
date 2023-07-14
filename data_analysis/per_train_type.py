import abc

import dask.dataframe as dd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import n_dask_workers
from data_analysis.packed_bubbles import BubbleChart
from database import cached_table_fetch
from helpers import RtdRay, groupby_index_to_flat


class PerCategoryAnalysis(abc.ABC):
    def __init__(self, **kwargs):
        self.data = cached_table_fetch(
            self.tablename,
            index_col=self.category_col,
            table_generator=lambda: self.generate_data(),
            **kwargs,
        )

        self.data = self.group_uncommon(self.data)

    def generate_data(self) -> pd.DataFrame:
        from dask.distributed import Client

        with Client(n_workers=n_dask_workers, threads_per_worker=2) as client:
            rtd = RtdRay.load_data(
                columns=[
                    'c',
                    'o',
                    'f',
                    't',
                    'n',
                    'pp',
                    'ar_pt',
                    'dp_pt',
                    'ar_delay',
                    'ar_happened',
                    'dp_delay',
                    'dp_happened',
                ]
            )
            rtd['c'] = rtd['c'].astype(str)
            rtd['o'] = rtd['o'].astype(str)
            rtd['f'] = rtd['f'].astype(str)
            rtd['t'] = rtd['t'].astype(str)
            rtd['n'] = rtd['n'].astype(str)
            rtd['pp'] = rtd['pp'].astype(str)

            data = (
                rtd.groupby(self.category_col, sort=False)
                .agg(
                    {
                        'ar_delay': ['count', 'mean'],
                        'ar_happened': ['sum'],
                        'dp_delay': ['count', 'mean'],
                        'dp_happened': ['sum'],
                    }
                )
                .compute()
            )

        data = groupby_index_to_flat(data)

        data['ar_happened_mean'] = data['ar_happened_sum'] / data['ar_delay_count']

        data['dp_happened_mean'] = data['dp_happened_sum'] / data['dp_delay_count']

        return data

    @staticmethod
    @abc.abstractmethod
    def group_uncommon(data: pd.DataFrame) -> pd.DataFrame:
        pass

    @property
    @abc.abstractmethod
    def category_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def category_col(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def tablename(self) -> str:
        pass

    def plot_type_delay(self, save_as=None):
        delays = (
            ((self.data['ar_delay_mean'] + self.data['dp_delay_mean']) / 2)
            .to_numpy()
            .astype(float)
        )
        use_trains = np.logical_not(np.isnan(delays))
        delays = delays[use_trains]

        fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
        ax.set_title(f'Ø Verspätung pro {self.category_name}', fontsize=30)
        ax.axis("off")

        type_count = (
            self.data.loc[use_trains, 'ar_delay_count']
            + self.data.loc[use_trains, 'dp_delay_count']
        ).to_numpy()

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", ["green", "yellow", "red"]
        )
        bubble_plot = BubbleChart(area=type_count, bubble_spacing=40)
        bubble_plot.collapse(n_iterations=100)
        bubble_plot.plot(
            ax,
            labels=self.data.loc[use_trains, :].index,
            colors=[
                cmap(delay)
                for delay in (delays - delays.min()) / max(delays - delays.min())
            ],
        )

        # scatter in order to set colorbar
        scatter = ax.scatter(
            np.zeros(len(delays)), np.zeros(len(delays)), s=0, c=delays, cmap=cmap
        )
        colorbar = fig.colorbar(scatter)
        colorbar.solids.set_edgecolor("face")
        colorbar.outline.set_linewidth(0)
        colorbar.ax.get_yaxis().labelpad = 15
        colorbar.ax.set_ylabel("Ø Verspätung in Minuten", rotation=270)

        ax.relim()
        ax.autoscale_view()
        # plt.show()

        if save_as:
            fig.set_size_inches(13.6, 8.5)
            fig.savefig(save_as, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_type_cancellations(self, save_as=None):
        happened = (
            ((self.data['ar_happened_mean'] + self.data['dp_happened_mean']) / 2)
            .to_numpy()
            .astype(float)
        )
        use_trains = np.logical_not(np.isnan(happened))
        happened = happened[use_trains]

        cancellations = 100 - (happened * 100)

        fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
        ax.set_title(f'Prozent ausgefallene Züge je {self.category_name}', fontsize=30)
        ax.axis("off")
        type_count = (
            self.data.loc[use_trains, 'ar_delay_count']
            + self.data.loc[use_trains, 'dp_delay_count']
        ).to_numpy()

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", ["green", "yellow", "red"]
        )
        bubble_plot = BubbleChart(area=type_count, bubble_spacing=40)
        bubble_plot.collapse(n_iterations=100)

        bubble_plot.plot(
            ax,
            labels=self.data.loc[use_trains, :].index,
            colors=[
                cmap(delay)
                for delay in (cancellations - cancellations.min())
                / max(cancellations - cancellations.min())
            ],
        )

        # scatter in order to set colorbar
        scatter = ax.scatter(
            np.zeros(len(cancellations)),
            np.zeros(len(cancellations)),
            s=0,
            c=cancellations,
            cmap=cmap,
        )
        colorbar = fig.colorbar(scatter)
        colorbar.solids.set_edgecolor("face")
        colorbar.outline.set_linewidth(0)
        colorbar.ax.get_yaxis().labelpad = 15
        colorbar.ax.set_ylabel("Prozent ausgefallene Züge", rotation=270)

        ax.relim()
        ax.autoscale_view()
        # plt.show()

        if save_as:
            fig.set_size_inches(13.6, 8.5)
            fig.savefig(save_as, dpi=300, bbox_inches='tight')
        else:
            plt.show()


class TrainTypeAnalysis(PerCategoryAnalysis):
    category_name = 'Zugtyp'
    category_col = 'c'
    tablename = 'per_train_type'

    @staticmethod
    def group_uncommon(data: pd.DataFrame) -> pd.DataFrame:
        # group train types that are uncommon under 'other'
        count = data[('ar_delay_count')] + data[('dp_delay_count')]
        cutoff = count.nsmallest(80).max()
        combine_mask = count <= cutoff
        groups_to_combine = data.loc[combine_mask, :]
        other = data.iloc[0, :].copy()
        for col in groups_to_combine.columns:
            if 'count' in col:
                count_col = col
                count = groups_to_combine[col].sum()
                other.loc[col] = count
            else:
                other.loc[col] = (
                    groups_to_combine[col] * groups_to_combine[count_col]
                ).sum() / count
        data = data.loc[~combine_mask, :].copy()
        data.loc['other', :] = other

        return data


class OperatorAnalysis(PerCategoryAnalysis):
    category_name = 'Beteiber'

    def __init__(self, rtd, **kwargs):
        self.data = cached_table_fetch(
            'per_operator',
            index_col='o',
            table_generator=lambda: self.generate_data(rtd),
            **kwargs,
        )

        self.data = self.group_uncommon(self.data)

    @staticmethod
    def generate_data(rtd: dd.DataFrame) -> pd.DataFrame:
        data = (
            rtd.groupby('o', sort=False)
            .agg(
                {
                    'ar_delay': ['count', 'mean'],
                    'ar_happened': ['mean'],
                    'dp_delay': ['count', 'mean'],
                    'dp_happened': ['mean'],
                }
            )
            .compute()
        )

        data = groupby_index_to_flat(data)

        return data

    @staticmethod
    def group_uncommon(data: pd.DataFrame) -> pd.DataFrame:
        # group train types that are uncommon under 'other'
        count = data[('ar_delay_count')] + data[('dp_delay_count')]
        cutoff = count.nsmallest(200).max()
        combine_mask = count <= cutoff
        groups_to_combine = data.loc[combine_mask, :]
        other = data.iloc[0, :].copy()
        for col in groups_to_combine.columns:
            if 'count' in col:
                count_col = col
                count = groups_to_combine[col].sum()
                other.loc[col] = count
            else:
                other.loc[col] = (
                    groups_to_combine[col] * groups_to_combine[count_col]
                ).sum() / count
        data = data.loc[~combine_mask, :].copy()
        data.loc['other', :] = other

        return data


class fAnalysis(PerCategoryAnalysis):
    category_name = 'f'

    def __init__(self, rtd, **kwargs):
        self.data = cached_table_fetch(
            'per_f',
            table_generator=lambda: self.generate_data(rtd),
            **kwargs,
        )

        self.data = self.group_uncommon(self.data)

    @staticmethod
    def generate_data(rtd: dd.DataFrame) -> pd.DataFrame:
        data = (
            rtd.groupby('f', sort=False)
            .agg(
                {
                    'ar_delay': ['count', 'mean'],
                    'ar_happened': ['mean'],
                    'dp_delay': ['count', 'mean'],
                    'dp_happened': ['mean'],
                }
            )
            .compute()
        )

        data = groupby_index_to_flat(data)

        return data

    @staticmethod
    def group_uncommon(data: pd.DataFrame) -> pd.DataFrame:
        return data


class tAnalysis(PerCategoryAnalysis):
    category_name = 't'

    def __init__(self, rtd, **kwargs):
        self.data = cached_table_fetch(
            'per_t',
            table_generator=lambda: self.generate_data(rtd),
            **kwargs,
        )

        self.data = self.group_uncommon(self.data)

    @staticmethod
    def generate_data(rtd: dd.DataFrame) -> pd.DataFrame:
        data = (
            rtd.groupby('t', sort=False)
            .agg(
                {
                    'ar_delay': ['count', 'mean'],
                    'ar_happened': ['mean'],
                    'dp_delay': ['count', 'mean'],
                    'dp_happened': ['mean'],
                }
            )
            .compute()
        )

        data = groupby_index_to_flat(data)

        return data

    @staticmethod
    def group_uncommon(data: pd.DataFrame) -> pd.DataFrame:
        return data


class TrainNumberAnalysis(PerCategoryAnalysis):
    category_name = 'Zugnummer'

    def __init__(self, rtd, **kwargs):
        self.data = cached_table_fetch(
            'per_train_number',
            table_generator=lambda: self.generate_data(rtd),
            **kwargs,
        )

        self.data = self.group_uncommon(self.data)

    @staticmethod
    def generate_data(rtd: dd.DataFrame) -> pd.DataFrame:
        data = (
            rtd.groupby('n', sort=False)
            .agg(
                {
                    'ar_delay': ['count', 'mean'],
                    'ar_happened': ['mean'],
                    'dp_delay': ['count', 'mean'],
                    'dp_happened': ['mean'],
                }
            )
            .compute()
        )

        data = groupby_index_to_flat(data)

        return data

    @staticmethod
    def group_uncommon(data: pd.DataFrame) -> pd.DataFrame:
        return data


class PlatformAnalysis(PerCategoryAnalysis):
    category_name = 'Gleis'

    def __init__(self, rtd, **kwargs):
        self.data = cached_table_fetch(
            'per_platform',
            table_generator=lambda: self.generate_data(rtd),
            **kwargs,
        )

        self.data = self.group_uncommon(self.data)

    @staticmethod
    def generate_data(rtd: dd.DataFrame) -> pd.DataFrame:
        data = (
            rtd.groupby('pp', sort=False)
            .agg(
                {
                    'ar_delay': ['count', 'mean'],
                    'ar_happened': ['mean'],
                    'dp_delay': ['count', 'mean'],
                    'dp_happened': ['mean'],
                }
            )
            .compute()
        )

        data = groupby_index_to_flat(data)

        return data

    @staticmethod
    def group_uncommon(data: pd.DataFrame) -> pd.DataFrame:
        return data


if __name__ == '__main__':
    import helpers.bahn_vorhersage

    tta = TrainTypeAnalysis(generate=False, prefer_cahce=True)
    tta.plot_type_delay(save_as='delay_per_train_type.png')
    tta.plot_type_cancellations(save_as='cancellations_per_train_type.png')

    # oa = OperatorAnalysis(rtd=rtd, generate=False, prefer_cahce=True)
    # oa.plot_type_delay()
    # oa.plot_type_cancellations()

    # fa = fAnalysis(rtd=rtd, generate=False, prefer_cahce=True)
    # fa.plot_type_delay()
    # fa.plot_type_cancellations()

    # ta = tAnalysis(rtd=rtd, generate=False, prefer_cahce=True)
    # ta.plot_type_delay()
    # ta.plot_type_cancellations()

    # tna = TrainNumberAnalysis(rtd=rtd, generate=False, prefer_cahce=True)
    # # tna.plot_type_delay()
    # # tna.plot_type_cancellations()

    # pa = PlatformAnalysis(generate=False, prefer_cahce=True)
    # pa.plot_type_delay()
    # pa.plot_type_cancellations()
