from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.colors import ListedColormap

from config import n_dask_workers
from database import cached_table_fetch
from helpers import RtdRay, groupby_index_to_flat


class DelayAnalysis:
    def __init__(self, min_delay=0, max_delay=20, **kwargs):
        self.data = cached_table_fetch(
            self.tablename,
            table_generator=self.generate_data,
            index_col='index',
            **kwargs,
        )
        self.data = self.data[
            (self.data.index >= min_delay) & (self.data.index <= max_delay)
        ]

    @property
    def tablename(self):
        return 'per_delay'

    def generate_data(self):
        rtd = RtdRay.load_data(
            columns=[
                'ar_delay',
                'dp_delay',
                'ar_pt',
                'dp_pt',
                'ar_happened',
                'dp_happened',
            ],
            long_distance_only=False,
        )
        from dask.distributed import Client

        with Client(n_workers=n_dask_workers, threads_per_worker=2) as client:
            rtd = (
                rtd.groupby('ar_delay')
                .agg(
                    {
                        'ar_pt': ['count'],
                        'ar_happened': ['sum'],
                        'dp_pt': ['count'],
                        'dp_happened': ['sum'],
                    }
                )
                .compute()
            )
            rtd = rtd.loc[~rtd.index.isna(), :]
            rtd = rtd.sort_index()
            rtd = rtd[rtd.index >= 0]
            rtd = groupby_index_to_flat(rtd)

            ar_count_sum = rtd['ar_pt_count'].sum()
            dp_count_sum = rtd['dp_pt_count'].sum()

            rtd['ar'] = rtd['ar_pt_count'] / ar_count_sum
            rtd['dp'] = rtd['dp_pt_count'] / dp_count_sum

            rtd['ar_cancellation'] = 1 - rtd['ar_happened_sum'] / rtd['ar_pt_count']
            rtd['dp_cancellation'] = 1 - rtd['dp_happened_sum'] / rtd['dp_pt_count']
        return rtd

    def plot(self, loggy=False, save_as: Optional[str] = None,):
        cols = ['ar', 'dp', 'ar_cancellation', 'dp_cancellation']
        fig, ax1 = plt.subplots()

        self.data[cols].plot(
            kind='line',
            colormap=ListedColormap(sns.color_palette('Paired', n_colors=len(cols))),
            ax=ax1,
            linewidth=3,
            legend=False,
        )

        ax1.tick_params(axis="both", labelsize=20)
        index = self.data.index.to_numpy()
        ax1.set_xlim(index.min(), index.max())
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins='auto', integer=True))
        if loggy:
            ax1.set_yscale('log')
        else:
            ax1.set_ylim(bottom=0)

        ax1.grid(True)

        ax1.set_title('Delay distribution', fontsize=50)
        ax1.set_xlabel('Delay in minutes', fontsize=30)
        ax1.set_ylabel('Probability density', fontsize=30)

        fig.legend(fontsize=20)
        
        if save_as:
            fig.set_size_inches(13.6, 8.5)
            fig.savefig(save_as, dpi=300, bbox_inches='tight')
        else:
            plt.show()


if __name__ == '__main__':
    import helpers.bahn_vorhersage

    delay = DelayAnalysis(prefere_cache=True, generate=False)
    delay.plot(save_as='per_delay_fern.png', loggy=False)
