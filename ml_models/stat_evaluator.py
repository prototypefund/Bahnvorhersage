import json
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

import helpers.colormaps as colormaps


def plot_ar(
    stats: pd.DataFrame, min_minute: int, max_minute: int, save_as: Optional[str] = None
):
    """Plot the accuracy and baseline of AR models for a given range of minutes.

    Parameters
    ----------
    stats : pd.DataFrame
        Dataframe containing the accuracy and baseline of AR models.
    min_minute : int
        Minimum model minute to plot, inclusive.
    max_minute : int
        Maximum model minute to plot, exclusive.
    save_as : Optional[str], optional
        Path where to save the Plot. If None, plot will be shown, by default None
    """
    fig, ax = plt.subplots()
    for minute in range(min_minute, max_minute):
        stats.loc[stats['ar_minute'] == minute, 'ar_accuracy'].plot(
            kind='line',
            color=colormaps.paired(30)(minute * 2),
            linewidth=3,
            ax=ax,
            label=f'AR Minute {minute}',
        )
        stats.loc[stats['ar_minute'] == minute, 'ar_baseline'].plot(
            kind='line',
            color=colormaps.paired(30)(minute * 2 + 1),
            linewidth=3,
            ax=ax,
            label=f'Baseline {minute}',
        )

    ax.set_title(f'AR {min_minute} - {max_minute - 1}', fontsize=50)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_ylabel('Genauigkeit', fontsize=30)
    ax.set_xlabel('Datum', fontsize=30)
    ax.grid()
    ax.set_xlim(stats.index.min(), stats.index.max())
    ax.set_ylim(0.5, 1)
    fig.legend(fontsize=18)
    fig.autofmt_xdate()
    if save_as:
        fig.set_size_inches(13.6, 8.5)
        fig.savefig(save_as, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_dp(
    stats: pd.DataFrame, min_minute: int, max_minute: int, save_as: Optional[str] = None
):
    """Plot the accuracy and baseline of DP models for a given range of minutes.

    Parameters
    ----------
    stats : pd.DataFrame
        Dataframe containing the accuracy and baseline of DP models.
    min_minute : int
        Minimum model minute to plot, inclusive.
    max_minute : int
        Maximum model minute to plot, exclusive.
    save_as : Optional[str], optional
        Path where to save the Plot. If None, plot will be shown, by default None
    """
    fig, ax = plt.subplots()
    for minute in range(min_minute, max_minute):
        stats.loc[stats['dp_minute'] == minute, 'dp_accuracy'].plot(
            kind='line',
            color=colormaps.paired(30)(minute * 2),
            linewidth=3,
            ax=ax,
            label=f'DP Minute {minute}',
        )
        stats.loc[stats['dp_minute'] == minute, 'dp_baseline'].plot(
            kind='line',
            color=colormaps.paired(30)(minute * 2 + 1),
            linewidth=3,
            ax=ax,
            label=f'Baseline {minute}',
        )

    ax.set_title(f'DP {min_minute} - {max_minute - 1}', fontsize=50)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_ylabel('Genauigkeit', fontsize=30)
    ax.set_xlabel('Datum', fontsize=30)
    ax.grid()
    ax.set_xlim(stats.index.min(), stats.index.max())
    ax.set_ylim(0.5, 1)
    fig.legend(fontsize=18)
    fig.autofmt_xdate()
    if save_as:
        fig.set_size_inches(13.6, 8.5)
        fig.savefig(save_as, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_simple_ar(stats: pd.DataFrame, save_as: Optional[str] = None):
    simple_stats = stats.groupby('ar_minute').mean()

    fig, ax = plt.subplots()
    simple_stats['ar_accuracy'].plot(
        kind='line',
        ax=ax,
        linewidth=3,
        label='Model accuracy',
    )
    simple_stats['ar_baseline'].plot(
        kind='line',
        ax=ax,
        linewidth=3,
        label='Baseline',
    )

    ax.set_title('Arrival', fontsize=50)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_ylabel('Accuracy', fontsize=30)
    ax.set_xlabel('Minute', fontsize=30)

    ax.grid()
    ax.set_xlim(0, 14)
    ax.set_ylim(0.5, 1)
    fig.legend(fontsize=18)

    if save_as is not None:
        fig.set_size_inches(13.6, 8.5)
        fig.savefig(save_as, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_simple_dp(stats: pd.DataFrame, save_as: Optional[str] = None):
    simple_stats = stats.groupby('dp_minute').mean()

    fig, ax = plt.subplots()
    simple_stats['dp_accuracy'].plot(
        kind='line',
        ax=ax,
        linewidth=3,
        label='Model accuracy',
    )
    simple_stats['dp_baseline'].plot(
        kind='line',
        ax=ax,
        linewidth=3,
        label='Baseline',
    )

    ax.set_title('Departure', fontsize=50)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_ylabel('Accuracy', fontsize=30)
    ax.set_xlabel('Minute', fontsize=30)

    ax.grid()
    ax.set_xlim(0, 14)
    ax.set_ylim(0.5, 1)
    fig.legend(fontsize=18)

    if save_as is not None:
        fig.set_size_inches(13.6, 8.5)
        fig.savefig(save_as, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    data = json.load(open('cache/test copy.json'))
    parsed = []

    for key in data:
        for minute in range(15):
            minute_model = {}
            for model in data[key]:
                if model['minute'] == minute:
                    minute_model.update(
                        {
                            model['ar_or_dp'] + '_' + k: model[k]
                            for k in model
                            if k != 'ar_or_dp'
                        }
                    )
            parsed.append({'date': key, **minute_model})

    stats = pd.DataFrame(parsed)
    stats['date'] = pd.to_datetime(stats['date'])
    stats = stats.sort_values(by='date')

    stats = stats[~stats['date'].isin({datetime(2022, 5, 31), datetime(2022, 6, 27)})]

    # stats = stats.groupby(['ar_minute']).mean(numeric_only=True)
    # stats.drop(columns=['ar_minute'], inplace=True)
    stats.set_index('date', inplace=True)

    plot_simple_ar(stats, save_as='pred_accu_ar_overview.png')
    plot_simple_dp(stats, save_as='pred_accu_dp_overview.png')

    # plot_ar(stats, 0, 6, save_as='stats_ar_0_6.png')
    # plot_ar(stats, 6, 12, save_as='stats_ar_6_12.png')
    # plot_ar(stats, 12, 15, save_as='stats_ar_12_15.png')

    # plot_dp(stats, 0, 6, save_as='stats_dp_0_6.png')
    # plot_dp(stats, 6, 12, save_as='stats_dp_6_12.png')
    # plot_dp(stats, 12, 15, save_as='stats_dp_12_15.png')
