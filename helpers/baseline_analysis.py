import os

import pandas as pd
import polars as pl
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             median_absolute_error)

from helpers import RtdRay


def calculate_baseline_regression_long_term(rtd: pd.DataFrame):
    clf = DummyRegressor(strategy='mean')
    xy = rtd['ar_delay'].dropna().to_numpy().astype(int)
    clf.fit(xy, xy)
    mse = mean_squared_error(xy, clf.predict(xy))
    mae = mean_absolute_error(xy, clf.predict(xy))
    medae = median_absolute_error(xy, clf.predict(xy))
    return {
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        'median_absolute_error': medae,
    }


def extract_memsized_dataset_with_full_train_lines(
    rtd: pd.DataFrame, n_lines=1_000
) -> pd.DataFrame:
    dayly_ids = rtd['dayly_id'].unique().compute()
    dayly_ids = set(dayly_ids.sample(n=n_lines, random_state=42))
    rtd = rtd[rtd['dayly_id'].isin(dayly_ids)].compute()
    return rtd


def calculate_baseline_regression_short_term(rtd: pl.DataFrame, prediction_horizon=20):
    rtd = rtd.sort(by=['dayly_id', 'date_id', 'stop_id'])

    return len(rtd)


if __name__ == '__main__':
    from dask.distributed import Client

    # Setting `threads_per_worker` is very important as Dask will otherwise
    # create as many threads as cpu cores which is to munch for big cpus with small RAM
    client = Client(n_workers=min(10, os.cpu_count() // 4), threads_per_worker=2)

    rtd = RtdRay.load_data(
        columns=[
            'ar_delay',
            'dp_delay',
            'ar_ct',
            'ar_pt',
            'dp_ct',
            'dp_pt',
            'dayly_id',
            'date_id',
            'stop_id',
        ]
    )
    # rtd_len = len(rtd)
    # frac_for_3_000_000 = 3_000_000 / rtd_len
    # print(
    #     f'Using about {(frac_for_3_000_000 * 100):.3f}% of {rtd_len} datapoints.'
    # )
    # rtd = rtd.sample(frac=frac_for_3_000_000).compute()
    # print(calculate_baseline_regression_long_term(rtd))

    rtd = extract_memsized_dataset_with_full_train_lines(rtd)
    rtd.to_parquet('cache/baseline_regression_short_term.parquet')

    rtd = pl.read_parquet('cache/baseline_regression_short_term.parquet')
    print(calculate_baseline_regression_short_term(rtd))
