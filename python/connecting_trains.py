import concurrent.futures
import datetime
import os
import random

import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client
from tqdm import tqdm

from config import CACHE_PATH
from helpers import RtdRay
from helpers.StationPhillip import StationPhillip


def seperate_station(station_name: str, station_code: int):
    with Client(n_workers=14, threads_per_worker=1):
        rtd = RtdRay.load_for_ml_model(label_encode=True, return_times=True)
        # TODO: this is not tested, but should be way more efficient
        station_rtd = rtd.set_index('station')
        station_rtd.to_parquet(CACHE_PATH + '/station_rtd/', engine='pyarrow')

        # station_rtd = rtd.loc[rtd['station'] == station_code, :].compute()
        # station_rtd.to_parquet(
        #     CACHE_PATH + '/station_rtd/' + station_name + '.parquet',
        #     engine='pyarrow',
        # )


def separate_stations():
    with Client(n_workers=14, threads_per_worker=1):
        rtd = RtdRay.load_for_ml_model(label_encode=True, return_times=True)
        # TODO: this is not tested, but should be way more efficient
        station_rtd = rtd.set_index('station')
        station_rtd.to_parquet(CACHE_PATH + '/station_rtd/', engine='pyarrow')

    # station_encoder = pickle.loads(
    #     open(ENCODER_PATH.format(encoder='station'), 'rb').read()
    # )

    # stations = StationPhillip()
    # for station_name in tqdm(stations.sta_list[177:], desc='separating stations'):
    #     try:
    #         seperate_station(station_name, station_encoder[station_name])
    #     except Exception as e:
    #         print(e)


def get_connecting_trains(df):
    df = df[~df.index.duplicated()]
    df = df.loc[
        (df['ar_pt'] > datetime.datetime(2022, 1, 1))
        | (df['dp_pt'] > datetime.datetime(2022, 1, 1)),
        :,
    ]
    df = df.sort_values(by=['ar_pt', 'dp_pt'])

    min_tranfer_time = datetime.timedelta(minutes=2)
    max_tranfer_time = datetime.timedelta(minutes=10)
    ar = []
    ar_index = []
    dp = []
    dp_index = []
    for i in range(0, len(df), 1000):
        subset = df.iloc[i : i + 1000]
        for index, row in subset.iterrows():
            connecting_trains = subset.loc[
                (subset['dp_pt']).between(
                    row['ar_pt'] + min_tranfer_time, row['ar_pt'] + max_tranfer_time
                ),
                'ar_pt',
            ].index
            for idx in connecting_trains:
                if idx != index:
                    ar_index.append(index)
                    dp_index.append(idx)
    if len(ar_index):
        if len(ar_index) != len(dp_index):
            raise ValueError('ar and dp index not of equal length')

        ar = pd.DataFrame(columns=df.columns, index=ar_index)
        ar.loc[:, :] = df.loc[:, :]
        dp = pd.DataFrame(columns=df.columns, index=dp_index)
        dp.loc[:, :] = df.loc[:, :]
        new_index = (ar.index + dp.index) // 2
        ar.index = new_index
        dp.index = new_index

        ar = ar.astype(df.dtypes)
        dp = dp.astype(df.dtypes)

        if len(ar) != len(dp):
            raise ValueError('ar and dp not of equal length')

        return ar, dp
    else:
        return None, None


def save_connecting_trains(station_name: str, part: int):
    try:
        load_path = CACHE_PATH + f'/station_rtd/{station_name}.parquet'
        ar, dp = get_connecting_trains(pd.read_parquet(load_path, engine='pyarrow'))
        if ar is not None:
            ar_path = CACHE_PATH + '/connecting_trains_{}/part.{}.parquet'.format(
                'ar', part
            )
            dp_path = CACHE_PATH + '/connecting_trains_{}/part.{}.parquet'.format(
                'dp', part
            )
            ar.to_parquet(ar_path, engine='pyarrow')
            dp.to_parquet(dp_path, engine='pyarrow')
        print(f'Saved {station_name}')
    except FileNotFoundError:
        pass
    except Exception as e:
        print(e)


def repartition():
    # Note: This might not be neccessary
    # Re-partition, so that each partition is 256 MiB. Otherwise some partitions
    # might be too large
    ddf: dd.DataFrame = dd.read_parquet(
        CACHE_PATH + '/connecting_trains_ar', engine='pyarrow'
    )
    ddf = ddf.repartition(partition_size='256 MiB')
    ddf.to_parquet(CACHE_PATH + '/ar_connections', engine='pyarrow')

    ddf: dd.DataFrame = dd.read_parquet(
        CACHE_PATH + '/connecting_trains_dp', engine='pyarrow'
    )
    ddf = ddf.repartition(partition_size='256 MiB')
    ddf.to_parquet(CACHE_PATH + '/dp_connections', engine='pyarrow')


if __name__ == '__main__':
    from helpers.bahn_vorhersage import COLORFUL_ART

    print(COLORFUL_ART)

    separate_stations()

    # station_rtd = pd.read_parquet(CACHE_PATH + '/station_rtd/part.0.parquet')
    # get_connecting_trains(station_rtd)

    newpath = CACHE_PATH + '/connecting_trains_ar'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    newpath = CACHE_PATH + '/connecting_trains_dp'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    stations = StationPhillip()
    # save_connecting_trains(stations.sta_list[100], 100)

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        # Put the stations in a random order, so that there are less groups of
        # big stations with similar names (München Hbf, München Pasing, ...)
        random_stations = stations.sta_list
        random.shuffle(random_stations)
        futures = {
            executor.submit(save_connecting_trains, station, i)
            for i, station in enumerate(random_stations)
        }
        for _ in tqdm(
            concurrent.futures.as_completed(futures),
            desc='Finding connecting trains',
            total=len(stations),
        ):
            pass
        # for _ in tqdm(
        #     executor.map(save_connecting_trains, stations, range(len(stations))),
        #     desc='Finding connecting trains',
        #     total=len(stations),
        # ):
        #     pass

    # ar = dd.read_parquet(
    #     CACHE_PATH + '/connecting_trains_ar', engine='pyarrow'
    # ).compute()
    # ar = dd.from_pandas(ar, 100, sort=False)
    # ar.to_parquet(CACHE_PATH + '/ar_connections', engine='pyarrow')

    # dp = dd.read_parquet(
    #     CACHE_PATH + '/connecting_trains_dp', engine='pyarrow'
    # ).compute()
    # dp = dd.from_pandas(dp, 100, sort=False)
    # dp.to_parquet(CACHE_PATH + '/dp_connections', engine='pyarrow')

    print(
        'len arrival connecting trains: ',
        len(dd.read_parquet(CACHE_PATH + '/connecting_trains_ar', engine='pyarrow')),
    )
    print(
        'len departure connecting trains: ',
        len(dd.read_parquet(CACHE_PATH + '/connecting_trains_dp', engine='pyarrow')),
    )
