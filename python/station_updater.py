from datetime import datetime
from typing import Set, Union

import pandas as pd
from redis import Redis
from tqdm import tqdm
from collections.abc import Iterator

from api.iris import IrisStation, search_iris_multiple, stations_equal
from api.ris import stop_place_by_eva
from config import redis_url
from database import cached_table_push
from helpers import StationPhillip


def get_found_iris_stations() -> Iterator[Union[IrisStation, None]]:
    """
    Get all stations from IRIS that were found in the historic delay
    data or already in the database.

    Yields
    ------
    Iterator[Union[IrisStation, None]
        IrisStations that were found in the historic delay data or
        already in the database. None if the station was not found.
    """
    station_names = set(StationPhillip().sta_list)

    redis_client = Redis.from_url(redis_url)

    newly_found_iris_stations = redis_client.smembers('station_names')
    newly_found_iris_stations = set(
        map(lambda x: x.decode('utf-8'), newly_found_iris_stations)
    )

    station_names.update(newly_found_iris_stations)

    yield from search_iris_multiple(station_names)


def add_iris_stations_to_station_phillip(
    iris_stations: Set[IrisStation],
) -> pd.DataFrame:
    """
    Add IRIS stations to StationPhillip. This function is complicated,
    because stations change over time and we want to keep track of the
    changes.

    Parameters
    ----------
    iris_stations : Set[IrisStation]
        IRIS stations to add to StationPhillip

    Returns
    -------
    pd.DataFrame
        Station df from StationPhillip with the added IRIS stations.
    """
    stations = StationPhillip()
    modified_stations = stations.stations.copy()
    modified_stations.reset_index(inplace=True, drop=True)
    new_stations = []

    for iris_station in tqdm(iris_stations, desc='Updating stations'):
        if iris_station is None:
            continue

        try:
            existing_station = stations._get_station(
                eva=iris_station.eva, date='latest'
            ).iloc[0]
            # If station attributes get changed in IRIS, the creation timestamp
            # will not be changed. Thus, the 'latest' station might not be created
            # at the timestamp that is saved in IRIS. Only if the station never
            # got changed, 'valid_from' should be set to the creation timestamp.
            update_valid_from = (
                len(stations._get_station(eva=iris_station.eva, date='all')) == 1
            )
        except KeyError:
            existing_station = None

        if existing_station is None:
            # No old station -> new station -> add new station valid from creation timestamp
            # (the station might be old, but we didn't have it before)
            new_stations.append(iris_station)
        else:
            if update_valid_from:
                if iris_station.creationts > existing_station['valid_from']:
                    modified_stations.at[
                        existing_station['index'], 'valid_from'
                    ] = iris_station.creationts

            if stations_equal(iris_station, existing_station):
                # Station itself didn't change, but some data about the station might have changed
                modified_stations.at[
                    existing_station['index'], 'meta'
                ] = iris_station.meta
                modified_stations.at[existing_station['index'], 'db'] = iris_station.db
            else:
                # Station changed -> old station is valid till now and new modified
                # station is valid from now
                modification_ts = datetime.now()
                modified_stations.at[
                    existing_station['index'], 'valid_to'
                ] = modification_ts

                iris_station.valid_from = modification_ts
                new_stations.append(iris_station)

    modified_stations = pd.concat(
        [modified_stations, pd.DataFrame(new_stations)], ignore_index=True
    )

    # IRIS does not give us any location information for the stations, so
    # new and modified stations will not have any location data. Also, the
    # location of some stations might have changed, so we update all the
    # location data.
    modified_stations = add_ris_info(
        modified_stations, only_update_missing_locations=True
    )

    print(
        f'{len(modified_stations) - len(stations.stations)} stations were added or changed'
    )
    if input('Do you want to upload these to the database? [y/N] ') == 'y':
        modified_stations = modified_stations.drop(columns=['index'])
        cached_table_push(modified_stations, 'stations', fast=False)


def add_ris_info(
    stations: pd.DataFrame, only_update_missing_locations=False
) -> pd.DataFrame:
    """Add RIS information to the stations DataFrame.

    Parameters
    ----------
    stations : pd.DataFrame
        DataFrame containing the stations
    only_update_missing_locations : bool, optional
        Only add info to stations, that do not have a location,
        by default False

    Returns
    -------
    pd.DataFrame
        stations DataFrame with added RIS info
    """    
    if 'available_transports' not in stations.columns:
        stations['available_transports'] = None
    if 'transport_associations' not in stations.columns:
        stations['transport_associations'] = None
    stations = stations.astype(
        {'available_transports': 'object', 'transport_associations': 'object'}
    )

    for eva in tqdm(stations['eva'].unique(), desc='Adding RIS info'):
        if only_update_missing_locations:
            if not stations.loc[stations['eva'] == eva, 'lat'].isna().any():
                continue

        ris_station = stop_place_by_eva(eva)
        if ris_station is None:
            continue

        stations.loc[stations['eva'] == eva, 'station_id'] = ris_station.station_id
        stations.loc[stations['eva'] == eva, 'lat'] = ris_station.lat
        stations.loc[stations['eva'] == eva, 'lon'] = ris_station.lon
        stations.loc[stations['eva'] == eva, 'country'] = ris_station.country
        stations.loc[stations['eva'] == eva, 'state'] = ris_station.state
        stations.loc[stations['eva'] == eva, 'metropolis'] = ris_station.metropolis

        # .loc cannot be used to set the value of a cell to a list, so we have to use .at
        matching_stations_indices = stations[
            (stations['eva'] == eva)
            & (stations['station_id'] == ris_station.station_id)
        ].index
        for index in matching_stations_indices:
            stations.at[
                index, 'available_transports'
            ] = ris_station.available_transports
            stations.at[
                index, 'transport_associations'
            ] = ris_station.transport_associations

    return stations


def main():
    iris_stations = get_found_iris_stations()
    add_iris_stations_to_station_phillip(iris_stations)
    print(iris_stations)


if __name__ == '__main__':
    main()
