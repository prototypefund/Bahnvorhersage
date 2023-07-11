import datetime
from typing import List, Literal, Optional, Tuple, Union

import geopy.distance
import pandas as pd

from config import CACHE_TIMEOUT_SECONDS
from database import cached_table_fetch
from helpers import ttl_lru_cache

DateSelector = Union[
    datetime.datetime, List[datetime.datetime], Literal['latest'], Literal['all']
]
AllowedDuplicates = Union[
    Literal['all'], Literal['first'], Literal['last'], Literal['none']
]


class StationPhillip:
    def __init__(self, **kwargs):
        if 'generate' in kwargs:
            kwargs['generate'] = False
            print('StationPhillip does not support generate')
        self.kwargs = kwargs

    @property
    @ttl_lru_cache(CACHE_TIMEOUT_SECONDS, 1)
    def stations(self) -> pd.DataFrame:
        stations = cached_table_fetch('stations', **self.kwargs)
        if 'valid_from' not in stations.columns:
            stations['valid_from'] = pd.NaT
        if 'valid_to' not in stations.columns:
            stations['valid_to'] = pd.NaT
        stations['valid_from'] = stations['valid_from'].fillna(pd.Timestamp.min)
        stations['valid_to'] = stations['valid_to'].fillna(pd.Timestamp.max)
        stations['eva'] = stations['eva'].astype(int)
        stations['name'] = stations['name'].astype(pd.StringDtype())
        stations['ds100'] = stations['ds100'].astype(pd.StringDtype())
        # TODO: parse meta, stations, ... arrays

        stations.set_index(
            ['name', 'eva', 'ds100'],
            drop=False,
            inplace=True,
        )
        stations.index.set_names(['name', 'eva', 'ds100'], inplace=True)
        stations = stations.sort_index()
        return stations

    @property
    @ttl_lru_cache(CACHE_TIMEOUT_SECONDS, 1)
    def name_index_stations(self) -> pd.DataFrame:
        name_index_stations = cached_table_fetch('stations', **self.kwargs).set_index(
            'name'
        )
        name_index_stations['eva'] = name_index_stations['eva'].astype(int)
        return name_index_stations

    @property
    @ttl_lru_cache(CACHE_TIMEOUT_SECONDS, 1)
    def sta_list(self) -> List[str]:
        return list(
            self._get_station(date=datetime.datetime.now())
            .sort_values(by='number_of_events', ascending=False)['name']
            .unique()
        )

    def __len__(self):
        return len(self.stations)

    def __iter__(self):
        """
        Iterate over station names

        Yields
        -------
        str
            Name of station
        """
        yield from self.stations['name'].unique()

    def to_gdf(self, date: DateSelector = None, index_cols: Tuple[str, ...] = None):
        """
        Convert stations to geopandas GeoDataFrame.

        Returns
        -------
        geopandas.GeoDataFrame
            Stations with coordinates as geometry for geopandas.GeoDataFrame.
        """
        import geopandas as gpd

        if date is not None:
            stations = self._get_station(date)
        else:
            stations = self.stations
        if index_cols is None:
            index_cols = ('name', 'eva', 'ds100', 'date')

        for level in index_cols:
            if level not in stations.index.names:
                stations.set_index(level, append=True, inplace=True)
        for level in stations.index.names:
            if level not in index_cols:
                stations = stations.droplevel(level=level)

        return gpd.GeoDataFrame(
            stations,
            geometry=gpd.points_from_xy(stations['lon'], stations['lat']),
        ).set_crs('EPSG:4326')

    @staticmethod
    def _filter_duplicate_station_attributes(
        stations: pd.DataFrame, by: str, allow_duplicates: AllowedDuplicates = 'all'
    ) -> pd.DataFrame:
        """Some stations have the same name for some reason. If one queries for one of these
        stations by name, StationPhillip will return more than one station for these. As this
        can be very impratical, it is possible to drop the duplicates.

        Parameters
        ----------
        stations : pd.DataFrame
            The stations in which a duplicate might exist
        by : str
            The column to find the duplicate in
        allow_duplicates : AllowedDuplicates, optional
            Duplicates to allow. Either `all`, `first`, `last` or `none`, by default 'all'

        Returns
        -------
        pd.DataFrame
            The stations without the duplicates
        """
        if allow_duplicates == 'all':
            return stations
        else:
            return stations.drop_duplicates(subset=by, keep=allow_duplicates)

    @staticmethod
    def _filter_stations_by_date(
        date: DateSelector,
        stations_to_filter: pd.DataFrame,
        drop_duplicates_by: Union[str, List],
        allow_duplicates: AllowedDuplicates = 'all',
    ):
        stations_to_filter = stations_to_filter.sort_index()
        if isinstance(date, str) and date == 'all':
            return stations_to_filter
        elif isinstance(date, str) and date == 'latest':
            # find max date for every station
            date = (
                stations_to_filter['valid_from']
                .groupby(level=stations_to_filter.index.names)
                .max()
            )
            date.name = 'date'
        elif not isinstance(date, datetime.datetime):
            # convert list / etc to dataframe
            date = pd.Series(
                index=stations_to_filter.index.unique(), data=list(date), name='date'
            )

        stations_to_filter = stations_to_filter.loc[
            (date >= stations_to_filter['valid_from'])
            & (date < stations_to_filter['valid_to'])
        ]

        stations_to_filter['date'] = date
        stations_to_filter = StationPhillip._filter_duplicate_station_attributes(
            stations_to_filter, allow_duplicates=allow_duplicates, by=drop_duplicates_by
        )

        stations_to_filter.set_index(['date'], append=True, inplace=True)
        return stations_to_filter

    def _get_station(
        self,
        date: DateSelector,
        name: Union[str, List[str]] = None,
        eva: Union[int, List[int]] = None,
        ds100: Union[str, List[str]] = None,
        allow_duplicates: AllowedDuplicates = 'all',
    ) -> pd.DataFrame:
        if name is not None:
            if isinstance(name, str):
                return self._filter_stations_by_date(
                    date,
                    self.stations.xs(name, level='name'),
                    drop_duplicates_by='name',
                    allow_duplicates=allow_duplicates,
                )
            else:
                stations = self.stations.loc[
                    (
                        name,
                        slice(None),
                        slice(None),
                    ),
                    :,
                ]
                stations = self._filter_stations_by_date(
                    date,
                    stations,
                    drop_duplicates_by='name',
                    allow_duplicates=allow_duplicates,
                )
                stations = stations.droplevel(level=['eva', 'ds100'])

                return stations

        elif eva is not None:
            if isinstance(eva, int):
                return self._filter_stations_by_date(
                    date,
                    self.stations.xs(eva, level='eva'),
                    drop_duplicates_by='eva',
                    allow_duplicates=allow_duplicates,
                )
            else:
                stations = self.stations.loc[
                    (
                        slice(None),
                        eva,
                        slice(None),
                    ),
                    :,
                ]
                stations = self._filter_stations_by_date(
                    date,
                    stations,
                    drop_duplicates_by='eva',
                    allow_duplicates=allow_duplicates,
                )
                stations = stations.droplevel(level=['name', 'ds100'])

                return stations

        elif ds100 is not None:
            if isinstance(ds100, str):
                return self._filter_stations_by_date(
                    date,
                    self.stations.xs(ds100, level='ds100'),
                    drop_duplicates_by='ds100',
                    allow_duplicates=allow_duplicates,
                )
            else:
                stations = self.stations.loc[
                    (
                        slice(None),
                        slice(None),
                        ds100,
                    ),
                    :,
                ]
                stations = self._filter_stations_by_date(
                    date,
                    stations,
                    drop_duplicates_by='ds100',
                    allow_duplicates=allow_duplicates,
                )
                stations = stations.droplevel(level=['name', 'eva'])

                return stations

        else:
            stations = self._filter_stations_by_date(
                date,
                self.stations,
                drop_duplicates_by=['name', 'eva', 'ds100'],
                allow_duplicates=allow_duplicates,
            )
            stations = stations.droplevel(level=['name', 'eva', 'ds100'])
            return stations

    def get_eva(
        self,
        date: DateSelector,
        name: Optional[Union[str, List[str]]] = None,
        ds100: Optional[Union[str, List[str]]] = None,
        allow_duplicates: AllowedDuplicates = 'all',
    ) -> Union[int, pd.Series]:
        """
        Get eva from name or ds100

        Parameters
        ----------
        date : DateSelector
            The date of the stations to get the location for.
            - datetime.datetime : Stations that were active on the given date
            - List[datetime.datetime] : Stations that were active on a given date. Each element of the list is matched to the corresponding element of the eva, name or ds100 list.
            - 'latest' : The latest or current active station
        name : str or ArrayLike[str], optional
            The name or names to get the eva or evas from, by default None
        ds100 : str or ArrayLike[str], optional
            The ds100 or ds100ths to get the eva or evas from, by default None

        Returns
        -------
        int | pd.Series
            int - the single eva matching name or ds100
            pd.Series: Series with the evas matching name or ds100. Contains NaNs
            if no eva was found for a given name or ds100.
        """
        if name is not None and ds100 is not None:
            raise ValueError('Either name or ds100 must be supplied not both')

        eva = self._get_station(
            date=date, name=name, ds100=ds100, allow_duplicates=allow_duplicates
        ).loc[:, 'eva']
        if isinstance(name, str) or isinstance(ds100, str):
            eva = eva.item()

        return eva

    def get_name(
        self,
        date: DateSelector,
        eva: Optional[Union[int, List[int]]] = None,
        ds100: Optional[Union[str, List[str]]] = None,
        allow_duplicates: AllowedDuplicates = 'all',
    ) -> Union[str, pd.Series]:
        """
        Get name from eva or ds100

        Parameters
        ----------
        date : DateSelector
            The date of the stations to get the location for.
            - datetime.datetime : Stations that were active on the given date
            - List[datetime.datetime] : Stations that were active on a given date. Each element of the list is matched to the corresponding element of the eva, name or ds100 list.
            - 'latest' : The latest or current active station
        eva : int or ArrayLike[int], optional
            The eva or evas to get the name or names from, by default None
        ds100 : str or ArrayLike[str], optional
            The ds100 or ds100ths to get the name or names from, by default None

        Returns
        -------
        str | pd.Series
            str - the single name matching eva or ds100
            pd.Series: Series with the names matching eva or ds100. Contains NaNs
            if no name was found for a given eva or ds100.
        """
        if eva is not None and ds100 is not None:
            raise ValueError('Either eva or ds100 must be supplied not both')

        name = self._get_station(
            date=date, eva=eva, ds100=ds100, allow_duplicates=allow_duplicates
        ).loc[:, 'name']
        if isinstance(eva, int) or isinstance(ds100, str):
            name = name.item()

        return name

    def get_ds100(
        self,
        date: DateSelector,
        eva: Optional[Union[int, List[int]]] = None,
        name: Optional[Union[str, List[str]]] = None,
        allow_duplicates: AllowedDuplicates = 'all',
    ) -> Union[str, pd.Series]:
        """
        Get ds100 from eva or name

        Parameters
        ----------
        date : DateSelector
            The date of the stations to get the location for.
            - datetime.datetime : Stations that were active on the given date
            - List[datetime.datetime] : Stations that were active on a given date. Each element of the list is matched to the corresponding element of the eva, name or ds100 list.
            - 'latest' : The latest or current active station
        eva : int or ArrayLike[int], optional
            The eva or evas to get the ds100 or ds100ths from, by default None
        name : str or ArrayLike[str], optional
            The name or names to get the ds100 or ds100ths from, by default None

        Returns
        -------
        str | pd.Series
            str - the single ds100 matching eva or name
            pd.Series: Series with the ds100 matching eva or name. Contains NaNs
            if no ds100 was found for a given eva or name.
        """
        if eva is not None and name is not None:
            raise ValueError('Either eva or name must be supplied not both')

        ds100 = self._get_station(
            date=date, eva=eva, name=name, allow_duplicates=allow_duplicates
        ).loc[:, 'ds100']
        if isinstance(eva, int) or isinstance(name, str):
            ds100 = ds100.item()

        return ds100

    def get_location(
        self,
        date: DateSelector,
        eva: Optional[Union[int, List[int]]] = None,
        name: Optional[Union[str, List[str]]] = None,
        ds100: Optional[Union[str, List[str]]] = None,
        allow_duplicates: AllowedDuplicates = 'all',
    ) -> Union[Tuple[int, int], pd.DataFrame]:
        """
        Get location from eva, name or ds100

        Parameters
        ----------
        date : DateSelector
            The date of the stations to get the location for.
            - datetime.datetime : Stations that were active on the given date
            - List[datetime.datetime] : Stations that were active on a given date. Each element of the list is matched to the corresponding element of the eva, name or ds100 list.
            - 'latest' : The latest or current active station
        eva : int or ArrayLike[int], optional
            The eva or evas to get the location or locations from, by default None
        name : str or ArrayLike[str], optional
            The name or names to get the location or locations from, by default None
        ds100 : str or ArrayLike[str], optional
            The ds100 or ds100ths to get the location or locations from, by default None

        Returns
        -------
        (int, int) | pd.DataFrame
            (int, int) - the single location matching (eva, name or ds100) and date
            pd.DataFrame - DataFrame with the locations matching (eva, name or ds100) and date
        """
        if eva is not None and name is not None and ds100 is not None:
            raise ValueError('Either eva, name or ds100 must be supplied not all')
        elif eva is None and name is None and ds100 is None:
            raise ValueError('Either eva, name or ds100 must be supplied not none')

        location = self._get_station(
            date=date,
            eva=eva,
            name=name,
            ds100=ds100,
            allow_duplicates=allow_duplicates,
        ).loc[:, ['lon', 'lat']]
        if isinstance(eva, int) or isinstance(name, str) or isinstance(ds100, str):
            # Duplicate Station names exist. Thus, location might have several stations in it.
            # In that case, we use the location of the first one.
            if len(location) == 0:
                raise KeyError(
                    f'No location found for station date={date} eva={eva} name={name} ds100={ds100}'
                )
            elif len(location) != 1:
                location = location.iloc[0, :]
            location = (location['lon'].item(), location['lat'].item())

        return location

    def geographic_distance(
        self,
        name1: str,
        name2: str,
        date: DateSelector = None,
    ) -> float:
        coords_1 = self.get_location(name=name1, date=date, allow_duplicates='first')
        coords_2 = self.get_location(name=name2, date=date, allow_duplicates='first')

        return geopy.distance.distance(coords_1, coords_2).meters

    def add_number_of_events(self):
        # TODO: This function does not work anymore. Switch to PerStationOverTime
        from data_analysis.per_station import PerStationAnalysis

        # Fail if cache does not exist
        per_station = PerStationAnalysis(None)

        self.stations['number_of_events'] = (
            per_station.data[('ar_delay', 'count')]
            + per_station.data[('dp_delay', 'count')]
        )


if __name__ == '__main__':
    import helpers.bahn_vorhersage

    stations = StationPhillip(prefer_cache=False)

    print(stations.sta_list[:10])
    # print(
    #     stations._get_station(
    #         name=pd.Series(['Tübingen Hbf', 'Köln Hbf']),
    #         date='latest',
    #     )
    # )
    # # print(stations._get_station(eva=pd.Series([8000141, 8000141]), date=pd.Series([datetime.datetime.now(), datetime.datetime.today()])))

    # print(stations.get_location(eva=8000141, date=datetime.datetime.now()))
    # print(
    #     stations.get_location(
    #         ds100=pd.Series(['TT', 'KK']),
    #         date=pd.Series([datetime.datetime.now(), datetime.datetime.today()]),
    #     )
    # )

    # print('len:', len(stations))
