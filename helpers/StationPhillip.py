import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.cached_table_fetch import cached_table_fetch
import pandas as pd
from typing import Tuple


class StationPhillip:
    def __init__(self, **kwargs):
        self.name_index_stations = cached_table_fetch('stations', index_col='name', **kwargs)
        self.name_index_stations['eva'] = self.name_index_stations['eva'].astype(int)

        self.eva_index_stations = self.name_index_stations.reset_index().set_index('eva')
        self.ds100_index_stations = self.name_index_stations.reset_index().set_index('ds100')
        self.sta_list = self.name_index_stations.sort_values(by='number_of_events', ascending=False).index.to_list()

    def __len__(self):
        return len(self.name_index_stations)

    def __iter__(self):
        """
        Iterate over station names

        Yields
        -------
        str
            Name of station
        """
        yield from self.name_index_stations.index

    def get_geopandas(self):
        """
        Convert stations to geopandas DataFrame.

        Returns
        -------
        geopandas.DateFrame
            Stations with coordinates as geometry for geopandas.DataFrame.
        """
        import geopandas as gpd
        return gpd.GeoDataFrame(
            self.name_index_stations,
            geometry=gpd.points_from_xy(self.name_index_stations.lon, self.name_index_stations.lat)
        ).set_crs("EPSG:4326")

    def get_eva(self, name=None, ds100=None):
        """
        Get the eva from name or ds100.

        Parameters
        ----------
        name : str, optional
            Official station name, by default None
        ds100 : str, optional
            ds100 of station, by default None

        Returns
        -------
        int
            Eva of station

        """
        if name is not None:
            return self.name_index_stations.at[name, 'eva']
        elif ds100 is not None:
            return self.ds100_index_stations.at[ds100, 'eva']

    def get_name(self, eva=None, ds100=None):
        """
        Get the name from eva or ds100.

        Parameters
        ----------
        eva : int, optional
            eva of station, by default None
        ds100 : str, optional
            ds100 of station, by default None

        Returns
        -------
        str
            official station name
        """
        if eva is not None:
            return self.eva_index_stations.at[eva, 'name']
        elif ds100 is not None:
            return self.ds100_index_stations.at[ds100, 'name']

    def get_ds100(self, name=None, eva=None):
        """
        Get the ds100 from eva or station name.

        Parameters
        ----------
        name : str, optional
            Official station name, by default None
        eva : int, optional
            eva of station, by default None

        Returns
        -------
        str
            ds100 of station
        """
        if name is not None:
            return self.name_index_stations.at[name, 'ds100']
        elif eva is not None:
            return self.eva_index_stations.at[eva, 'ds100']

    def get_location(self, name=None, eva=None, ds100=None) -> Tuple[float, float]:
        """
        Get the location (lon, lat) of a station.

        Parameters
        ----------
        name : str, optional
            Official station name, by default None
        eva : int, optional
            eva of station, by default None
        ds100 : str, optional
            ds100 of station, by default None

        Returns
        -------
        tuple(float, float)
            longitude and latitide of station
        """
        if eva is None:
            return self.get_location(eva=self.get_eva(name=name, ds100=ds100))
        else:
            return (self.eva_index_stations.at[eva, 'lon'],
                    self.eva_index_stations.at[eva, 'lat'])

    @staticmethod
    def search_station(search_term):
        import requests
        search_term = search_term.replace('/', ' ')
        matches = requests.get(f'https://marudor.de/api/hafas/v1/station/{search_term}').json()
        return matches

    @staticmethod
    def search_iris(search_term):
        import requests
        from rtd_crawler.xml_parser import xml_to_json
        import lxml.etree as etree

        search_term = search_term.replace('/', ' ')
        matches = requests.get(f'http://iris.noncd.db.de/iris-tts/timetable/station/{search_term}').text
        matches = etree.fromstring(matches.encode())
        matches = list(xml_to_json(match) for match in matches)
        return matches

    def read_stations_from_derf_Travel_Status_DE_IRIS(self):
        import requests

        derf_stations = requests.get(
            'https://raw.githubusercontent.com/derf/Travel-Status-DE-IRIS/master/share/stations.json'
        ).json()
        parsed_derf_stations = {
            'name': [],
            'eva': [],
            'ds100': [],
            'lat': [],
            'lon': [],
        }
        for station in derf_stations:
            parsed_derf_stations['name'].append(station['name'])
            parsed_derf_stations['eva'].append(station['eva'])
            parsed_derf_stations['ds100'].append(station['ds100'])
            parsed_derf_stations['lat'].append(station['latlong'][0])
            parsed_derf_stations['lon'].append(station['latlong'][1])
        return pd.DataFrame(parsed_derf_stations)

    def add_number_of_events(self):
        from data_analysis.per_station import PerStationAnalysis
        # Fail if cache does not exist
        per_station = PerStationAnalysis(None)

        self.name_index_stations['number_of_events'] = (
            per_station.data[('ar_delay', 'count')] + per_station.data[('dp_delay', 'count')]
        )

    def push_to_db(self):
        from database.engine import DB_CONNECT_STRING
        self.name_index_stations.to_sql('stations', DB_CONNECT_STRING, if_exists='replace', method='multi')


if __name__ == "__main__":
    import helpers.fancy_print_tcp
    stations = StationPhillip()

    stations.add_number_of_events()
    stations.push_to_db()

    print('len:', len(stations))
