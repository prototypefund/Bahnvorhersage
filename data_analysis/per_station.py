import datetime
import hashlib
import io
import os
from dataclasses import dataclass

import cartopy
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry
from matplotlib import colors
from PIL import Image
from shapely.ops import clip_by_rect

from config import CACHE_PATH, n_dask_workers
from database.cached_table_fetch import cached_table_fetch
from helpers import RtdRay, groupby_index_to_flat
from helpers.cache import ttl_lru_cache
from helpers.StationPhillip import StationPhillip

matplotlib.use('Agg')
plt.style.use('dark_background')
# Configure cache path in kubernetes
if os.path.isdir('/usr/src/app/cache'):
    cartopy.config['data_dir'] = '/usr/src/app/cache'


def image_to_webp(buffer: io.BytesIO, path: str) -> None:
    """Convert image to webp format and save it to path

    Parameters
    ----------
    buffer: io.BytesIO
        Buffer containing image data
    path: str
        Path to save image to. The path must exist.
    """
    image = Image.open(buffer)
    image = image.convert('RGBA')
    image.save(path, 'webp')


# Bounding Box of Germany
MIN_LON = 5.67
MAX_LON = 15.64
MIN_LAT = 47.06
MAX_LAT = 55.06

BBOX_GERMANY = (MIN_LON, MAX_LON, MIN_LAT, MAX_LAT)


def create_base_plot(
    crs: cartopy.crs.CRS,
    bbox: tuple[float, float, float, float],
    color_scheme: str = 'dark',
) -> tuple[plt.Figure, plt.Axes]:
    """Create pretty geo base plot with given crs and bbox.

    Parameters
    ----------
    crs: cartopy.crs.CRS
        Coordinate reference system for the plot.
    bbox: Tuple[float, float, float, float]
        Bounding box of the plot. Must be in Geodetic coordinates.
        (min_lon, max_lon, min_lat, max_lat)

    Returns
    -------
    plt.Figure, plt.Axes
        Figure and axes of the plot.
    """
    if color_scheme == 'dark':
        colors = {
            'background': '#191a1a',
            'land': '#343332',
            'land_edge': '#5c5b5b',
            'states': '#444444',
            'state_borders': '#5c5b5b',
        }
    elif color_scheme == 'light':
        colors = {
            'background': '#ffffff',
            'land': (240 / 256, 240 / 256, 220 / 256),
            'land_edge': '#adb5bd',
            'states': 'lightgray',
            'state_borders': '#adb5bd',
        }
    else:
        raise ValueError(f'Unknown color scheme {color_scheme}')

    fig, ax = plt.subplots(subplot_kw={'projection': crs})

    ax.set_extent(bbox, crs=ccrs.Geodetic())

    # Add natural earth features to the base plot. We don't use ax.add_feature here,
    # because it doesn't support shape clipping and therefore is way slower.
    # https://stackoverflow.com/questions/61157293/speeding-up-high-res-coastline-plotting-with-cartopy

    ax.add_geometries(
        [shapely.geometry.box(*bbox)],
        crs=ccrs.PlateCarree(),
        facecolor=colors['background'],
    )

    # ax.add_feature(cartopy.feature.LAND, facecolor=colors['land'], edgecolor=colors['land_edge'])
    land = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m')
    land = [clip_by_rect(geom, *bbox) for geom in land.geometries() if geom is not None]
    ax.add_geometries(
        land,
        crs=ccrs.PlateCarree(),
        facecolor=colors['land'],
        edgecolor=colors['land_edge'],
    )

    # ax.add_feature(cartopy.feature.STATES, edgecolor=colors['states'])
    states = cartopy.feature.NaturalEarthFeature(
        'cultural', 'admin_1_states_provinces_lakes', '10m'
    )
    states = [
        clip_by_rect(geom, *bbox) for geom in states.geometries() if geom is not None
    ]
    ax.add_geometries(
        states,
        crs=ccrs.PlateCarree(),
        facecolor=colors['land'],
        edgecolor=colors['states'],
    )

    # ax.add_feature(cartopy.feature.BORDERS, edgecolor=colors['land_edge'])
    borders = cartopy.feature.NaturalEarthFeature(
        'cultural', 'admin_0_boundary_lines_land', '10m'
    )
    borders = [
        clip_by_rect(geom, *bbox) for geom in borders.geometries() if geom is not None
    ]
    ax.add_geometries(
        borders,
        crs=ccrs.PlateCarree(),
        facecolor=colors['land'],
        edgecolor=colors['land_edge'],
    )

    ax.set_axis_off()
    return fig, ax


class PerStationOverTime(StationPhillip):
    DEFAULT_PLOTS = ['no data available', 'default']
    MAP_CRS = ccrs.Mercator(
        central_longitude=MAX_LON - (MAX_LON - MIN_LON) / 2,
        min_latitude=MIN_LAT,
        max_latitude=MAX_LAT,
    )
    PLOTS_DIR = f'{CACHE_PATH}/plots/'
    PLOT_PATH = os.path.join(PLOTS_DIR, '{version}_{title}.webp')

    version: str
    rtd: pd.DataFrame | None
    kwargs: dict
    data: pd.DataFrame

    @dataclass
    class Limits:
        freq_hours: int
        max: datetime.datetime = None
        min: datetime.datetime = None

        @property
        def freq(self) -> str:
            return str(self.freq_hours) + 'H'

    limits = Limits(freq_hours=int(24 * 7))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not os.path.exists(self.PLOTS_DIR):
            os.mkdir(self.PLOTS_DIR)

        self.kwargs = kwargs

        self.data_loader()

        # Setup Plot https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot
        self.fig, self.ax = create_base_plot(crs=self.MAP_CRS, bbox=BBOX_GERMANY)

        self.cmap = colors.LinearSegmentedColormap.from_list(
            '', ['green', 'yellow', 'red']
        )

        self.sc = self.ax.scatter(
            np.zeros(1),
            np.zeros(1),
            marker='o',
            edgecolors='none',
            c=np.zeros(1),
            s=np.zeros(1),
            cmap=self.cmap,
            vmin=0,
            vmax=7,
            alpha=0.5,
            zorder=10,
            transform=self.MAP_CRS,
        )

        self.colorbar = self.fig.colorbar(self.sc)
        self.colorbar.solids.set_edgecolor('face')
        self.colorbar.outline.set_linewidth(0)

        self.colorbar.ax.get_yaxis().labelpad = 15
        self.colorbar.ax.set_ylabel('Ø Verspätung in Minuten', rotation=270)

    def data_generator(self) -> pd.DataFrame:
        rtd = RtdRay.load_data(
            columns=[
                'ar_pt',
                'dp_pt',
                'station',
                'ar_delay',
                'ar_happened',
                'dp_delay',
                'dp_happened',
                'lat',
                'lon',
            ],
            min_date=datetime.datetime(2021, 9, 1),
        )

        # Use dask Client to do groupby as the groupby is complex and scales well on local cluster.
        from dask.distributed import Client

        with Client(n_workers=n_dask_workers, threads_per_worker=1):
            # Generate an index with self.limits.freq for groupby over time and station
            rtd['stop_hour'] = (
                rtd['ar_pt'].fillna(value=rtd['dp_pt']).dt.round(self.limits.freq)
            )

            rtd['single_index_for_groupby'] = (
                rtd['stop_hour'].astype('str') + rtd['station'].astype('str')
            ).apply(hash, meta=(None, 'int64'))

            # Label encode station, as this speeds up the groupby tremendously (10 minutes
            # instead of >24h)
            rtd['station'] = rtd['station'].cat.codes
            rtd = rtd.drop(columns=['ar_pt', 'dp_pt'])

            data: pd.DataFrame = (
                rtd.groupby('single_index_for_groupby', sort=False)
                .agg(
                    {
                        'ar_delay': ['mean'],
                        'ar_happened': ['sum'],
                        'dp_delay': ['mean'],
                        'dp_happened': ['sum'],
                        'stop_hour': ['first'],
                        'station': ['first'],
                        'lat': ['first'],
                        'lon': ['first'],
                    }
                )
                .compute()
            )
            del rtd

        data = groupby_index_to_flat(data)
        # remove rows where ar_happened_sum is 0 and dp_happened_sum is 0
        data = data.loc[(data['ar_happened_sum'] > 0) | (data['dp_happened_sum'] > 0)]
        data_types = {
            'ar_delay_mean': 'float16',
            'ar_happened_sum': 'int32',
            'dp_delay_mean': 'float16',
            'dp_happened_sum': 'int32',
            'stop_hour': 'datetime64[ns]',
            'station': 'int16',
            'lat': 'float64',
            'lon': 'float64',
        }
        data = data.astype(data_types)
        return data

    @ttl_lru_cache(seconds_to_live=60 * 60 * 4)
    def data_loader(self):
        self.data = cached_table_fetch(
            'per_station_over_time',
            table_generator=self.data_generator,
            **self.kwargs,
        )

        # The cache from an older version of this class on potentially older data should
        # not be used. Thus, we create a hash aka version ot the data that is attached
        # to the filenames in the cache.
        self.version = hashlib.sha256(
            pd.util.hash_pandas_object(self.data, index=True).values
        ).hexdigest()

        self.limits.max = self.data['stop_hour'].max()
        self.limits.min = self.data['stop_hour'].min()

    def aggregate_preagregated_data(
        self, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> pd.DataFrame:
        self.data_loader()

        # Extract data that is between start_time and end_time
        current_data = self.data.loc[
            (start_time <= self.data['stop_hour']) & (self.data['stop_hour'] < end_time)
        ].copy()

        if not current_data.empty:
            # As self.data is already pre-aggregated we need to compute the weighted
            # mean of the delays. This requires several steps with pandas.
            # Get the number of data-points in each pre-aggregated datapoint
            group_sizes = current_data.groupby('station').agg(
                {
                    'ar_happened_sum': 'sum',
                    'dp_happened_sum': 'sum',
                }
            )
            # For each pre-aggregated datapoint of each station, calculate its fraction of stops
            # compared to the total stops at the station
            group_sizes = (
                current_data.set_index('station')[
                    ['ar_happened_sum', 'dp_happened_sum']
                ]
                / group_sizes
            )
            # rename columns in order to multiply them with the mean delays
            group_sizes.rename(
                columns={
                    'ar_happened_sum': 'ar_delay_mean',
                    'dp_happened_sum': 'dp_delay_mean',
                },
                inplace=True,
            )
            group_sizes.reset_index(drop=True, inplace=True)
            # calculate the minutes of delay from each pre-aggregated datapoint of each station
            weighted_mean = (
                current_data.reset_index()[['ar_delay_mean', 'dp_delay_mean']]
                * group_sizes
            )
            weighted_mean.index = current_data.index
            # re-insert it into the original pre-aggregated datapoint in order to aggregate it
            current_data.loc[:, ['ar_delay_mean', 'dp_delay_mean']] = weighted_mean[
                ['ar_delay_mean', 'dp_delay_mean']
            ]

            current_data = current_data.groupby('station').agg(
                {
                    'ar_delay_mean': 'sum',
                    'ar_happened_sum': 'sum',
                    'dp_delay_mean': 'sum',
                    'dp_happened_sum': 'sum',
                    'lat': 'first',
                    'lon': 'first',
                }
            )
            current_data = current_data.fillna(0)

        return current_data

    def generate_default(self, plot_title: str) -> str:
        plot_path = self.PLOT_PATH.format(version=self.version, title=plot_title)
        if not os.path.isfile(plot_path):
            if plot_title == 'default':
                self.ax.set_title('', fontsize=16)
            else:
                self.ax.set_title(plot_title, fontsize=16)
            memory_buffer = io.BytesIO()
            self.fig.savefig(
                memory_buffer, bbox_inches='tight', dpi=250, transparent=True
            )
            image_to_webp(memory_buffer, plot_path)
        return plot_path

    def generate_plot(self, start_time, end_time, use_cached_images=False) -> str:
        """
        Generates a plot that visualizes all the delays on a Germany map between `start_time` and `end_time`
        The file is generated relative to this execution path inside of  `cache/plots/{plot_name}.webp`

        Parameters
        ----------
        start_time : datetime.datetime
            Start of time range
        end_time : datetime.datetime
            End of time range

        Returns
        -------
        str
            Path to the generated plot
        """

        if start_time + datetime.timedelta(hours=self.limits.freq_hours) > end_time:
            # We generate plots over a minimum timespan of limits.freq_hours
            end_time = end_time + datetime.timedelta(hours=self.limits.freq_hours)

        plot_title = (
            start_time.strftime('%d.%m.%Y') + '-' + end_time.strftime('%d.%m.%Y')
        )

        plot_path = self.PLOT_PATH.format(version=self.version, title=plot_title)

        if use_cached_images and os.path.isfile(plot_path):
            # Return cached image
            return plot_path

        current_data = self.aggregate_preagregated_data(start_time, end_time)

        if not current_data.empty:
            # Normalize the data so that the circles are equally sized, no matter
            # how many days of date were requested.
            n_days = (end_time - start_time).days
            size = (
                current_data.loc[:, ['ar_happened_sum']].to_numpy()[:, 0]
                + current_data.loc[:, ['dp_happened_sum']].to_numpy()[:, 0]
            )
            size = size / n_days
            # 2000 is the average max number of trains on the busiest station.
            # This is used to scale the size of the markers
            size = (size / 2000) * 70

            color = (
                current_data.loc[:, ['ar_delay_mean']].to_numpy().astype(float)[:, 0]
            )

            # change the positions
            self.sc.set_offsets(
                self.MAP_CRS.transform_points(
                    ccrs.Geodetic(), current_data['lon'], current_data['lat']
                )[:, :2]
            )
            # change the sizes
            self.sc.set_sizes(size)
            # change the color
            self.sc.set_array(color)

            self.ax.set_title(
                plot_title.replace('_', ':').replace('-', ' - '), fontsize=12
            )
            memory_buffer = io.BytesIO()
            self.fig.savefig(
                memory_buffer,
                bbox_inches='tight',
                dpi=250,
                transparent=True,
                format='png',
            )
            image_to_webp(memory_buffer, plot_path)
        else:
            plot_path = self.generate_default(plot_title='no data available')

        return plot_path


if __name__ == '__main__':
    from helpers.bahn_vorhersage import COLORFUL_ART

    print(COLORFUL_ART)

    # per_station_time = PerStationOverTime(generate=False, prefer_cache=True)
    # per_station_time.generate_plot(
    #     datetime.datetime(2022, 6, 1), datetime.datetime(2022, 8, 31)
    # )
    # per_station_time.generate_plot(
    #     datetime.datetime(2022, 1, 1), datetime.datetime(2022, 5, 30)
    # )

    import time

    start = time.time()
    per_station_time = PerStationOverTime(generate=True, prefer_cache=True)
    data = per_station_time.aggregate_preagregated_data(
        datetime.datetime(2022, 1, 1), datetime.datetime(2022, 12, 31)
    )
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data.lon, data.lat),
        crs='EPSG:4326',
    )
    gdf.to_file('stations_2022.gpkg', driver='GPKG', layer='stations')

    # per_station_time.generate_plot(
    #     datetime.datetime(2021, 3, 1, hour=0), datetime.datetime(2021, 3, 10, hour=0)
    # )
    # print('took:', time.time() - start)
