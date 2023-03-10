import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shapely
import geopandas as gpd
from helpers import StationPhillip, BetriebsstellenBill, ObstacleOlly
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from database import cached_table_fetch
import cartopy.crs as ccrs
import cartopy
import matplotlib
from data_analysis.per_station import create_base_plot

matplotlib.use('TkAgg')

def wkb_reverse_hexer(wbk_hex):
    return shapely.wkb.loads(wbk_hex, hex=True)


def plot_construction_work(ax):
    obstacles = ObstacleOlly(prefer_cache=True)

    rows = []
    station_obsacles = []
    for _, obstacle in obstacles.obstacles.iterrows():
        if obstacle['dir'] != 3:
            rows.append((obstacle['u'], obstacle['v'], 0))
            rows.append((obstacle['v'], obstacle['u'], 0))
        else:
            station_obsacles.append(obstacle['from_edge'])

    obstacle_edges = streckennetz.loc[streckennetz.index.isin(rows)]
    return obstacle_edges.plot(color='red', ax=ax)


def annotate_station_names(ax, names, station_gdf):
    for name, row in station_gdf.iterrows():
        if name in names:
            ax.annotate(text=name, xy=row['geometry'].coords[0])


stations = StationPhillip(prefer_cache=True)
station_gdf = stations.to_gdf()

betriebsstellen = BetriebsstellenBill(prefer_cache=True)
betriebsstellen_gdf = betriebsstellen.to_gdf()

streckennetz = cached_table_fetch('full_streckennetz', prefer_cache=True).set_index(
    ['u', 'v', 'key']
)
streckennetz['geometry'] = streckennetz['geometry'].apply(wkb_reverse_hexer)

nodes = cached_table_fetch('full_streckennetz_nodes', prefer_cache=True)
nodes['geometry'] = nodes['geometry'].apply(wkb_reverse_hexer)

streckennetz = gpd.GeoDataFrame(streckennetz, geometry='geometry', crs='epsg:4326')
# streckennetz = streckennetz.to_crs(epsg=ccrs.Mercator())

nodes = gpd.GeoDataFrame(nodes, geometry='geometry')
station_nodes = nodes.loc[~nodes['type'].isna()]

fig, strecke = create_base_plot(
    ccrs.Mercator(),
    streckennetz.total_bounds,
    color_scheme='light'
)

strecke.add_geometries(
    streckennetz['geometry'],
    crs=ccrs.Geodetic(),
    facecolor=(0, 0, 0, 0),
    edgecolor='black',
    linewidth=0.5
)


# station_nodes.plot(color='black', ax=strecke)

# strecke = plot_construction_work(strecke)

# plt.show()
# strecke.set_aspect('equal', 'datalim')
plt.savefig('streckennetz.png', dpi=1200)
# plt.show()
