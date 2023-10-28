import itertools
import math
from typing import Dict, Iterable, List, Tuple

from helpers import pairwise

import geopandas as gpd
import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
import shapely
from osmnx import _downloader, settings
from shapely.geometry import LineString, MultiPoint, Point, Polygon, MultiLineString
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import igraph as ig
import xxhash
import sqlalchemy
import geoalchemy2

from database.cached_table_fetch import cached_table_fetch_postgis
from database.engine import get_engine
from helpers.StationPhillip import StationPhillip

RAIL_FILTER = (
    f'["railway"~"rail|tram|narrow_gauge|light_rail|subway"]'
    f'["railway"!~"proposed|construction|disused|abandoned|razed|miniature"]'
    f'["service"!~"yard|spur"]'
    f'["gauge"~"750|760|800|900|1000|1435|1445|1450|1458|1520|1524|1620|1668|750;1435|1435;750|760;1435|1435;760|1000;1435|1435;1000|1435;1520|1520;1435|1435;1668"]'
)

USEFUL_TAGS_RAIL_WAYS = [
    'usage',
    'highspeed',
    'electrified',
    'gauge',
    'maxspeed',
    'service',
    'railway:preferred_direction',
    'railway:etcs',
    'railway:kvb',
    'railway:lzb',
    'railway:pzb',
    'bridge',
    'tunnel',
]

USEFUL_TAGS_RAIL_NODES = [
    'railway',
    'crossing:barrier',
    'crossing:light',
    'name',
]

SPLITTER_LINE_LENGTH = 100
BBOX_HALF_SIZE = 100
EPSG_PROJECTED = 'EPSG:3857'


def _convert_node(element, useful_tags):
    node = {'y': element['lat'], 'x': element['lon']}
    if 'tags' in element:
        for useful_tag in useful_tags:
            if useful_tag in element['tags']:
                node[useful_tag] = element['tags'][useful_tag]
    return element['id'], node


def _convert_path(element: Dict, useful_tags: Iterable[str]) -> List[Dict]:
    tags = {}

    # remove any consecutive duplicate elements in the list of nodes
    nodes = [group[0] for group in itertools.groupby(element['nodes'])]

    if 'tags' in element:
        for useful_tag in useful_tags:
            if useful_tag in element['tags']:
                tags[useful_tag] = element['tags'][useful_tag]

    edges = []
    for u, v in pairwise(nodes):
        edges.append({'u': u, 'v': v, **tags})

    return edges


def _add_geometry_to_edges(edges: List[Dict], nodes: Dict[int, Dict]):
    for edge in edges:
        edge['geometry'] = LineString(
            [
                (nodes[edge['u']]['x'], nodes[edge['u']]['y']),
                (nodes[edge['v']]['x'], nodes[edge['v']]['y']),
            ]
        )
        edge['length'] = length_of_line(edge['geometry'])

    return edges


def _parse_osm_nodes_paths(response_json: Dict) -> Tuple[Dict[int, Dict], List[Dict]]:
    nodes = {}
    edges = []
    for element in response_json['elements']:
        if element['type'] == 'node':
            osmid, node = _convert_node(element, USEFUL_TAGS_RAIL_NODES)
            nodes[osmid] = node
        elif element['type'] == 'way':
            paths = _convert_path(element, USEFUL_TAGS_RAIL_WAYS)
            edges.extend(paths)
    edges = _add_geometry_to_edges(edges, nodes)

    return nodes, edges


def get_gdf_from_osm(polygon: Polygon) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    # download the network data from OSM within buffered polygon
    response_jsons = _downloader._osm_network_download(
        polygon=polygon, network_type=None, custom_filter=RAIL_FILTER
    )

    nodes = {}
    edges = []

    # Non concurrent version for debugging
    # for response_json in response_jsons:
    #     current_nodes, current_edges = _parse_osm_nodes_paths(response_json)
    #     nodes.update(current_nodes)
    #     edges.extend(current_edges)

    with ProcessPoolExecutor() as executor:
        tasks = [
            executor.submit(_parse_osm_nodes_paths, response_json)
            for response_json in response_jsons
        ]
        for task in tqdm(as_completed(tasks), total=len(tasks)):
            current_nodes, current_edges = task.result()
            nodes.update(current_nodes)
            edges.extend(current_edges)

    nodes = pd.DataFrame.from_dict(nodes, orient='index')
    nodes = gpd.GeoDataFrame(
        nodes, geometry=gpd.points_from_xy(nodes.x, nodes.y), crs='EPSG:4326'
    )
    nodes = nodes[~nodes.index.duplicated(keep='first')]

    edges = pd.DataFrame(edges)
    edges = gpd.GeoDataFrame(edges, geometry=edges['geometry'], crs='EPSG:4326')
    edges.drop_duplicates(subset=['u', 'v'], inplace=True)

    return nodes, edges


def angle_three_points(a, b, c):
    """Calculate the smallest angle β from the points abc"""
    β = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return (β + 360 if β < 0 else β) % 180


flatten = itertools.chain.from_iterable


def _add_length_to_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    original_crs = gdf.crs
    gdf = gdf.to_crs('EPSG:4326')
    gdf['length'] = gdf['geometry'].apply(length_of_line)
    gdf = gdf.to_crs(original_crs)
    return gdf


def length_of_line(line: LineString) -> float:
    """
    Calculate length of line in meters.

    Parameters
    ----------
    line: LineString
        Line to calculate length of. Must be EPSG:4326 (lat/lon in degree)

    Returns
    -------
    float:
        Length of line.
    """
    if line is not None:
        return sum(
            (
                geopy.distance.distance(p1, p2).meters
                for p1, p2 in pairwise(tuple(zip(line.xy[0], line.xy[1])))
            )
        )
    else:
        return None


def edges_mergeable(edge1: ig.Edge, edge2: ig.Edge) -> bool:
    attributes1 = edge1.attributes()
    attributes2 = edge2.attributes()

    # Remove attributes that are not useful for comparison
    for key in ['length', 'geometry']:
        attributes1.pop(key, None)
        attributes2.pop(key, None)

    return attributes1 == attributes2


def simplify(nodes, edges) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    replace_ids = {key: i for i, key in enumerate(nodes['index'])}
    edges['u'] = edges['u'].map(replace_ids.get)
    edges['v'] = edges['v'].map(replace_ids.get)
    nodes['index'] = nodes['index'].map(replace_ids.get)
    # nodes.set_index('index', inplace=True, drop=True)

    streckennetz = ig.Graph.DataFrame(edges, directed=False, vertices=nodes)

    while True:
        node_indices_to_touch = set()
        nodes_to_remove = []
        edges_to_add = []
        edge_attributes_to_add = []
        # Remove all nodes with degree 2, that are not stations and whose edges have the same attributes
        for node in tqdm(streckennetz.vs, desc='Simplifying Streckennetz'):
            if node.degree() == 2 and node['railway'] not in [
                'station',
                'stop',
                'halt',
            ]:
                # Do not simplify nodes that will already be simplified
                if (node.index in node_indices_to_touch):
                    continue
                if edges_mergeable(node.all_edges()[0], node.all_edges()[1]):
                    node_indices_to_touch.add(node.index)
                    node_indices_to_touch.add(node.neighbors()[0].index)
                    node_indices_to_touch.add(node.neighbors()[1].index)

                    # new_geometry = MultiLineString(
                    #     [
                    #         node.all_edges()[0]['geometry'],
                    #         node.all_edges()[1]['geometry'],
                    #     ]
                    # )
                    # new_geometry = shapely.line_merge(new_geometry)
                    # TODO: Contact linestring correctly
                    new_geometry = LineString()

                    new_edge = node.all_edges()[0].attributes()
                    new_edge['geometry'] = new_geometry
                    new_edge['length'] = (
                        node.all_edges()[0]['length'] + node.all_edges()[1]['length']
                    )

                    edges_to_add.append(
                        (node.neighbors()[0].index, node.neighbors()[1].index)
                    )
                    edge_attributes_to_add.append(new_edge)

                    nodes_to_remove.append(node.index)
        if edges_to_add:
            # Reorient attributes
            edge_attributes_to_add = pd.DataFrame.from_records(
                edge_attributes_to_add
            ).to_dict(orient='list')
            streckennetz.add_edges(edges_to_add, edge_attributes_to_add)
            streckennetz.delete_vertices(nodes_to_remove)
        else:
            break

    nodes, edges = streckennetz.to_dict_list()
    nodes = gpd.GeoDataFrame(pd.DataFrame.from_records(nodes), crs='EPSG:4326')
    edges = gpd.GeoDataFrame(pd.DataFrame.from_records(edges), crs='EPSG:4326')
    return nodes, edges


def to_postgis(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame, table_name: str):
    """
    Upload Streckennetz, including edge attributes, to database

    Parameters
    ----------
    nodes : gpd.GeoDataFrame
        GeoDataFrame of nodes
    edges : gpd.GeoDataFrame
        GeoDataFrame of edges
    table_name : str
        Name of the table to upload to
    """

    # dtypes used for sqlalchemy
    edges_dtypes = {
        'u': sqlalchemy.BIGINT,
        'v': sqlalchemy.BIGINT,
        'usage': sqlalchemy.String,
        'highspeed': sqlalchemy.String,
        'electrified': sqlalchemy.String,
        'gauge': sqlalchemy.String,
        'maxspeed': sqlalchemy.String,
        'service': sqlalchemy.String,
        'railway:preferred_direction': sqlalchemy.String,
        'railway:etcs': sqlalchemy.String,
        'railway:kvb': sqlalchemy.String,
        'railway:lzb': sqlalchemy.String,
        'railway:pzb': sqlalchemy.String,
        'bridge': sqlalchemy.String,
        'tunnel': sqlalchemy.String,
        'length': sqlalchemy.Float,
        'geometry': geoalchemy2.Geometry('LINESTRING'),
    }

    nodes_dtypes = {
        'index': sqlalchemy.BIGINT,
        'name': sqlalchemy.String,
        'railway': sqlalchemy.String,
        'crossing:barrier': sqlalchemy.String,
        'crossing:light': sqlalchemy.String,
        'x': sqlalchemy.Float,
        'y': sqlalchemy.Float,
        'geometry': geoalchemy2.Geometry('POINT'),
    }

    if nodes.crs is None:
        nodes.set_crs(EPSG_PROJECTED, inplace=True)
    nodes.to_crs('EPSG:4326', inplace=True)
    nodes.reset_index(inplace=True)

    if edges.crs is None:
        edges.set_crs(EPSG_PROJECTED, inplace=True)
    edges.to_crs('EPSG:4326', inplace=True)

    nodes.to_postgis(
        f'{table_name}_nodes',
        if_exists='replace',
        con=get_engine(),
        chunksize=10_000,
        dtype=nodes_dtypes,
    )

    edges.to_postgis(
        f'{table_name}_edges',
        if_exists='replace',
        con=get_engine(),
        chunksize=10_000,
        dtype=edges_dtypes,
    )


def plot_algorithm(
    name,
    close_edges,
    line,
    orth_line,
    points=None,
    cuts=None,
    rep_points=None,
    intersections=None,
    u=None,
    v=None,
    splitted=None,
    closest_edge=None,
    more_plots: List[Tuple] = None,
):
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'datalim')
    ax.set_title(name)

    for index, edge in close_edges.iterrows():
        ax.plot(*edge['geometry'].xy, color='black')

    ax.plot(*line.xy, color='red')
    ax.plot(*orth_line.xy, color='red')

    if points:
        ax.scatter(*points[0].xy, color='blue')
        ax.scatter(*points[1].xy, color='blue')

    if cuts:
        for cut in cuts:
            ax.plot(*cut.xy, color='purple')

    if rep_points:
        for rep_point in rep_points:
            ax.scatter(*rep_point.xy, color='gold', zorder=3)

    if intersections:
        for intersextion in intersections:
            ax.scatter(*intersextion.xy, color='green', zorder=3)

    if u:
        ax.scatter(*u.xy, color='green', zorder=3)
    if v:
        ax.scatter(*v.xy, color='orange', zorder=3)

    if splitted:
        # Plot splitted edges. Note: there can be 1 or 2 splitted edges
        for geom, color in zip(splitted.geoms, ['green', 'orange']):
            ax.plot(*geom.xy, color=color)

    if closest_edge:
        ax.plot(*closest_edge.xy, color='brown')

    if more_plots:
        for plot in more_plots:
            if hasattr(plot[0], 'geoms'):
                for geom in plot[0].geoms:
                    ax.plot(*geom.xy, color=plot[1])
            else:
                ax.plot(*plot[0].xy, color=plot[1])

    plt.show()


def split_geom(
    geom,
    line,
) -> Point | None:
    if geom.intersects(line):
        intersection = geom.intersection(line)
        angle = angle_three_points(
            geom.representative_point().coords[0],
            intersection.coords[0],
            line.representative_point().coords[0],
        )
        if 70 < angle < 110:
            return intersection


def create_new_edges(
    name: str, edge: pd.Series, nodes: gpd.GeoDataFrame, intersection: Point
) -> gpd.GeoDataFrame:
    name_hash = xxhash.xxh32_intdigest(name)
    v = nodes.loc[edge['v']]
    u = nodes.loc[edge['u']]

    v_edge = edge.copy()
    v_edge['v'] = name_hash
    v_edge['geometry'] = LineString([v['geometry'], intersection])

    u_edge = edge.copy()
    u_edge['u'] = name_hash
    u_edge['geometry'] = LineString([intersection, u['geometry']])

    add = gpd.GeoDataFrame([v_edge, u_edge], crs=EPSG_PROJECTED)
    return _add_length_to_gdf(add)


def split_edge(
    index: int,
    name: str,
    edge: pd.Series,
    nodes: gpd.GeoDataFrame,
    close_edges: gpd.GeoDataFrame,
    line: LineString,
    orth_line: LineString,
    plot: bool = False,
) -> Dict | None:
    """Split the geometry of an edge.

    Parameters
    ----------
    index : int
        Index of the edge to split
    name : str
        Name of the station where the edge is split
    edge : gpd.GeoSeries
        GeoSeries of the edge to split
    close_edges : gpd.GeoDataFrame
        GeoDataFrame of this and other edges close to the station
    line : LineString
        Line, with the station as center and orthogonal to the nearest edge
    orth_line : LineString
        Line, with the station as center and orthogonal to line
    plot : bool, optional
        Whether to plot the process or not, by default False
    """
    # split by first line
    intersection = split_geom(edge['geometry'], line)
    if intersection is None:
        # split by orthogonal line if split by line did not result in a split
        intersection = split_geom(edge['geometry'], orth_line)

    if intersection is not None:
        if plot:
            plot_algorithm(
                name,
                close_edges,
                line,
                orth_line,
            )

        add = create_new_edges(name, edge, nodes, intersection)

        return {
            'drop': index,
            'add': add,
            'intersection': intersection,
        }

    else:
        return None


def insert_station(
    name: str,
    station: gpd.GeoSeries,
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    plot=False,
):
    # Get edges close to station using r-tree indexing
    # (this is way faster than using GeoDataFrame.cx[])
    bbox = shapely.geometry.box(
        station['geometry'].x - BBOX_HALF_SIZE,
        station['geometry'].y - BBOX_HALF_SIZE,
        station['geometry'].x + BBOX_HALF_SIZE,
        station['geometry'].y + BBOX_HALF_SIZE,
    )
    close_edges = edges.iloc[list(edges.sindex.intersection(bbox.bounds))]

    if not close_edges.empty:
        multipoint = close_edges['geometry'].unary_union

        points = shapely.ops.nearest_points(station['geometry'], multipoint)
        # Calculate vector from station nearest point
        bhf_vec = np.array([points[0].x - points[1].x, points[0].y - points[1].y])
        # Shorten vector to have a length of 1
        bhf_vec = bhf_vec / np.sqrt(bhf_vec.dot(bhf_vec))
        # Create a vector orthogonal to bhf_vec
        bhf_orth_vec = np.array([bhf_vec[1], -bhf_vec[0]])

        station_coords = np.array(station['geometry'].xy).flatten()

        # line and orth_line make a cross on the station
        line = LineString(
            [
                station_coords - (SPLITTER_LINE_LENGTH * bhf_vec),
                station_coords + (SPLITTER_LINE_LENGTH * bhf_vec),
            ]
        )
        orth_line = LineString(
            [
                station_coords - (SPLITTER_LINE_LENGTH * bhf_orth_vec),
                station_coords + (SPLITTER_LINE_LENGTH * bhf_orth_vec),
            ]
        )

        # split edges either with line or orth_line
        changes = {
            'drop': [],
            'add': [],
            'intersection': [],
        }
        for index, edge in close_edges.iterrows():
            current_changes = split_edge(
                index=index,
                name=name,
                edge=edge,
                close_edges=close_edges,
                nodes=nodes,
                line=line,
                orth_line=orth_line,
                plot=plot,
            )
            if current_changes is not None:
                changes['drop'].append(current_changes['drop'])
                changes['add'].append(current_changes['add'])
                changes['intersection'].append(current_changes['intersection'])
        if plot:
            plot_algorithm(
                name=name,
                close_edges=close_edges,
                line=line,
                orth_line=orth_line,
                points=points,
            )
        if changes['drop']:
            return changes
        else:
            return None
    else:
        print('there are no edges close to', name)
    # If nothing was split at all, return None instead
    return None


def station_already_exists(
    name: str,
    station: pd.Series,
    nodes: gpd.GeoDataFrame,
) -> bool:
    bbox = shapely.geometry.box(
        station['geometry'].x - 100,
        station['geometry'].y - 100,
        station['geometry'].x + 100,
        station['geometry'].y + 100,
    )

    close_nodes = nodes.iloc[list(nodes.sindex.intersection(bbox.bounds))]
    if not close_nodes.empty:
        if (
            (close_nodes['name'] == name)
            & (close_nodes['railway'].isin(['stop', 'station', 'halt']))
        ).any():
            return True
    return False


def download_and_process_streckennetz():
    stations = StationPhillip(prefer_cache=False)

    stations_gdf = stations.to_gdf(date='latest', index_cols=['name'])
    stations_gdf = stations_gdf.loc[~stations_gdf.index.duplicated(keep='first')]

    station_point_cloud = np.array(
        list(zip(stations_gdf['geometry'].x, stations_gdf['geometry'].y))
    )

    settings.log_console = False
    settings.use_cache = True
    settings.cache_folder = 'cache/ox_cache'

    try:
        nodes = gpd.read_parquet('cache/nodes.parquet')
        edges = gpd.read_parquet('cache/edges.parquet')
        print('Using cached streckennetz instead of downloading new one from osm')
    except FileNotFoundError:
        nodes, edges = get_gdf_from_osm(MultiPoint(station_point_cloud).convex_hull)

        nodes.to_parquet('cache/nodes.parquet')
        edges.to_parquet('cache/edges.parquet')

    stations_gdf = stations_gdf.to_crs(EPSG_PROJECTED)
    nodes = nodes.to_crs(EPSG_PROJECTED)
    nodes['x'] = nodes['geometry'].x
    nodes['y'] = nodes['geometry'].y
    edges = edges.to_crs(EPSG_PROJECTED)

    replace_edges = set()
    add_edges = []
    add_nodes = {}
    recompute = []
    stations_to_insert = stations_gdf.index.to_list()
    while stations_to_insert:
        print('inserting', len(stations_to_insert), 'stations')
        # rebuild r-tree index for fast spatial indexing
        edges.sindex
        nodes.sindex
        for name in tqdm(stations_to_insert):
            if station_already_exists(name, stations_gdf.loc[name], nodes):
                # print(name, ' already exists')
                continue

            changes = insert_station(
                name=name,
                station=stations_gdf.loc[name],
                edges=edges,
                nodes=nodes,
                plot=False,
            )
            if changes is not None:
                # test whether one of the spitted edges was already split for
                # another station
                for index in changes['drop']:
                    if index in replace_edges:
                        # save the station to be recomputed in the next run
                        recompute.append(name)
                        break
                else:
                    # save changes that should be made in order to insert this station
                    replace_edges.update(changes['drop'])
                    add_edges.append(changes['add'])

                    add_nodes[xxhash.xxh32_intdigest(name)] = {
                        'geometry': stations_gdf.loc[name, 'geometry'],
                        'name': name,
                        'railway': 'stop',
                        'x': stations_gdf.loc[name, 'geometry'].x,
                        'y': stations_gdf.loc[name, 'geometry'].y,
                    }
            else:
                pass
                # print('there are no edges close to', name)
                # insert_station(name, stations_gdf.loc[name], edges, nodes, plot=True)
        # remove splitted edges, append new edges and nodes
        add_edges = pd.concat(flatten(add_edges), ignore_index=True)
        add_nodes = pd.DataFrame().from_dict(add_nodes, orient='index')
        edges = edges.drop(replace_edges)
        edges = pd.concat([edges, add_edges], ignore_index=True)
        nodes = pd.concat([nodes, add_nodes])

        # reset variables to store splitting results
        replace_edges = set()
        replace_edges = set()
        add_edges = []
        add_nodes = {}
        stations_to_insert = recompute
        recompute = []

    print('Uploading full')
    to_postgis(nodes, edges, 'streckennetz_zwei_null')


def main():
    download_and_process_streckennetz()

    nodes = cached_table_fetch_postgis(
        'streckennetz_zwei_null_nodes', prefer_cache=False
    )
    edges = cached_table_fetch_postgis(
        'streckennetz_zwei_null_edges', prefer_cache=False
    )

    nodes, edges = simplify(nodes, edges)

    to_postgis(nodes, edges, 'streckennetz_zwei_null_simplified')

    # download_and_process_streckennetz()


if __name__ == '__main__':
    import helpers.bahn_vorhersage

    main()
