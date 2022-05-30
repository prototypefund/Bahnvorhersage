import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import osmnx as ox
import osmnx.downloader as downloader
import osmnx.stats as stats
import osmnx.utils as utils
import osmnx.simplification as simplification
import osmnx.truncate as truncate
import osmnx.graph as graph
import osmnx.settings as settings
import osmnx.projection as projection
import osmnx.distance as distance
import osmnx.utils_geo as utils_geo
from osmnx._errors import EmptyOverpassResponse
import itertools
import functools
import numpy as np
import pandas as pd
import geopandas as gpd
import pygeos
import shapely
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm

import warnings

from helpers import profile

oneway_values = {"yes", "true", "1", "-1", "reverse", "T", "F"}
reversed_values = {"-1", "reverse", "T"}


def _osm_network_download(polygon, network_type, custom_filter):
    """
    Retrieve networked ways and nodes within boundary from the Overpass API.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        boundary to fetch the network ways/nodes within
    network_type : string
        what type of street network to get if custom_filter is None
    custom_filter : string
        a custom ways filter to be used instead of the network_type presets

    Returns
    -------
    response_jsons : list
        list of JSON responses from the Overpass server
    """
    # create a filter to exclude certain kinds of ways based on the requested
    # network_type, if provided, otherwise use custom_filter
    if custom_filter is not None:
        osm_filter = custom_filter
    else:
        osm_filter = downloader._get_osm_filter(network_type)

    response_jsons = []

    # create overpass settings string
    overpass_settings = downloader._make_overpass_settings()

    # subdivide query polygon to get list of sub-divided polygon coord strings
    polygon_coord_strs = downloader._make_overpass_polygon_coord_strs(polygon)

    # pass each polygon exterior coordinates in the list to the API, one at a
    # time. The '>' makes it recurse so we get ways and the ways' nodes.
    for polygon_coord_str in polygon_coord_strs:
        query_str = f"{overpass_settings};(way{osm_filter}(poly:'{polygon_coord_str}');>;);out;"
        yield downloader.overpass_request(data={"data": query_str})
        # response_json = downloader.overpass_request(data={"data": query_str})
        # response_jsons.append(response_json)
    # utils.log(
    #     f"Got all network data within polygon from API in {len(polygon_coord_strs)} request(s)"
    # )

    # if settings.cache_only_mode:  # pragma: no cover
    #     raise CacheOnlyModeInterrupt("settings.cache_only_mode=True")

    # return response_jsons


def _convert_node(element):
    """
    Convert an OSM node element into the format for a networkx node.

    Parameters
    ----------
    element : dict
        an OSM node element

    Returns
    -------
    node : dict
    """
    if 'tags' in element:
        tags = [element["tags"].get(tag, np.nan) for tag in settings.useful_tags_node]
    else:
        tags = [pd.NA] * len(settings.useful_tags_node)
    node = [element["id"], element["lon"], element["lat"]] + tags

    return node


def _is_path_one_way(path, bidirectional):
    """
    Determine if a path of nodes allows travel in only one direction.

    Parameters
    ----------
    path : dict
        a path's tag:value attribute data
    bidirectional : bool
        whether this is a bi-directional network type
    oneway_values : set
        the values OSM uses in its 'oneway' tag to denote True

    Returns
    -------
    bool
    """
    # rule 1
    if settings.all_oneway:
        # if globally configured to set every edge one-way, then it's one-way
        return True

    # rule 2
    elif bidirectional:
        # if this is a bidirectional network type, then nothing in it is
        # considered one-way. eg, if this is a walking network, this may very
        # well be a one-way street (as cars/bikes go), but in a walking-only
        # network it is a bidirectional edge (you can walk both directions on
        # a one-way street). so we will add this path (in both directions) to
        # the graph and set its oneway attribute to False.
        return False

    # rule 3
    elif "oneway" in path and path["oneway"] in oneway_values:
        # if this path is tagged as one-way and if it is not a bidirectional
        # network type then we'll add the path in one direction only
        return True

    # rule 4
    elif "junction" in path and path["junction"] == "roundabout":
        # roundabouts are also one-way but are not explicitly tagged as such
        return True

    else:
        # otherwise, this path is not tagged as a one-way
        return False


def _is_path_reversed(path):
    """
    Determine if the order of nodes in a path should be reversed.

    Parameters
    ----------
    path : dict
        a path's tag:value attribute data
    reversed_values : set
        the values OSM uses in its 'oneway' tag to denote travel can only
        occur in the opposite direction of the node order

    Returns
    -------
    bool
    """
    if "oneway" in path and path["oneway"] in reversed_values:
        return True
    else:
        return False

unique_tags = {'total_paths': 0}

def _convert_path(element, bidirectional):
    """
    Convert an OSM way element into the format for a networkx path.

    Parameters
    ----------
    element : dict
        an OSM way element

    Returns
    -------
    path : dict
    """
    if 'disused' in element['tags']:
        if element['tags']['disused'] == 'yes':
            return []
        else:
            pass

    is_one_way = _is_path_one_way(element["tags"], bidirectional)
    is_reversed = _is_path_reversed(element["tags"])

    unique_tags['total_paths'] += 1
    for tag in element["tags"]:
        if tag not in unique_tags:
            unique_tags[tag] = [element["tags"][tag], 1]
        else:
            unique_tags[tag][1] += 1

    if not settings.all_oneway:
        element["tags"]["oneway"] = is_one_way

    tags = [element["tags"].get(tag, np.nan) for tag in settings.useful_tags_way]
    path_attrbs = [element["id"]] + tags

    # remove any consecutive duplicate elements in the list of nodes
    nodes = [group[0] for group in itertools.groupby(element["nodes"])]

    path = []

    for u, v in zip(nodes[:-1], nodes[1:]):
        if is_one_way and is_reversed:
            path.append([v, u, 0] + path_attrbs)

        if not is_one_way:
            path.append([v, u, 1] + path_attrbs)

        path.append([u, v, 0] + path_attrbs)
    return path


# @profile(sort='cumulative', lines=50)
def _parse_nodes_paths(elements, bidirectional=False, multipoly=None, polygon=None):
    """
    Construct dicts of nodes and paths from an Overpass response.

    Parameters
    ----------
    response_json : dict
        JSON response from the Overpass API

    Returns
    -------
    nodes, paths : tuple of dicts
        dicts' keys = osmid and values = dict of attributes
    """
    paths = []
    nodes = []
    for element in elements:
        if element["type"] == "node":
            nodes.append(_convert_node(element))
        elif element['type'] == 'way':
            paths.extend(_convert_path(element, bidirectional))

    if nodes:
        nodes = pd.DataFrame(
            nodes,
            columns=['osmid', 'x', 'y'] + settings.useful_tags_node
        ).set_index('osmid')
        nodes = nodes.loc[~nodes.index.duplicated(keep='first')]
        nodes = gpd.GeoDataFrame(
            nodes,
            geometry=gpd.points_from_xy(nodes['x'], nodes['y']),
            crs=settings.default_crs
        )

        paths = pd.DataFrame(
            paths,
            columns=['u', 'v', 'key', 'osmid'] + settings.useful_tags_way
        ).set_index(['u', 'v', 'key'])
        paths = paths.loc[~paths.index.duplicated(keep='first')]

        geom1 = nodes.loc[paths.index.get_level_values('u'), 'geometry']
        geom2 = nodes.loc[paths.index.get_level_values('v'), 'geometry']

        coords = np.array([[geom1.x, geom2.x], [geom1.y, geom2.y]]).T

        paths['geometry'] = pygeos.linestrings(coords)
        paths['length'] = distance.great_circle_vec(
            coords[:, 0, 0],
            coords[:, 0, 1],
            coords[:, 1, 0],
            coords[:, 1, 1]
        )

        paths = gpd.GeoDataFrame(
            paths,
            geometry='geometry',
            crs=settings.default_crs
        )

        if multipoly is not None:

            geoms_in_poly = set()
            # loop through each chunk of the polygon to find intersecting geometries
            for poly in multipoly.geoms:
                # first find approximate matches with spatial index, then precise
                # matches from those approximate ones
                poly = poly.buffer(0)
                if poly.is_valid and poly.area > 0:
                    possible_matches_iloc = nodes.sindex.intersection(poly.bounds)
                    if len(possible_matches_iloc):
                        possible_matches = nodes.iloc[list(possible_matches_iloc)]
                        precise_matches = possible_matches[possible_matches.intersects(polygon)]
                        if not precise_matches.empty:
                            geoms_in_poly.update(precise_matches.index)

            paths = paths.loc[
                paths.index.get_level_values('u').isin(geoms_in_poly) & paths.index.get_level_values('v').isin(
                    geoms_in_poly)]
            nodes = nodes.loc[
                nodes.index.isin(set(paths.index.get_level_values('u')) | set(paths.index.get_level_values('v')))]

        nodes.dropna(axis=1, how='all', inplace=True)
        paths.dropna(axis=1, how='all', inplace=True)

        return nodes, paths
    else:
        return None, None


def batchify_response_elements(response_jsons: dict, batch_size=100_000):
    # Convert response JSONs to batches of min size batch_size
    ever_yielded = False
    elements_batch = []
    for response in response_jsons:
        if response['elements']:
            elements_batch.extend(response.pop('elements'))
            if len(elements_batch) >= batch_size:
                yield elements_batch
                ever_yielded = True
                # elements.extend([elements_batch])
                elements_batch = []
    if elements_batch:
        yield elements_batch
        # elements.extend([elements_batch])
    elif not ever_yielded:
        raise EmptyOverpassResponse("There are no data elements in the response JSON")


def _create_gdfs(response_jsons, retain_all=False, bidirectional=False, polygon=None, batch_size=10_000, max_workers=0):
    utils.log("Creating graph from downloaded OSM data...")

    elements = batchify_response_elements(response_jsons, batch_size)

    geometry_proj, crs_proj = projection.project_geometry(polygon, crs='EPSG:4326', to_crs='EPSG:3857')
    multipoly = utils_geo._quadrat_cut_geometry(geometry_proj, quadrat_width=50 * 1000, min_num=3)
    multipoly, _ = projection.project_geometry(multipoly, crs=crs_proj, to_latlong=True)
    geometry_proj, _ = projection.project_geometry(geometry_proj, crs=crs_proj, to_latlong=True)

    # multipoly = utils_geo._quadrat_cut_geometry(polygon, quadrat_width=0.5, min_num=3)

    # multipoly = utils_geo._quadrat_cut_geometry(polygon, quadrat_width=0.5, min_num=3)

    # fig, ax = plt.subplots(figsize=(16, 16))
    # ax.set_aspect('equal', 'datalim')

    # gpd.GeoSeries([polygon], crs='EPSG:4326').plot(ax=ax, color='grey')
    # gpd.GeoSeries([geometry_proj]).plot(ax=ax, color='orange')
    # gpd.GeoSeries([multipoly]).plot(ax=ax, edgecolor='black', facecolor='none')
    # plt.show()

    if max_workers == 0:
        parse_nodes_paths = lambda e: _parse_nodes_paths(e, bidirectional, multipoly, polygon)
        elements = tqdm(map(parse_nodes_paths, elements))
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn')) as executor:
            elements = list(tqdm(
                executor.map(
                    _parse_nodes_paths, elements, itertools.repeat(bidirectional), itertools.repeat(multipoly)
                )
            ))

    nodes, edges = zip(*elements)

    print(unique_tags)

    def get_cmap(n, name='hsv'):
        """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name."""
        return plt.cm.get_cmap(name, n)

    cmap = get_cmap(len(edges))

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_aspect('equal', 'datalim')

    gpd.GeoSeries([polygon]).plot(ax=ax, color='grey')
    gpd.GeoSeries([multipoly]).plot(ax=ax, edgecolor='black', facecolor='none')

    for i, path in enumerate(edges):
        path.plot(ax=ax, color=cmap(i), linewidth=1)
        print(i)

    plt.show()

    return gpd.concat(nodes), gpd.concat(edges)


def gdfs_from_polygon(
        polygon,
        network_type="all_private",
        simplify=True,
        retain_all=False,
        truncate_by_edge=False,
        clean_periphery=True,
        custom_filter=None,
):
    # verify that the geometry is valid and is a shapely Polygon/MultiPolygon
    # before proceeding
    if not polygon.is_valid:  # pragma: no cover
        raise ValueError("The geometry to query within is invalid")
    if not isinstance(polygon, (Polygon, MultiPolygon)):  # pragma: no cover
        raise TypeError(
            "Geometry must be a shapely Polygon or MultiPolygon. If you requested "
            "graph from place name, make sure your query resolves to a Polygon or "
            "MultiPolygon, and not some other geometry, like a Point. See OSMnx "
            "documentation for details."
        )

    if clean_periphery:
        # create a new buffered polygon 0.5km around the desired one
        buffer_dist = 500
        poly_proj, crs_utm = projection.project_geometry(polygon)
        poly_proj_buff = poly_proj.buffer(buffer_dist)
        poly_buff, _ = projection.project_geometry(poly_proj_buff, crs=crs_utm, to_latlong=True)

        # download the network data from OSM within buffered polygon
        response_jsons = _osm_network_download(poly_buff, network_type, custom_filter)

        # create buffered graph from the downloaded data
        bidirectional = network_type in settings.bidirectional_network_types
        nodes, edges = _create_gdfs(response_jsons, retain_all=True, bidirectional=bidirectional, polygon=polygon)

        return nodes, edges

        # # simplify the graph topology
        # if simplify:
        #     G_buff = simplification.simplify_graph(G_buff)

        # # count how many physical streets in buffered graph connect to each
        # # intersection in un-buffered graph, to retain true counts for each
        # # intersection, even if some of its neighbors are outside the polygon
        # spn = stats.count_streets_per_node(G_buff, nodes=G.nodes)
        # ox.nx.set_node_attributes(G, values=spn, name="street_count")

    # if clean_periphery=False, just use the polygon as provided
    else:
        # # download the network data from OSM
        # response_jsons = downloader._osm_network_download(polygon, network_type, custom_filter)

        # # create graph from the downloaded data
        # bidirectional = network_type in settings.bidirectional_network_types
        # G = ox._create_graph(response_jsons, retain_all=True, bidirectional=bidirectional)

        # # truncate the graph to the extent of the polygon
        # G = truncate.truncate_graph_polygon(G, polygon, retain_all, truncate_by_edge)

        # # simplify the graph topology after truncation. don't truncate after
        # # simplifying or you may have simplified out to an endpoint beyond the
        # # truncation distance, which would strip out the entire edge
        # if simplify:
        #     G = simplification.simplify_graph(G)

        # # count how many physical streets connect to each intersection/deadend
        # # note this will be somewhat inaccurate due to periphery effects, so
        # # it's best to parameterize function with clean_periphery=True
        # spn = stats.count_streets_per_node(G)
        # ox.nx.set_node_attributes(G, values=spn, name="street_count")
        msg = (
            "the graph-level street_count attribute will likely be inaccurate "
            "when you set clean_periphery=False"
        )
        warnings.warn(msg)

    utils.log(f"graph_from_polygon returned graph with {len(G)} nodes and {len(G.edges)} edges")
    return G
