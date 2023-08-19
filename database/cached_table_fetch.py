from typing import Optional, Callable
import pandas as pd
import geopandas as gpd
from database import DB_CONNECT_STRING, get_engine
from config import CACHE_PATH
from cityhash import CityHash64


def cached_table_fetch_postgis(
    tablename: str,
    prefer_cache: Optional[bool] = False,
    **kwargs,
) -> gpd.GeoDataFrame:
    cache_path = f'{CACHE_PATH}/{tablename}.parquet'

    if prefer_cache:
        try:
            return gpd.read_parquet(cache_path)
        except FileNotFoundError:
            pass
    try:
        df = gpd.read_postgis(
            f'SELECT * FROM {tablename}',
            con=get_engine(),
            geom_col='geometry',
            **kwargs,
        )
        df.to_parquet(cache_path)
    except Exception as ex:
        df = gpd.read_parquet(cache_path)

    return df


def cached_sql_fetch(
    sql: str,
    prefer_cache: Optional[bool] = False,
    **kwargs,
) -> gpd.GeoDataFrame:
    cache_name = hex(CityHash64(sql))
    cache_path = f'{CACHE_PATH}/{cache_name}.parquet'

    if prefer_cache:
        try:
            return pd.read_parquet(cache_path)
        except FileNotFoundError:
            pass
    try:
        df = pd.read_sql(sql, con=DB_CONNECT_STRING, **kwargs)
        df.to_parquet(cache_path)
    except Exception as ex:
        df = pd.read_parquet(cache_path)

    return df


def cached_table_fetch(
    tablename: str,
    prefer_cache: Optional[bool] = False,
    generate: Optional[bool] = False,
    table_generator: Optional[Callable[[], pd.DataFrame]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Fetch table from database and create a local cache of it

    Parameters
    ----------
    tablename : str
        Name of the sql-table to fetch
    prefer_cache : bool, optional
        Whether to try to only load the cache and not ping the database. Usefully for big tables, by default False
    generate : bool, optional
        Whether to use table_generator to generate the DataFrame and not look for cache or database, by default False
    table_generator : Callable[[], pd.DataFrame], optional
        Callable that generates the data of table tablename, by default None

    Returns
    -------
    pd.DataFrame
        DataFrame containing the fetched table

    Raises
    ------
    FileNotFoundError
        The Database is not reachable and there was no local cache found
    """
    cache_path = CACHE_PATH + '/' + tablename + '.pkl'
    if generate:
        if table_generator is None:
            raise ValueError('Cannot generate if no table_generator was supplied')
        df = table_generator()
        cached_table_push(df, tablename)
        return df

    if prefer_cache:
        try:
            return pd.read_pickle(cache_path)
        except FileNotFoundError:
            pass

    try:
        df = pd.read_sql_table(tablename, DB_CONNECT_STRING, **kwargs)
        return df
    except Exception as ex:
        try:
            return pd.read_pickle(cache_path)
        except FileNotFoundError:
            if table_generator is not None:
                df = table_generator()
                cached_table_push(df, tablename)
                return df

            raise FileNotFoundError(
                f'There is no connection to the database and no cache of {tablename}'
            )


def pd_to_psql(df, uri, table_name, schema_name=None, if_exists='fail', sep=','):
    """
    Load pandas Dataframe into a sql table using native postgres COPY FROM.
    Args:
        df (Dataframe): pandas Dataframe
        uri (str): postgres psycopg2 sqlalchemy database uri
        table_name (str): table to store data in
        schema_name (str): name of schema in db to write to
        if_exists (str): {`fail`, `replace`, `append`}, default `fail`. See `pandas.to_sql()` for details
        sep (str): separator for temp file, eg ',' or '\t'
    Returns:
        bool: True if loader finished
    """

    if 'psycopg2' not in uri:
        raise ValueError(
            'need to use psycopg2 uri eg postgresql+psycopg2://psqlusr:psqlpwdpsqlpwd@localhost/psqltest. install with `pip install psycopg2-binary`'
        )
    table_name = table_name.lower()
    if schema_name:
        schema_name = schema_name.lower()

    import sqlalchemy
    import io

    if schema_name is not None:
        sql_engine = sqlalchemy.create_engine(
            uri, connect_args={'options': '-csearch_path={}'.format(schema_name)}
        )
    else:
        sql_engine = sqlalchemy.create_engine(uri)
    sql_cnxn = sql_engine.raw_connection()
    cursor = sql_cnxn.cursor()

    df[:0].to_sql(
        table_name, sql_engine, schema=schema_name, if_exists=if_exists, index=False
    )

    fbuf = io.StringIO()
    df.to_csv(fbuf, index=False, header=False, sep=sep)
    fbuf.seek(0)
    cursor.copy_from(fbuf, table_name, sep=sep, null='')
    sql_cnxn.commit()
    cursor.close()

    return True


def cached_table_push(df: pd.DataFrame, tablename: str, fast: bool = True, **kwargs):
    """
    Save df to local cache file and replace the table in the database.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to push
    tablename : str
        Name of the table in the database
    fast : bool, optional
        Whether to use a faster push method or not, by default False
        True: use the fast method, which might not be as accurate
        False: use the slow method, which is more accurate
    """
    cache_path = CACHE_PATH + '/' + tablename + '.pkl'
    df.to_pickle(cache_path)
    # d6stack is way faster than pandas at inserting data to sql.
    # It exports the Dataframe to a csv and then inserts it to the database.
    try:
        if fast:
            pd_to_psql(df, DB_CONNECT_STRING, tablename, if_exists='replace')

        else:
            df.to_sql(
                tablename,
                DB_CONNECT_STRING,
                if_exists='replace',
                method='multi',
                chunksize=10_000,
                **kwargs,
            )
    except Exception as ex:
        print('could not write to database\n', 'fast=', fast, ex)
