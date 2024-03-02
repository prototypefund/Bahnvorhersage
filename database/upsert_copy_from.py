import csv
import io
from typing import List

import sqlalchemy
from sqlalchemy.orm import Session


class psql_csv_dialect(csv.Dialect):
    """Describe the usual properties of Unix-generated CSV files."""

    delimiter = ','
    quotechar = '"'
    doublequote = True
    skipinitialspace = False
    lineterminator = '\n'
    quoting = csv.QUOTE_MINIMAL


def tuples_to_csv(tuples: List[tuple]) -> str:
    """Convert a list of tuples to a csv string

    Parameters
    ----------
    tuples : List[tuple]
        list of tuples to convert to csv

    Returns
    -------
    str
        csv string
    """
    fbuf = io.StringIO()
    writer = csv.writer(fbuf, dialect=psql_csv_dialect)
    writer.writerows(tuples)
    fbuf.seek(0)
    return fbuf.read()


def upsert_copy_from(
    table: sqlalchemy.schema.Table,
    temp_table: sqlalchemy.schema.Table,
    csv: str,
    engine: sqlalchemy.engine.Engine,
):
    sql_clear = f'TRUNCATE {temp_table.fullname}'
    with Session(engine) as session:
        session.execute(sqlalchemy.text(sql_clear))
        session.commit()

    # Use copy from to insert the date into a temporary table
    sql_cnxn = engine.raw_connection()
    cursor = sql_cnxn.cursor()

    fbuf = io.StringIO(csv)
    cursor.copy_from(fbuf, temp_table.fullname, sep=',', null='')
    sql_cnxn.commit()
    cursor.close()
    sql_cnxn.close()

    # INSTERT INTO table SELECT * FROM temp_table ON CONFLICT DO UPDATE
    # Insert the data from the temporary table into the real table using raw sql
    update_cols = [c.name for c in table.c if c not in list(table.primary_key.columns)]

    sql_insert = f"""
        INSERT INTO {table.fullname}
        SELECT * FROM {temp_table.fullname}
        ON CONFLICT ({', '.join(table.primary_key.columns.keys())}) DO UPDATE
        SET {', '.join([f'{col} = excluded.{col}' for col in update_cols])}
    """

    with Session(engine) as session:
        session.execute(sqlalchemy.text(sql_insert))
        # session.execute(sqlalchemy.text(sql_clear))
        session.commit()
