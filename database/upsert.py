import random
import time
from typing import List

import sqlalchemy
from sqlalchemy.dialects.postgresql import Insert, insert
from sqlalchemy.orm import Session

from helpers.batcher import batcher


def create_upsert_statement(
    table: sqlalchemy.sql.schema.Table, rows: List[dict]
) -> Insert:
    stmt = insert(table).values(rows)

    update_cols = [c.name for c in table.c if c not in list(table.primary_key.columns)]

    on_conflict_stmt = stmt.on_conflict_do_update(
        index_elements=table.primary_key.columns,
        set_={k: getattr(stmt.excluded, k) for k in update_cols},
    )

    return on_conflict_stmt


def upsert_base(
    session: sqlalchemy.orm.Session,
    table: sqlalchemy.sql.schema.Table,
    rows: List[dict],
) -> None:
    """Upsert rows to table using session

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        A session used to execute the upsert
    table : sqlalchemy.sql.schema.Table
        The table to upsert the rows to
    rows : List[dict]
        The actual data to upsert
    """
    on_conflict_stmt = create_upsert_statement(table, rows)

    session.execute(on_conflict_stmt)


def do_nothing_upsert_base(
    session: sqlalchemy.orm.Session,
    table: sqlalchemy.sql.schema.Table,
    rows: List[dict],
) -> None:
    """Upsert rows to table using session with do nothing if exists logic

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        A session used to execute the upsert
    table : sqlalchemy.sql.schema.Table
        The table to upsert the rows to
    rows : List[dict]
        The actual data to upsert
    """
    stmt = insert(table).values(rows)

    on_conflict_stmt = stmt.on_conflict_do_nothing(
        index_elements=table.primary_key.columns
    )

    session.execute(on_conflict_stmt)


def upsert_with_retry(
    engine: sqlalchemy.engine.Engine,
    table: sqlalchemy.sql.schema.Table,
    rows: List[dict],
):
    with Session(engine) as session:
        for row_batch in batcher(rows, 5_000):
            while True:
                try:
                    upsert_base(session, table, row_batch)
                    session.commit()
                    break
                except sqlalchemy.exc.OperationalError as ex:
                    if 'deadlock detected' in ex.args[0]:
                        timeout = random.randint(20, 80)
                        print(
                            f'{table.fullname} deadlock detected. Waiting {timeout} seconds.'
                        )
                        time.sleep(timeout)
                    elif 'QueryCanceled' in ex.args[0]:
                        print(
                            f'{table.fullname} QueryCanceled due to timeout. Retrying.'
                        )
                        time.sleep(120)
                    else:
                        raise ex
