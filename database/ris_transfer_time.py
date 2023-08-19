from dataclasses import dataclass
from datetime import timedelta
from typing import List, Literal

from neo4j import GraphDatabase, Session
from tqdm import tqdm

from api.ris import RisTransfer, RisTransferDuration, transfer_times_by_eva
from config import NEO4J_AUTH, NEO4J_URI
from helpers.StationPhillip import StationPhillip


@dataclass
class Platform:
    eva: int
    platform: str | None


@dataclass
class Connection:
    identical_physical_platform: bool
    frequent_traveller: RisTransferDuration
    mobility_impaired: RisTransferDuration
    occasional_traveller: RisTransferDuration
    source: Literal['RIL420', 'INDOOR_ROUTING', 'EFZ', 'FALLBACK']

    def to_dict(self):
        return {
            'identicalPhysicalPlatform': self.identical_physical_platform,
            'frequentTraveller': self.frequent_traveller.to_dict(),
            'mobilityImpaired': self.mobility_impaired.to_dict(),
            'occasionalTraveller': self.occasional_traveller.to_dict(),
            'source': self.source,
        }


def neo4j_result_to_connection(result) -> Connection:
    return Connection(
        result['c.identical_physical_platform'],
        RisTransferDuration(
            connection_duration=None,
            duration=timedelta(seconds=result['c.frequent_traveller_duration'].seconds),
            distance=result['c.frequent_traveller_distance'],
        ),
        RisTransferDuration(
            connection_duration=None,
            duration=timedelta(seconds=result['c.mobility_impaired_duration'].seconds)
            if result['c.mobility_impaired_duration'] is not None
            else None,
            distance=result['c.mobility_impaired_distance'],
        ),
        RisTransferDuration(
            connection_duration=None,
            duration=timedelta(
                seconds=result['c.occasional_traveller_duration'].seconds
            ),
            distance=result['c.occasional_traveller_distance'],
        ),
        result['c.source'],
    )


def fastest_connection(connection: List[Connection]) -> Connection:
    return min(
        connection,
        key=lambda connection: connection.frequent_traveller.duration,
    )


def get_transfer_time(
    tx: Session, start: Platform, destination: Platform
) -> Connection:
    if start.platform is None:
        start_platform_query = (
            'MATCH (from:Platform {eva: $from_eva} WHERE from.platform IS NULL)'
        )
    else:
        start_platform_query = (
            'MATCH (from:Platform {eva: $from_eva, platform: $from_platform})'
        )

    if destination.platform is None:
        destination_platform_query = (
            'MATCH (to:Platform {eva: $to_eva} WHERE to.platform IS NULL)'
        )
    else:
        destination_platform_query = (
            'MATCH (to:Platform {eva: $to_eva, platform: $to_platform})'
        )

    query = (
        start_platform_query
        + destination_platform_query
        + 'MATCH (from)-[c:TRANSFER]-(to) '
        + 'RETURN c.identical_physical_platform, c.frequent_traveller_duration, '
        + 'c.frequent_traveller_distance, c.mobility_impaired_duration, '
        + 'c.mobility_impaired_distance, c.occasional_traveller_duration, '
        + 'c.occasional_traveller_distance, c.source'
    )

    result = tx.run(
        query,
        from_eva=start.eva,
        from_platform=start.platform,
        to_eva=destination.eva,
        to_platform=destination.platform,
    )

    if result.peek() is None:
        if start.platform is not None or destination.platform is not None:
            # Try to find a connection without platform, e.g. RIL420 or EFZ
            # transfer times.
            return get_transfer_time(
                tx, Platform(start.eva, None), Platform(destination.eva, None)
            )
        return Connection(
            False,
            RisTransferDuration(
                connection_duration=None, duration=timedelta(minutes=2)
            ),
            RisTransferDuration(
                connection_duration=None, duration=timedelta(minutes=2)
            ),
            RisTransferDuration(
                connection_duration=None, duration=timedelta(minutes=2)
            ),
            'FALLBACK',
        )

    connections = [neo4j_result_to_connection(connection) for connection in result]
    return fastest_connection(connections)


def add_connection_time(tx: Session, transfer: RisTransfer):
    """Add / merge a connection time to the database.

    Parameters
    ----------
    tx : Session
        neo4j session to execute the query
    transfer : RisTransfer
        Transfer to add to the database
    """
    if transfer.from_platform is None:
        from_platform_query = 'MERGE (from:Platform {eva: $from_eva})'
    else:
        from_platform_query = (
            'MERGE (from:Platform {eva: $from_eva, platform: $from_platform})'
        )

    if transfer.to_platform is None:
        to_platform_query = 'MERGE (to:Platform {eva: $to_eva})'
    else:
        to_platform_query = 'MERGE (to:Platform {eva: $to_eva, platform: $to_platform})'

    transfer_attributes = ''

    if transfer.frequent_traveller.duration is not None:
        transfer_attributes += (
            'frequent_traveller_duration: $frequent_traveller_duration,'
        )
        if transfer.source == 'INDOOR_ROUTING':
            transfer_attributes += (
                'frequent_traveller_distance: $frequent_traveller_distance,'
            )

    if transfer.mobility_impaired.duration is not None:
        transfer_attributes += (
            'mobility_impaired_duration: $mobility_impaired_duration,'
        )
        if transfer.source == 'INDOOR_ROUTING':
            transfer_attributes += (
                'mobility_impaired_distance: $mobility_impaired_distance,'
            )

    if transfer.occasional_traveller.duration is not None:
        transfer_attributes += (
            'occasional_traveller_duration: $occasional_traveller_duration,'
        )
        if transfer.source == 'INDOOR_ROUTING':
            transfer_attributes += (
                'occasional_traveller_distance: $occasional_traveller_distance,'
            )

    tx.run(
        from_platform_query
        + to_platform_query
        + '''
        MERGE (from)-[c:TRANSFER { 
            identical_physical_platform: $identical_physical_platform,
        '''
        + transfer_attributes
        + '''
            source: $source
        }]-(to)''',
        from_eva=transfer.from_eva,
        from_platform=transfer.from_platform,
        to_eva=transfer.to_eva,
        to_platform=transfer.to_platform,
        identical_physical_platform=transfer.identical_physical_platform,
        frequent_traveller_duration=transfer.frequent_traveller.duration,
        frequent_traveller_distance=transfer.frequent_traveller.distance,
        mobility_impaired_duration=transfer.mobility_impaired.duration,
        mobility_impaired_distance=transfer.mobility_impaired.distance,
        occasional_traveller_duration=transfer.occasional_traveller.duration,
        occasional_traveller_distance=transfer.occasional_traveller.distance,
        source=transfer.source,
    )


def gather_transfer_times():
    stations = StationPhillip()
    evas = set(stations.stations['eva'])

    with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        with driver.session() as session:
            for eva in tqdm(evas, desc='Collecting transfer times'):
                connection_times = transfer_times_by_eva(eva)
                if len(connection_times) == 0:
                    print(
                        f'No transfer times found for eva {eva}, name {stations.get_name(eva=eva, date="latest")}'
                    )
                for connection_time in connection_times:
                    add_connection_time(session, connection_time)


def main():
    with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        with driver.session() as session:
            get_transfer_time(
                session, Platform(8000240, '2a/b'), Platform(8000240, '3a/b')
            )

    transfer_times_by_eva(8000141)
    gather_transfer_times()


if __name__ == '__main__':
    main()
