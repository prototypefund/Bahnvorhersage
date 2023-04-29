from dataclasses import dataclass
from api.ris import RisTransferDuration, transfer_times_by_eva, RisTransfer
from typing import Literal
from neo4j import GraphDatabase, Session
from config import NEO4J_URI, NEO4J_AUTH
from helpers import StationPhillip
from tqdm import tqdm


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
    source: Literal['RIL420', 'INDOOR_ROUTING', 'EFZ']


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
        ''' + transfer_attributes + '''
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


def get_transfer_times():
    stations = StationPhillip()
    evas = set(stations.stations['eva'])

    evas_to_remove = set()
    for eva in evas:
        if eva != 8011139:
            evas_to_remove.add(eva)
        else:
            break
    
    evas = evas - evas_to_remove

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
    get_transfer_times()


if __name__ == '__main__':
    main()
