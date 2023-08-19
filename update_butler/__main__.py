import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.isfile("/mnt/config/config.py"):
    sys.path.append("/mnt/config/")
from helpers import RtdRay
from data_analysis import data_stats
from datetime import datetime, timedelta, time
from data_analysis.per_station import PerStationOverTime
from dask.distributed import Client
from ml_models.xgboost_multi_model import train_models, test_models
from database import MlStat

if __name__ == '__main__':
    # Print Logo
    import helpers.bahn_vorhersage

    print("Init")

    with Client(n_workers=4, threads_per_worker=1, memory_limit='16GB') as client:
        print("Done")

        print("Refreshing local Cache...")
        # TODO switch to RtdRay.upgrade_rtd()
        RtdRay.download_rtd()

        print("Done")

    print("Generating Statistics...")

    print("--Overview")

    data_stats.load_stats(
        table_generator=data_stats.stats_generator,
        generate=True,
    )

    print("--Done")

    print("--Per Station Data")

    rtd_df = RtdRay.load_data(
        columns=[
            "ar_pt",
            "dp_pt",
            "station",
            "ar_delay",
            "ar_happened",
            "dp_delay",
            "dp_happened",
            "lat",
            "lon",
        ],
        min_date=datetime(2021, 1, 1),
    )

    PerStationOverTime(rtd_df, generate=True)
    print("Done")

    midnight = datetime.combine(datetime.now().date(), time.min)

    # Setting `threads_per_worker` is very important as Dask will otherwise
    # create as many threads as cpu cores which is to munch for big cpus with small RAM
    with Client(n_workers=min(10, os.cpu_count() // 4), threads_per_worker=2) as client:
        print('Testing old ml models...')
        test_result = test_models(
            max_date=midnight,
            min_date=midnight - timedelta(days=1),
            return_status=True,
            obstacles=False,
        )
        MlStat.add_stats(test_result)
        print('Done')

        print("Training new ml models...")
        train_models(
            min_date=midnight - timedelta(days=7 * 6),
            return_status=True,
            obstacles=False,
        )
        print("Done")
