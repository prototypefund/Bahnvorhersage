import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.isfile("/mnt/config/config.py"):
    sys.path.append("/mnt/config/")
from datetime import datetime, timedelta
from dask.distributed import Client
from ml_models.xgboost_multi_model import train_models, test_models
from tqdm import tqdm
import json

START_DATE = datetime(2022, 5, 1) + timedelta(days=28+24+57)
END_DATE = datetime(2022, 9, 1)

dates = [START_DATE + timedelta(days=i) for i in range((END_DATE - START_DATE).days + 1)]
data = json.load(open("test.json", "r"))


if __name__ == '__main__':
    from helpers import bahn_vorhersage

    # Setting `threads_per_worker` is very important as Dask will otherwise
    # create as many threads as cpu cores which is to munch for big cpus with small RAM
    with Client(n_workers=min(10, os.cpu_count() // 4), threads_per_worker=2) as client:
        for date in tqdm(dates):
            try:
                print("Training new ml models...")
                train_models(
                    max_date=date,
                    min_date=date - timedelta(days=7 * 6),
                    return_status=True,
                    obstacles=False,
                )
            except Exception as e:
                print("Error while training models:", e)

            try:
                print('Testing old ml models...')
                test_result = test_models(
                    max_date=date,
                    min_date=date - timedelta(days=1),
                    return_status=True,
                    obstacles=False,
                )
                data.update({date.strftime("%Y-%m-%d"): test_result})
            except Exception as e:
                print("Error while testing models:", e)
            
            json.dump(data, open("test.json", "w"))
