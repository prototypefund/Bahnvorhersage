from config import CACHE_PATH
from database import Rtd
from helpers import RtdRay


class ConnectionRay(Rtd):
    def __init__(self):
        self.AR_PATH = CACHE_PATH + '/connecting_trains_ar'
        self.DP_PATH = CACHE_PATH + '/connecting_trains_dp'

    def load_ar(self, **kwargs):
        return RtdRay.load_for_ml_model(
            path=self.AR_PATH, load_categories=False, label_encode=False, **kwargs
        )

    def load_dp(self, **kwargs):
        return RtdRay.load_for_ml_model(
            path=self.DP_PATH, load_categories=False, label_encode=False, **kwargs
        )


if __name__ == '__main__':
    from dask.distributed import Client

    from helpers.bahn_vorhersage import COLORFUL_ART

    print(COLORFUL_ART)

    client = Client(n_workers=2, threads_per_worker=1)

    connection_ray = ConnectionRay()
    rtd = connection_ray.load_ar()
    print(f'length of data: {len(rtd)} rows')
    # RtdRay.refresh_local_buffer()
    # rtd = RtdRay.load_for_ml_model()
