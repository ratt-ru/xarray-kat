"""Reading must not deadlock tensorstore's thread pool.

The virtual_chunked stores read from other TensorStores: the top-level
visibility, weight and flag stores from the base stores, and the base stores
from the http kvstore. A read function that blocks on such a read holds a pool
thread while it waits, and once every pool thread is waiting, nothing is left
to complete the reads (tensorstore 0.1.86 hangs here in about one load in three).
"""

import threading

import xarray

LOADS = 10


def test_loading_visibilities_weights_and_flags_does_not_deadlock(mock_archive_server):
  url, cbid, token = mock_archive_server(ntime=32, nfreq=64, nants=4)
  rdb_url = f"{url}{cbid}/{cbid}_sdp_l0.full.rdb?token={token}"

  def load():
    for _ in range(LOADS):
      dt = xarray.open_datatree(
        rdb_url, engine="xarray-kat", chunked_array_type="xarray-kat", chunks={}
      )
      for node in dt.subtree:
        if "VISIBILITY" in node.ds.data_vars:
          node.ds[["VISIBILITY", "WEIGHT", "FLAG"]].load()

  reader = threading.Thread(target=load, daemon=True)
  reader.start()
  reader.join(timeout=120)

  assert not reader.is_alive(), "loading deadlocked"
