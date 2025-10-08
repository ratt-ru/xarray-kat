from typing import Any, Dict

import numpy as np
import xarray
from katsdptelstate import TelescopeState
from xarray.core.indexing import LazilyIndexedArray

from xarray_kat.array import TensorstoreArray
from xarray_kat.tensorstore_factory import StoreFactory


class GroupFactory:
  @classmethod
  def make(
    cls, telstate: TelescopeState, endpoint: str, token: str | None = None
  ) -> Dict[str, Any]:
    capture_block_id = telstate["capture_block_id"]
    stream_name = telstate["stream_name"]
    chunk_info = telstate["chunk_info"]
    start_time = telstate["sync_time"] + telstate["first_timestamp"]
    integration_time = telstate["int_time"]
    ntime = chunk_info["correlator_data"]["shape"][0]

    # band = telstate["sub_band"]
    nchan = telstate["n_chans"]
    bandwidth = telstate["bandwidth"]
    center_freq = telstate["center_freq"]
    channel_width = bandwidth / nchan
    timestamps = start_time + np.arange(ntime) * integration_time
    chan_freqs = (center_freq - (bandwidth / 2)) + np.arange(nchan) * channel_width

    ant_names = []

    for resource in telstate["sub_pool_resources"].split(","):
      try:
        ant_description = telstate[resource + "_observer"]
        ant_name, _ = ant_description.split(",", maxsplit=1)
        ant_names.append(ant_name)
      except (KeyError, ValueError):
        continue

    # TODO
    # This should align with sorted corrprods,
    # but we should be more accurate
    ant_names = np.array(sorted(ant_names))
    ant1, ant2 = np.triu_indices(len(ant_names), 0)

    stores = StoreFactory.make(telstate, endpoint, token)

    vis = LazilyIndexedArray(TensorstoreArray(stores["correlator_data"]))
    flags = LazilyIndexedArray(TensorstoreArray(stores["flags"]))
    weights = LazilyIndexedArray(TensorstoreArray(stores["weights"]))

    dims = ("time", "baseline_id", "frequency", "polarization")

    ds = xarray.Dataset(
      {
        "VISIBILITY": xarray.Variable(dims, vis),
        "FLAG": xarray.Variable(dims, flags),
        "WEIGHT": xarray.Variable(dims, weights),
      },
      coords={
        "time": xarray.Variable("time", timestamps),
        "frequency": xarray.Variable("frequency", chan_freqs),
        "polarization": xarray.Variable("polarization", ["XX", "XY", "YX", "YY"]),
        "baseline_id": xarray.Variable("baseline_id", np.arange(len(ant1))),
        "baseline_antenna1_name": xarray.Variable("baseline_id", ant_names[ant1]),
        "baseline_antenna2_name": xarray.Variable("baseline_id", ant_names[ant2]),
      },
    )

    return {f"{capture_block_id}_{stream_name}": ds}
