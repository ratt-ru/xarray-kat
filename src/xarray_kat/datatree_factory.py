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

    band = telstate["sub_band"]
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

    def schunks(n):
      return stores[n].chunk_layout.read_chunk.shape

    dims = ("time", "baseline_id", "frequency", "polarization")
    vis_encoding = {"preferred_chunks": dict(zip(dims, schunks("correlator_data")))}
    flag_encoding = {"preferred_chunks": dict(zip(dims, schunks("flags")))}
    weight_encoding = {"preferred_chunks": dict(zip(dims, schunks("weights")))}

    time_attrs = {
      "type": "time",
      "units": "s",
      "scale": "utc",
      "format": "unix",
      "integration_time": {
        "attrs": {"type": "quanitity", "units": "s"},
        "data": integration_time,
      },
    }

    freq_attrs = {
      "type": "spectral_coord",
      "observer": "TOPO",
      "units": "Hz",
      "spectral_window_name": f"{band}-band",
      "frequency_group_name": "none",
      "reference_frequency": {
        "attrs": {"type": "spectral_coord", "observer": "TOPO", "units": "Hz"},
        # TODO(sjperkins): Confirm this
        "data": center_freq,
      },
      "channel_width": {
        "attrs": {"type": "quantity", "units": "Hz"},
        "data": channel_width,
      },
      "effective_channel_width": "EFFECTIVE_CHANNEL_WIDTH",
    }

    ds = xarray.Dataset(
      {
        "VISIBILITY": xarray.Variable(dims, vis, None, vis_encoding),
        "FLAG": xarray.Variable(dims, flags, None, flag_encoding),
        "WEIGHT": xarray.Variable(dims, weights, None, weight_encoding),
      },
      coords={
        "time": xarray.Variable("time", timestamps, time_attrs),
        "frequency": xarray.Variable("frequency", chan_freqs, freq_attrs),
        "polarization": xarray.Variable("polarization", ["XX", "XY", "YX", "YY"]),
        "baseline_id": xarray.Variable("baseline_id", np.arange(len(ant1))),
        "baseline_antenna1_name": xarray.Variable("baseline_id", ant_names[ant1]),
        "baseline_antenna2_name": xarray.Variable("baseline_id", ant_names[ant2]),
      },
    )

    return {f"{capture_block_id}_{stream_name}": ds}
