import re
from typing import Any, Dict

import numpy as np
import numpy.typing as npt
import xarray
from katsdptelstate import TelescopeState
from xarray.core.indexing import LazilyIndexedArray

from xarray_kat.array import CorrProductTensorstore
from xarray_kat.tensorstore_factory import StoreFactory

CORRPROD_REGEX = re.compile(
  r"(?P<dish>[mMsSeE])(?P<number>\d+)(?P<polarization>[hHvV])"
)

HV_TO_LINEAR_MAP = {
  "hh": "XX",
  "hv": "XY",
  "vh": "YX",
  "vv": "YY",
}


def _corrprods_to_baseline_pols(corrprods: npt.NDArray):
  """Split correlation products of the form ``["m001v", "m002h"]`` into
  tuples of the form ``(("m001", "m002"), "vh")``"""
  result = []

  for cp1, cp2 in corrprods:
    if (
      (cp1_match := re.match(CORRPROD_REGEX, cp1)) is None
      or (cp1_dish := cp1_match.group("dish")) is None
      or (cp1_nr := cp1_match.group("number")) is None
      or (cp1_pol := cp1_match.group("polarization")) is None
    ):
      raise ValueError(f"{cp1} is not a valid correlation product string {cp1_match}")
    if (
      (cp2_match := re.match(CORRPROD_REGEX, cp2)) is None
      or (cp2_dish := cp2_match.group("dish")) is None
      or (cp2_nr := cp2_match.group("number")) is None
      or (cp2_pol := cp2_match.group("polarization")) is None
    ):
      raise ValueError(f"{cp2} is not a valid correlation product string {cp2_match}")

    result.append(
      (
        f"{cp1_dish.lower()}{cp1_nr}",
        f"{cp2_dish.lower()}{cp2_nr}",
        f"{cp1_pol.lower()}{cp2_pol.lower()}",
      )
    )

  return result


class GroupFactory:
  @classmethod
  def make(
    cls, telstate: TelescopeState, endpoint: str, token: str | None = None
  ) -> Dict[str, Any]:
    capture_block_id = telstate["capture_block_id"]
    stream_name = telstate["stream_name"]
    chunk_info = telstate["chunk_info"]

    # Time metadata
    start_time = telstate["sync_time"] + telstate["first_timestamp"]
    ntime = chunk_info["correlator_data"]["shape"][0]
    integration_time = telstate["int_time"]
    timestamps = start_time + np.arange(ntime) * integration_time

    # Frequency metadata
    band = telstate["sub_band"]
    nchan = telstate["n_chans"]
    bandwidth = telstate["bandwidth"]
    center_freq = telstate["center_freq"]
    channel_width = bandwidth / nchan
    chan_freqs = (center_freq - (bandwidth / 2)) + np.arange(nchan) * channel_width

    # Correlation Product metadata
    ant_names = []

    for resource in telstate["sub_pool_resources"].split(","):
      try:
        ant_description = telstate[resource + "_observer"]
        ant_name, _ = ant_description.split(",", maxsplit=1)
        ant_names.append(ant_name)
      except (KeyError, ValueError):
        continue

    corrprods = telstate["bls_ordering"]
    baseline_pols = _corrprods_to_baseline_pols(corrprods)
    cp_argsort = np.array(
      sorted(range(len(baseline_pols)), key=lambda i: baseline_pols[i])
    )

    cp_ant1_names, cp_ant2_names, cp_pols = zip(*(baseline_pols[i] for i in cp_argsort))
    upols = list(sorted(set(cp_pols)))
    if len(baseline_pols) % len(upols) != 0:
      raise ValueError(
        f"Polarizations {len(upols)} don't divide "
        f"correlation products {len(baseline_pols)} exactly"
      )

    ant1_names = np.array(cp_ant1_names[:: len(upols)], dtype=str)
    ant2_names = np.array(cp_ant2_names[:: len(upols)], dtype=str)
    pols = np.array([HV_TO_LINEAR_MAP[p] for p in cp_pols[: len(upols)]], dtype=str)

    stores = StoreFactory.make(telstate, endpoint, token)
    data_vars: Dict[str, xarray.Variable] = {}

    meerkat_to_msv4_name = {
      "correlator_data": "VISIBILITY",
      "flags": "FLAG",
      "weights": "WEIGHT",
    }

    for mk_name, msv4_name in meerkat_to_msv4_name.items():
      ts_store = stores[mk_name]
      dims = ts_store.domain.labels[:2]
      dims = (dims[0], "baseline_id", dims[1], "polarization")
      tf_chunks = ts_store.chunk_layout.read_chunk.shape[:2]
      chunks = (tf_chunks[0], len(ant1_names), tf_chunks[1], len(pols))
      array = LazilyIndexedArray(
        CorrProductTensorstore(ts_store, cp_argsort, len(pols))
      )
      data_vars[msv4_name] = xarray.Variable(
        dims, array, None, {"preferred_chunks": dict(zip(dims, chunks))}
      )

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
      data_vars=data_vars,
      coords={
        "time": xarray.Variable("time", timestamps, time_attrs),
        "frequency": xarray.Variable("frequency", chan_freqs, freq_attrs),
        "polarization": xarray.Variable("polarization", pols),
        "baseline_id": xarray.Variable("baseline_id", np.arange(len(ant1_names))),
        "baseline_antenna1_name": xarray.Variable("baseline_id", ant1_names),
        "baseline_antenna2_name": xarray.Variable("baseline_id", ant2_names),
      },
    )

    return {f"{capture_block_id}_{stream_name}": ds}
