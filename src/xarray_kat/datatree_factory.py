from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import numpy.typing as npt
import xarray
from katsdptelstate import TelescopeState
from xarray.core.indexing import LazilyIndexedArray

from xarray_kat.array import CorrProductArray, WeightArray
from xarray_kat.meerkat_stores import http_store_factory, virtual_chunked_store
from xarray_kat.multiton import Multiton

if TYPE_CHECKING:
  import tensorstore as ts

MISSING_VALUES = {
  "correlator_data": 0j,
  "flags": 1,
  "weights": 0,
  "weights_channel": 0.0,
}

DATA_TYPE_LABELS = {
  "correlator_data": ("time", "frequency", "corrprod"),
  "flags": ("time", "frequency", "corrprod"),
  "weights": ("time", "frequency", "corrprod"),
  "weights_channel": ("time", "frequency"),
}

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
  _telstate: TelescopeState
  _endpoint: str
  _token: str | None

  def __init__(self, telstate: TelescopeState, endpoint: str, token: str | None = None):
    self._telstate = telstate
    self._endpoint = endpoint
    self._token = token

  def http_backed_store(self, data_type: str) -> Multiton[ts.TensorStore]:
    """ Create a virtual chunked tensorstore backed by an http kvstore
    that pulls data off the MeerKAT archive """
    chunk_info = self._telstate["chunk_info"]
    chunk_schema = chunk_info[data_type]
    http_store = Multiton(
      http_store_factory, self._endpoint, chunk_schema["prefix"], self._token, None
    )
    return Multiton(
      virtual_chunked_store,
      http_store,
      data_type,
      chunk_schema,
      DATA_TYPE_LABELS[data_type],
      MISSING_VALUES[data_type],
      None,
    )

  def create(self) -> Dict[str, Any]:
    capture_block_id = self._telstate["capture_block_id"]
    stream_name = self._telstate["stream_name"]
    chunk_info = self._telstate["chunk_info"]

    # Time metadata
    start_time = self._telstate["sync_time"] + self._telstate["first_timestamp"]
    ntime = chunk_info["correlator_data"]["shape"][0]
    integration_time = self._telstate["int_time"]
    timestamps = start_time + np.arange(ntime) * integration_time

    # Frequency metadata
    band = self._telstate["sub_band"]
    nchan = self._telstate["n_chans"]
    bandwidth = self._telstate["bandwidth"]
    center_freq = self._telstate["center_freq"]
    channel_width = bandwidth / nchan
    chan_freqs = (center_freq - (bandwidth / 2)) + np.arange(nchan) * channel_width

    # Correlation Product metadata
    ant_names = []

    for resource in self._telstate["sub_pool_resources"].split(","):
      try:
        ant_description = self._telstate[resource + "_observer"]
        ant_name, _ = ant_description.split(",", maxsplit=1)
        ant_names.append(ant_name)
      except (KeyError, ValueError):
        continue

    corrprods = self._telstate["bls_ordering"]
    baseline_pols = _corrprods_to_baseline_pols(corrprods)
    assert len(corrprods) == len(baseline_pols)
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

    vis_array = CorrProductArray(
      self.http_backed_store("correlator_data"), cp_argsort, len(pols)
    )
    flag_array = CorrProductArray(
      self.http_backed_store("flags"), cp_argsort, len(pols)
    )
    weight_array = WeightArray(
      self.http_backed_store("weights"),
      self.http_backed_store("weights_channel"),
      cp_argsort,
      len(pols),
    )

    name_array_map = [
      ("VISIBILITY", vis_array),
      ("WEIGHT", weight_array),
      ("FLAG", flag_array),
    ]

    data_vars = {
      n: xarray.Variable(
        a.dims,
        LazilyIndexedArray(a),
        None,
        {"preferred_chunks": dict(zip(a.dims, a.chunks))},
      )
      for n, a in name_array_map
    }
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
