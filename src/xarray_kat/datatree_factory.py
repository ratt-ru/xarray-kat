from __future__ import annotations

import calendar
import re
import time
from datetime import datetime, timezone
from importlib.metadata import version as importlib_version
from typing import TYPE_CHECKING, Any, Dict, Iterable, Set

import numpy as np
import numpy.typing as npt
import tensorstore as ts
import xarray
from xarray.core.indexing import LazilyIndexedArray

from xarray_kat.array import CorrProductArray
from xarray_kat.katdal_types import AutoCorrelationIndices, corrprod_to_autocorr
from xarray_kat.multiton import Multiton
from xarray_kat.stores.base_store import base_virtual_store
from xarray_kat.stores.http_store import http_store_factory
from xarray_kat.stores.vis_store import vis_virtual_store
from xarray_kat.stores.weight_store import scaled_weight_store
from xarray_kat.types import VanVleckLiteralType

if TYPE_CHECKING:
  from xarray_kat.katdal_types import SensorCache, TelstateDataSource


CACHE_SIZE = 100 * 1024 * 1024

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

# Map scan states to particples
STATE_PARTICIPLE_MAP = {
  "scan": "scanning",
  "slew": "slewing towards",
  "track": "tracking",
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


def _index_store(store: Multiton[ts.TensorStore], index) -> ts.TensorStore:
  """Helper function for delaying indexing of a TensorStore held by a Multiton"""
  return store.instance[index]


class DataTreeFactory:
  _datasource: Multiton[TelstateDataSource]
  _sensor_cache: Multiton[SensorCache]
  _scan_states: Set[str]
  _van_vleck: VanVleckLiteralType
  _endpoint: str
  _token: str | None

  def __init__(
    self,
    datasource: Multiton[TelstateDataSource],
    sensor_cache: Multiton[SensorCache],
    scan_states: Iterable[str],
    van_vleck: VanVleckLiteralType,
    endpoint: str,
    token: str | None = None,
  ):
    self._datasource = datasource
    self._sensor_cache = sensor_cache
    self._scan_states = set(scan_states)
    self._van_vleck = van_vleck
    self._endpoint = endpoint
    self._token = token

  def get_context(self, spec: Dict[str, Any]) -> ts.Context:
    return ts.Context(spec=ts.Context.Spec(spec))

  def http_store(self, data_type: str) -> Multiton[ts.TensorStore]:
    """Create an http kvstore with a path looking like ``1234567890_sdp_l0/correlator_data/``"""
    chunk_info = self._datasource.instance.telstate["chunk_info"]
    chunk_schema = chunk_info[data_type]
    path = f"{chunk_schema['prefix']}/{data_type}/"
    return Multiton(http_store_factory, self._endpoint, path, self._token, None)

  def http_backed_store(self, data_type: str) -> Multiton[ts.TensorStore]:
    """Create a virtual chunked tensorstore backed by an http kvstore
    that pulls data off the MeerKAT archive"""
    return Multiton(
      base_virtual_store,
      self.http_store(data_type),
      self._datasource.instance.telstate["chunk_info"][data_type],
      DATA_TYPE_LABELS[data_type],
      MISSING_VALUES[data_type],
      self.get_context({"cache_pool": {"total_bytes_limit": CACHE_SIZE}}),
    )

  def http_backed_vis_store(
    self,
    autocorrs: Multiton[AutoCorrelationIndices],
    data_type: str = "correlator_data",
  ) -> Multiton[ts.TensorStore]:
    return Multiton(
      vis_virtual_store,
      self.http_store(data_type),
      self._datasource.instance.telstate["chunk_info"][data_type],
      DATA_TYPE_LABELS[data_type],
      MISSING_VALUES[data_type],
      autocorrs,
      self._van_vleck,
      self.get_context({"cache_pool": {"total_bytes_limit": CACHE_SIZE}}),
    )

  def http_backed_weight_store(
    self,
    vis_store: Multiton[ts.TensorStore],
    autocorrs: Multiton[AutoCorrelationIndices],
  ) -> Multiton[ts.TensorStore]:
    return Multiton(
      scaled_weight_store,
      self.http_backed_store("weights"),
      self.http_backed_store("weights_channel"),
      vis_store,
      autocorrs,
      self._datasource,
      DATA_TYPE_LABELS["correlator_data"],
      self.get_context({"data_copy_concurrency": {"limit": 12}}),
    )

  def create(self) -> Dict[str, xarray.Dataset]:
    telstate = self._datasource.instance.telstate
    capture_block_id = telstate["capture_block_id"]
    stream_name = telstate["stream_name"]
    chunk_info = telstate["chunk_info"]

    # Time metadata
    start_time = telstate["sync_time"] + telstate["first_timestamp"]
    ntime = chunk_info["correlator_data"]["shape"][0]
    integration_time = telstate["int_time"]
    timestamps = start_time + np.arange(ntime) * integration_time

    # Observation information
    start_utc = calendar.timegm(time.gmtime(timestamps[0]))
    start_iso = datetime.fromtimestamp(start_utc, timezone.utc).isoformat()
    end_utc = calendar.timegm(time.gmtime(timestamps[-1]))
    end_iso = datetime.fromtimestamp(end_utc, timezone.utc).isoformat()
    observer = telstate["obs_params"].get("observer", "unknown")
    experiment_id = telstate["obs_params"].get("experiment_id", "unknown")

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
    autocorrs = Multiton(corrprod_to_autocorr, corrprods)
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

    corr_data_store = self.http_backed_vis_store(autocorrs)
    weight_store = self.http_backed_weight_store(corr_data_store, autocorrs)
    flag_store = self.http_backed_store("flags")

    sensor_cache = self._sensor_cache.instance
    targets = sensor_cache["Observation/target"]
    scan_indices = sensor_cache["Observation/scan_index"]
    scan_states = sensor_cache["Observation/scan_state"]

    unique_scans, scan_inv = np.unique(scan_indices, return_inverse=True)

    tree: Dict[str, xarray.Dataset] = {}

    for i, scan_index in enumerate(unique_scans):
      mask = scan_inv == i
      if (state := next(iter(scan_states[mask]))) not in self._scan_states:
        continue

      target = next(iter(targets[mask]))

      if np.all(np.diff(mask_index := np.where(mask)[0]) == 1):
        mask_index = slice(mask_index[0], mask_index[-1] + 1)

      vis_array = CorrProductArray(
        Multiton(_index_store, corr_data_store, mask_index), cp_argsort, len(pols)
      )

      weight_array = CorrProductArray(
        Multiton(_index_store, weight_store, mask_index), cp_argsort, len(pols)
      )

      flag_array = CorrProductArray(
        Multiton(_index_store, flag_store, mask_index), cp_argsort, len(pols)
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

      scan_timestamps = timestamps[mask_index]

      ds = xarray.Dataset(
        data_vars=data_vars,
        coords={
          "time": xarray.Variable("time", scan_timestamps, time_attrs),
          "field_name": xarray.Variable("time", [target.name] * len(scan_timestamps)),
          "scan_name": xarray.Variable(
            "time", [str(scan_index)] * len(scan_timestamps)
          ),
          "frequency": xarray.Variable("frequency", chan_freqs, freq_attrs),
          "polarization": xarray.Variable("polarization", pols),
          "baseline_id": xarray.Variable("baseline_id", np.arange(len(ant1_names))),
          "baseline_antenna1_name": xarray.Variable("baseline_id", ant1_names),
          "baseline_antenna2_name": xarray.Variable("baseline_id", ant2_names),
        },
        attrs={
          "creation_date": start_iso,
          "creator": {
            "software_name": "xarray-kat",
            "version": importlib_version("xarray-kat"),
          },
          "description": f"Scan {scan_index} {STATE_PARTICIPLE_MAP[state]} {target.name}",
          "observation_info": {
            "observer": observer,
            "project_uid": experiment_id,
            "release_date": end_iso,
          },
          "schema_version": "4.0.0",
          "processor_info": {"sub_type": "MEERKAT", "type": "CORRELATOR"},
          "type": "visibility",
        },
      )

      tree[f"{capture_block_id}_{stream_name}_{i:03d}"] = ds

    return tree
