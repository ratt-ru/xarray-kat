from __future__ import annotations

import calendar
import re
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version as importlib_version
from typing import TYPE_CHECKING, Dict, Iterable, Set

import numpy as np
import numpy.typing as npt
import tensorstore as ts
import xarray
from xarray.core.indexing import LazilyIndexedArray

from xarray_kat.array import (
  AbstractMeerkatArchiveArray,
  DelayedCorrProductArray,
  ImmediateCorrProductArray,
)
from xarray_kat.katdal_types import corrprod_to_autocorr
from xarray_kat.multiton import Multiton
from xarray_kat.stores.vis_weight_flag_store_factory import VisWeightFlagFactory
from xarray_kat.types import VanVleckLiteralType

if TYPE_CHECKING:
  from xarray_kat.katdal_types import TelstateDataProducts

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
  _chunks: Dict[str, int] | None
  _chunked_array_type: str | None
  _preferred_chunks: Dict[str, int]
  _data_products: Multiton[TelstateDataProducts]
  _scan_states: Set[str]
  _applycal: str | Iterable[str]
  _van_vleck: VanVleckLiteralType
  _endpoint: str
  _token: str | None

  def __init__(
    self,
    chunks: Dict[str, int] | None,
    chunked_array_type: str | None,
    preferred_chunks: Dict[str, int],
    data_products: Multiton[TelstateDataProducts],
    applycal: str | Iterable[str],
    scan_states: Iterable[str],
    van_vleck: VanVleckLiteralType,
    endpoint: str,
    token: str | None = None,
  ):
    self._chunks = chunks
    self._chunked_array_type = chunked_array_type
    self._preferred_chunks = preferred_chunks
    self._data_products = data_products
    self._applycal = applycal
    self._scan_states = set(scan_states)
    self._van_vleck = van_vleck
    self._endpoint = endpoint
    self._token = token

  def merge_chunks(
    self, name: str, array: AbstractMeerkatArchiveArray
  ) -> Dict[str, int]:
    """Merge underlying store chunks with preferred chunks"""
    preferred_chunks = tuple(
      self._preferred_chunks.get(al, ac) for ac, al in zip(array.chunks, array.dims)
    )
    chunks = {}

    for d, (store_chunk, dim, preferred_chunk) in enumerate(
      zip(array.chunks, array.dims, preferred_chunks)
    ):
      if (dim_chunk := max(store_chunk, preferred_chunk)) % store_chunk != 0:
        warnings.warn(
          f"Array {name}'s preferred chunks {preferred_chunks} "
          f"are not divisible by the underlying store chunks "
          f"{array.chunks} in dimension {d} ({dim}). "
          f"This is benign but may result in repeated requests for data.",
          UserWarning,
        )

      chunks[dim] = dim_chunk

    return chunks

  def create(self) -> Dict[str, xarray.Dataset]:
    if self._chunks is not None:
      ArrayClass = DelayedCorrProductArray
    else:
      warnings.warn(
        f"xarray.open_{{groups,datatree}} was invoked without "
        f'the "chunks" argument. This should be specified, '
        f'along with the "chunked_array_type" ({self._chunked_array_type}), '
        f'which should be set to "xarray-kat" or "dask"',
        UserWarning,
      )
      ArrayClass = ImmediateCorrProductArray

    telstate = self._data_products.instance.telstate
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

    vfw_factory = VisWeightFlagFactory(
      self._data_products,
      autocorrs,
      self._applycal,
      self._van_vleck,
      self._endpoint,
      self._token,
    )

    corr_data_store = vfw_factory.vis()
    weight_store = vfw_factory.weight()
    flag_store = vfw_factory.flag()

    sensor_cache = self._data_products.instance.sensor_cache
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

      vis_array = ArrayClass(
        Multiton(_index_store, corr_data_store, mask_index), cp_argsort, len(pols)
      )

      weight_array = ArrayClass(
        Multiton(_index_store, weight_store, mask_index), cp_argsort, len(pols)
      )

      flag_array = ArrayClass(
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
          {"preferred_chunks": self.merge_chunks(n, a)},
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

      tree[f"{self._data_products.instance.name}_{i:03d}"] = ds

    return tree
