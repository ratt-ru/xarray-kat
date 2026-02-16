from __future__ import annotations

import calendar
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version as importlib_version
from typing import TYPE_CHECKING, Dict, Iterable, NamedTuple, Set

import numpy as np
import tensorstore as ts
from xarray import Dataset, Variable
from xarray.core.indexing import LazilyIndexedArray

from xarray_kat.array import (
  AbstractMeerkatArchiveArray,
  DelayedBackendArray,
  ImmediateBackendArray,
)
from xarray_kat.katdal_types import corrprod_to_autocorr
from xarray_kat.multiton import Multiton
from xarray_kat.stores.vis_weight_flag_store_factory import VisWeightFlagFactory
from xarray_kat.utils import corrprods_to_baseline_pols
from xarray_kat.xkat_types import VanVleckLiteralType

if TYPE_CHECKING:
  from katpoint import Target

  from xarray_kat.katdal_types import TelstateDataProducts


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


TELESCOPE_NAME = "MeerKat"


def _index_store(store: Multiton[ts.TensorStore], index, origin=0) -> ts.TensorStore:
  """Helper function for delaying indexing of a TensorStore held by a Multiton.
  Commonly this is used to slice the time axis of the entire observation into
  separate scans/partitions. For this reason the origin is reset to zero"""
  return store.instance[index].translate_to[origin]


class ObservationMetadata(NamedTuple):
  timestamps: np.ndarray
  integration_time: float
  start_iso: str
  end_iso: str
  observer: str
  experiment_id: str
  chan_freqs: np.ndarray
  band: str
  center_freq: float
  channel_width: float
  ant1_names: np.ndarray
  ant2_names: np.ndarray
  pols: np.ndarray
  cp_argsort: np.ndarray


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

  def _build_antenna_dataset(self) -> Dataset:
    """Build the antenna xds Dataset (scan-invariant)."""
    antennas = self._data_products.instance.antennas
    antenna_polarization_types = ["X", "Y"]
    receptor_labels = ["pol_0", "pol_1"]

    return Dataset(
      data_vars={
        "ANTENNA_POSITION": Variable(
          ("antenna_name", "cartesian_pos_label"),
          np.asarray([a.position_ecef for a in antennas]),
          {
            "coordinate_system": "geocentric",
            "origin_object_name": "earth",
            "type": "location",
            "units": "m",
            "frame": "ITRS",
          },
        ),
        "ANTENNA_DISH_DIAMETER": Variable(
          "antenna_name",
          np.asarray([a.diameter for a in antennas]),
          {"type": "quantity", "units": "m"},
        ),
        "ANTENNA_EFFECTIVE_DISH_DIAMETER": Variable(
          "antenna_name",
          np.asarray([a.diameter for a in antennas]),
          {"type": "quantity", "units": "m"},
        ),
        # The reference angle for polarisation (double, 1-dim). A parallactic angle of
        # 0 means that V is aligned to x (celestial North), but we are mapping H to x
        # so we have to correct with a -90 degree rotation.
        "ANTENNA_RECEPTOR_ANGLE": Variable(
          ("antenna_name", "receptor_angle"),
          np.full((len(antennas), 2), -np.pi / 2, np.float64),
          {"type": "quantity", "units": "rad"},
        ),
      },
      coords={
        "antenna_name": Variable("antenna_name", [a.name for a in antennas]),
        "mount": Variable("antenna_name", ["ALT-AZ"] * len(antennas)),
        "telescope_name": Variable("antenna_name", [TELESCOPE_NAME] * len(antennas)),
        "station_name": Variable("antenna_name", [a.name for a in antennas]),
        "cartesian_pos_label": Variable("cartesian_pos_label", ["x", "y", "z"]),
        "polarization_type": Variable(
          ("antenna_name", "receptor_label"),
          [antenna_polarization_types] * len(antennas),
        ),
        "receptor_label": Variable("receptor_label", receptor_labels),
      },
      attrs={
        "type": "antenna",
        "overall_telescope_name": TELESCOPE_NAME,
        "relocatable_antennas": False,
      },
    )

  def _build_correlated_dataset(
    self,
    meta: ObservationMetadata,
    scan_timestamps: np.ndarray,
    target: Target,
    scan_index: int,
    state: str,
    data_vars: Dict[str, Variable],
  ) -> Dataset:
    """Build a correlated visibility Dataset for a single scan."""
    description = f"Scan {scan_index} {STATE_PARTICIPLE_MAP[state]} {target.name}"

    return Dataset(
      data_vars=data_vars,
      coords={
        "time": Variable(
          "time",
          scan_timestamps,
          {
            "type": "time",
            "units": "s",
            "scale": "utc",
            "format": "unix",
            "integration_time": {
              "attrs": {"type": "quantity", "units": "s"},
              "data": meta.integration_time,
            },
          },
        ),
        "field_name": Variable("time", [target.name] * len(scan_timestamps)),
        "scan_name": Variable("time", [str(scan_index)] * len(scan_timestamps)),
        "frequency": Variable(
          "frequency",
          meta.chan_freqs,
          {
            "type": "spectral_coord",
            "observer": "TOPO",
            "units": "Hz",
            "spectral_window_name": f"{meta.band}-band",
            "frequency_group_name": "none",
            "reference_frequency": {
              "attrs": {"type": "spectral_coord", "observer": "TOPO", "units": "Hz"},
              # TODO(sjperkins): Confirm this
              "data": meta.center_freq,
            },
            "channel_width": {
              "attrs": {"type": "quantity", "units": "Hz"},
              "data": meta.channel_width,
            },
            "effective_channel_width": "EFFECTIVE_CHANNEL_WIDTH",
          },
        ),
        "polarization": Variable("polarization", meta.pols),
        "baseline_id": Variable("baseline_id", np.arange(len(meta.ant1_names))),
        "baseline_antenna1_name": Variable("baseline_id", meta.ant1_names),
        "baseline_antenna2_name": Variable("baseline_id", meta.ant2_names),
      },
      attrs={
        "creation_date": meta.start_iso,
        "creator": {
          "software_name": "xarray-kat",
          "version": importlib_version("xarray-kat"),
        },
        "description": description,
        "observation_info": {
          "observer": meta.observer,
          "project_uid": meta.experiment_id,
          "release_date": meta.end_iso,
        },
        "schema_version": "4.0.0",
        "processor_info": {"sub_type": TELESCOPE_NAME, "type": "CORRELATOR"},
        "type": "visibility",
      },
    )

  def create(self) -> Dict[str, Dataset]:
    if self._chunks is not None:
      ArrayClass = DelayedBackendArray

      def WrappedArray(a):
        return a
    else:
      warnings.warn(
        f"xarray.open_{{groups,datatree}} was invoked without "
        f'the "chunks" argument. This should be specified, '
        f'along with the "chunked_array_type" ({self._chunked_array_type}), '
        f'which should be set to "xarray-kat" or "dask"',
        UserWarning,
      )
      ArrayClass = ImmediateBackendArray

      def WrappedArray(a):
        return LazilyIndexedArray(a)

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

    # Antenna metadata
    antennas = self._data_products.instance.antennas
    array_centre = antennas[0].array_reference_antenna()

    # Correlation Product metadata
    corrprods = telstate["bls_ordering"]
    autocorrs = Multiton(corrprod_to_autocorr, corrprods)
    baseline_pols = corrprods_to_baseline_pols(corrprods)
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
    concat_ant_names = np.concatenate([ant1_names, ant2_names])
    _, inv = np.unique(concat_ant_names, return_inverse=True)
    ant1_index = inv[len(inv) // 2 :]
    ant2_index = inv[: len(inv) // 2]

    pols = np.array([HV_TO_LINEAR_MAP[p] for p in cp_pols[: len(upols)]], dtype=str)

    meta = ObservationMetadata(
      timestamps=timestamps,
      integration_time=integration_time,
      start_iso=start_iso,
      end_iso=end_iso,
      observer=observer,
      experiment_id=experiment_id,
      chan_freqs=chan_freqs,
      band=band,
      center_freq=center_freq,
      channel_width=channel_width,
      ant1_names=ant1_names,
      ant2_names=ant2_names,
      pols=pols,
      cp_argsort=cp_argsort,
    )

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

    antenna_ds = self._build_antenna_dataset()
    tree: Dict[str, Dataset] = {}

    for i, scan_index in enumerate(unique_scans):
      mask = scan_inv == i
      if (state := next(iter(scan_states[mask]))) not in self._scan_states:
        continue

      target = next(iter(targets[mask]))

      if np.all(np.diff(mask_index := np.where(mask)[0]) == 1):
        mask_index = slice(mask_index[0], mask_index[-1] + 1)

      vis_array = ArrayClass(
        Multiton(_index_store, corr_data_store, mask_index), meta.cp_argsort, len(pols)
      )

      weight_array = ArrayClass(
        Multiton(_index_store, weight_store, mask_index), meta.cp_argsort, len(pols)
      )

      flag_array = ArrayClass(
        Multiton(_index_store, flag_store, mask_index), meta.cp_argsort, len(pols)
      )

      data_vars = {
        n: Variable(
          a.dims,
          WrappedArray(a),
          None,
          {"preferred_chunks": self.merge_chunks(n, a)},
        )
        for n, a in [
          ("VISIBILITY", vis_array),
          ("WEIGHT", weight_array),
          ("FLAG", flag_array),
        ]
      }

      # Pre-calculate UVW coordinates
      scan_timestamps = meta.timestamps[mask_index]
      # Compute UVW coordinates for a chunk of timesteps.
      uvw_ant = target.uvw(antennas, scan_timestamps, array_centre)
      # Permute from axis, time, antenna to time, antenna, axis
      uvw_ant = np.transpose(uvw_ant, (1, 2, 0))
      # Compute baseline UVW coordinates from per-antenna coordinates.
      # The sign convention matches `CASA`_, rather than the
      # Measurement Set `definition`_.
      # .. _CASA: https://casa.nrao.edu/Memos/CoordConvention.pdf
      # .. _definition: https://casa.nrao.edu/Memos/229.html#SECTION00064000000000000000
      uvw_coordinates = np.take(uvw_ant, ant1_index, axis=1) - np.take(
        uvw_ant, ant2_index, axis=1
      )

      flag_p_chunks = data_vars["FLAG"].encoding["preferred_chunks"]
      uvw_preferred_chunks = {
        "time": flag_p_chunks["time"],
        "baseline_id": flag_p_chunks["baseline_id"],
        "uvw_label": 3,
      }

      data_vars["UVW"] = Variable(
        ("time", "baseline_id", "uvw_label"),
        uvw_coordinates,
        {"type": "uvw", "units": "m", "frame": "fk5"},
        {"preferred_chunks": uvw_preferred_chunks},
      )

      correlated_ds = self._build_correlated_dataset(
        meta,
        scan_timestamps,
        target,
        scan_index,
        state,
        data_vars,
      )

      correlated_node_name = f"{self._data_products.instance.name}_{i:03d}"
      tree[correlated_node_name] = correlated_ds
      tree[f"{correlated_node_name}/antenna_xds"] = antenna_ds

    return tree
