from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING, Any, Dict, Iterable, cast

import tensorstore as ts

from xarray_kat.multiton import Multiton
from xarray_kat.stores.base_store import base_virtual_store
from xarray_kat.stores.flag_store import final_flag_store
from xarray_kat.stores.http_store import http_store_factory
from xarray_kat.stores.visibility_stores import (
  base_visibility_virtual_store,
  final_visibility_virtual_store,
)
from xarray_kat.stores.weight_store import scaled_weight_store
from xarray_kat.xkat_types import ArchiveArrayMetadata, VanVleckLiteralType

if TYPE_CHECKING:
  from katsdptelstate.telescope_state import TelescopeState

  from xarray_kat.katdal_types import AutoCorrelationIndices, TelstateDataProducts

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


class VisWeightFlagFactory:
  """Generates TensorStores representing visibilities, weights and flags.

  The final values in each store are interdependent on each other and
  require the construction of base stores that apply data transformations
  at different points.
  """

  _data_products: Multiton[TelstateDataProducts]
  _autocorrs: Multiton[AutoCorrelationIndices]
  _applycal: str | Iterable[str]
  _van_vleck: VanVleckLiteralType
  _endpoint: str
  _token: str | None

  _vis: Multiton[ts.TensorStore] | None
  _weight: Multiton[ts.TensorStore] | None
  _flag: Multiton[ts.TensorStore] | None

  def __init__(
    self,
    data_products: Multiton[TelstateDataProducts],
    autocorrs: Multiton[AutoCorrelationIndices],
    applycal: str | Iterable[str],
    van_vleck: VanVleckLiteralType,
    endpoint: str,
    token: str | None = None,
  ):
    self._data_products = data_products
    self._autocorrs = autocorrs
    self._applycal = applycal
    self._van_vleck = van_vleck
    self._endpoint = endpoint
    self._token = token

    self._vis = None
    self._weight = None
    self._flag = None

  def get_context(self, spec: Dict[str, Any]) -> ts.Context:
    return ts.Context(spec=ts.Context.Spec(spec))

  def http_store(self, array_meta: ArchiveArrayMetadata) -> Multiton[ts.TensorStore]:
    """Create an http kvstore with a path of the form
    ``1234567890_sdp_l0/correlator_data/``"""
    path = f"{array_meta.prefix}/{array_meta.name}/"
    return Multiton(http_store_factory, self._endpoint, path, self._token, None)

  def http_backed_store(
    self, array_meta: ArchiveArrayMetadata
  ) -> Multiton[ts.TensorStore]:
    """Create a virtual chunked tensorstore backed by an http kvstore
    that retrieves data from the MeerKAT archive"""
    return Multiton(
      base_virtual_store,
      self.http_store(array_meta),
      array_meta,
      self.get_context({"cache_pool": {"total_bytes_limit": CACHE_SIZE}}),
    )

  def array_metadata(self, telstate: TelescopeState) -> Dict[str, ArchiveArrayMetadata]:
    """Derive metadata for the main MeerKAT data arrays

    Also performs consistency checks"""
    chunk_info = telstate["chunk_info"]
    array_meta = {
      dt: ArchiveArrayMetadata(
        dt,
        MISSING_VALUES[dt],
        DATA_TYPE_LABELS[dt],
        schema["prefix"],
        schema["chunks"],
        schema["dtype"],
      )
      for dt, schema in chunk_info.items()
    }
    label_keys = list(sorted(DATA_TYPE_LABELS.keys()))

    if list(sorted(array_meta.keys())) != label_keys:
      raise ValueError(
        f"Mismatch between telstate arrays {array_meta.keys()} "
        f"and expected arrays {label_keys}."
      )

    if (
      len(
        cp_shapes := {v.shape for k, v in array_meta.items() if k != "weights_channel"}
      )
      != 1
    ):
      raise ValueError(
        f"Array shapes ({cp_shapes}) involving correlation products  "
        f"are not consistent: {array_meta}."
      )

    cp_shape = next(iter(cp_shapes))

    if cp_shape[:-1] != (cw_shape := array_meta["weights_channel"].shape):
      raise ValueError(
        f"weights_channel shape {cw_shape} "
        f"does not match corrprod array "
        f"shapes {cp_shape}."
      )

    return array_meta

  def create(self) -> None:
    telstate = self._data_products.instance.telstate
    array_meta = self.array_metadata(telstate)

    # Create the base visibility store
    base_vis = Multiton(
      base_visibility_virtual_store,
      self.http_store(array_meta["correlator_data"]),
      array_meta["correlator_data"],
      self._autocorrs,
      self._van_vleck,
      self.get_context({"cache_pool": {"total_bytes_limit": CACHE_SIZE}}),
    )

    # Create the base integer weights, channel weights and flag stores
    base_int_weights = self.http_backed_store(array_meta["weights"])
    base_chan_weights = self.http_backed_store(array_meta["weights_channel"])
    base_flags = self.http_backed_store(array_meta["flags"])

    # Create a top level context for performing data copies
    top_level_thread_ctx = self.get_context(
      {"data_copy_concurrency": {"limit": mp.cpu_count()}}
    )

    # Create a top level metadata object to ensure that
    # all top level stores have the same chunking as
    # the visibilities. This ensures consistent
    # chunking along dimensions for the xarray layer.
    # The weight and flag stores use different dtypes
    final_metadata = array_meta["correlator_data"].copy()

    # Create the top level visibility store
    self._vis = Multiton(
      final_visibility_virtual_store,
      base_vis,
      final_metadata,
      self._data_products,
      top_level_thread_ctx,
    )

    # Create the top level weight store
    self._weight = Multiton(
      scaled_weight_store,
      base_int_weights,
      base_chan_weights,
      base_vis,
      self._data_products,
      self._autocorrs,
      final_metadata.copy(dtype=array_meta["weights_channel"].dtype),
      telstate.get("needs_weight_power_scale", False),
      top_level_thread_ctx,
    )

    # Create the top level flag store
    self._flag = Multiton(
      final_flag_store,
      base_flags,
      self._data_products,
      final_metadata.copy(dtype=array_meta["flags"].dtype),
      top_level_thread_ctx,
    )

  def vis(self) -> Multiton[ts.TensorStore]:
    """Return the visibility TensorStore"""
    if self._vis is None:
      self.create()

    return cast(Multiton[ts.TensorStore], self._vis)

  def weight(self) -> Multiton[ts.TensorStore]:
    """Return the weight TensorStore"""
    if self._weight is None:
      self.create()

    return cast(Multiton[ts.TensorStore], self._weight)

  def flag(self) -> Multiton[ts.TensorStore]:
    """Return the flag TensorStore"""
    if self._flag is None:
      self.create()

    return cast(Multiton[ts.TensorStore], self._flag)
