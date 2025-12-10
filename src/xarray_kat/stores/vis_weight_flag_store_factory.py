from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import tensorstore as ts

from xarray_kat.multiton import Multiton
from xarray_kat.stores.base_store import base_virtual_store
from xarray_kat.stores.http_store import http_store_factory
from xarray_kat.stores.visibility_stores import base_visibility_virtual_store
from xarray_kat.stores.weight_store import scaled_weight_store

if TYPE_CHECKING:
  from xarray_kat.katdal_types import AutoCorrelationIndices, TelstateDataProducts
  from xarray_kat.types import VanVleckLiteralType

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

  The final values in each stores are interdependent on each other and
  require the construction of base stores that apply data transformations
  at different times.

  """
  _data_products: Multiton[TelstateDataProducts]
  _autocorrs: Multiton[AutoCorrelationIndices]
  _van_vleck: VanVleckLiteralType
  _applycal: str
  _endpoint: str
  _token: str | None

  _vis: ts.TensorStore | None
  _weight: ts.TensorStore | None
  _flag: ts.TensorStore | None

  def __init__(
    self,
    data_products: Multiton[TelstateDataProducts],
    autocorrs: Multiton[AutoCorrelationIndices],
    van_vleck: VanVleckLiteralType,
    endpoint: str,
    token: str | None = None
  ):
    self._data_products = data_products
    self._autocorrs = autocorrs
    self._van_vleck = van_vleck
    self._endpoint = endpoint
    self._token = token

    self._vis = None
    self._weight = None
    self._flag = None

  def get_context(self, spec: Dict[str, Any]) -> ts.Context:
    return ts.Context(spec=ts.Context.Spec(spec))

  def http_store(self, data_type: str) -> Multiton[ts.TensorStore]:
    """Create an http kvstore with a path of the form
    ``1234567890_sdp_l0/correlator_data/``"""
    chunk_info = self._data_products.instance.telstate["chunk_info"]
    prefix = chunk_info[data_type]["prefix"]
    path = f"{prefix}/{data_type}/"
    return Multiton(http_store_factory, self._endpoint, path, self._token, None)

  def http_backed_store(self, data_type: str) -> Multiton[ts.TensorStore]:
    """Create a virtual chunked tensorstore backed by an http kvstore
    that retrieves data from the MeerKAT archive"""
    return Multiton(
      base_virtual_store,
      self.http_store(data_type),
      self._data_products.instance.telstate["chunk_info"][data_type],
      DATA_TYPE_LABELS[data_type],
      MISSING_VALUES[data_type],
      self.get_context({"cache_pool": {"total_bytes_limit": CACHE_SIZE}}),
    )

  def base_visibility_store(
    self,
    data_type: str = "correlator_data",
  ) -> Multiton[ts.TensorStore]:
    """Creates a base visibility store which may apply a Van Vleck correction."""
    return Multiton(
      base_visibility_virtual_store,
      self.http_store(data_type),
      self._data_products.instance.telstate["chunk_info"][data_type],
      DATA_TYPE_LABELS[data_type],
      MISSING_VALUES[data_type],
      self._autocorrs,
      self._van_vleck,
      self.get_context({"cache_pool": {"total_bytes_limit": CACHE_SIZE}}),
    )

  def final_weight_store(
    self,
    vis_store: Multiton[ts.TensorStore],
    int_weights_store: Multiton[ts.TensorStore],
    channel_weights_store: Multiton[ts.TensorStore],
  ) -> Multiton[ts.TensorStore]:
    telstate = self._data_products.instance.telstate

    return Multiton(
      scaled_weight_store,
      int_weights_store,
      channel_weights_store,
      vis_store,
      self._autocorrs,
      telstate["chunk_info"],
      telstate.get("needs_weight_power_scale", False),
      DATA_TYPE_LABELS["correlator_data"],
      self.get_context({"data_copy_concurrency": {"limit": 12}}),
    )

  def create(self):
    if self._vis is None:
      self._vis = self.base_visibility_store()

    if self._weight is None:
      base_int_weights = self.http_backed_store("weights")
      base_chan_weights = self.http_backed_store("weights_channel")
      self._weight = self.final_weight_store(
        self._vis,
        base_int_weights,
        base_chan_weights
      )

    if self._flag is None:
      self._flag = self.http_backed_store("flags")

  def vis(self):
    if self._vis is None:
      self.create()

    return self._vis

  def weight(self):
    if self._weight is None:
      self.create()

    return self._weight

  def flag(self):
    if self._flag is None:
      self.create()

    return self._flag
