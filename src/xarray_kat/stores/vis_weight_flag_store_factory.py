from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING, Any, Dict, Iterable, cast

import tensorstore as ts

from xarray_kat.multiton import Multiton
from xarray_kat.stores.base_store import base_virtual_store
from xarray_kat.stores.calibration import calibration_solutions_store
from xarray_kat.stores.flag_store import final_flag_store
from xarray_kat.stores.http_store import http_store_factory
from xarray_kat.stores.visibility_stores import (
  base_visibility_virtual_store,
  final_visibility_virtual_store,
)
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

  def create(self):
    telstate = self._data_products.instance.telstate

    # Create the base visibility store
    base_vis = Multiton(
      base_visibility_virtual_store,
      self.http_store("correlator_data"),
      telstate["chunk_info"]["correlator_data"],
      DATA_TYPE_LABELS["correlator_data"],
      MISSING_VALUES["correlator_data"],
      self._autocorrs,
      self._van_vleck,
      self.get_context({"cache_pool": {"total_bytes_limit": CACHE_SIZE}}),
    )

    # Create the base integer and channel weights stores
    base_int_weights = self.http_backed_store("weights")
    base_chan_weights = self.http_backed_store("weights_channel")

    # Possibly create a calibration solutions store
    # if applycal is configure
    calibration_solutions: Multiton[ts.TensorStore] | None = None

    if self._data_products.instance.calibration_params is not None:
      calibration_solutions = Multiton(
        calibration_solutions_store,
        self._data_products,
        ("time", "frequency", "corrprod"),
        self.get_context({"cache_pool": {"total_bytes_limit": CACHE_SIZE}}),
      )

    # Create a top level context for performing data copies
    top_level_thread_ctx = self.get_context(
      {"data_copy_concurrency": {"limit": mp.cpu_count()}}
    )

    # Create the top level weight store
    self._weight = Multiton(
      scaled_weight_store,
      base_int_weights,
      base_chan_weights,
      base_vis,
      calibration_solutions,
      self._autocorrs,
      telstate["chunk_info"],
      telstate.get("needs_weight_power_scale", False),
      DATA_TYPE_LABELS["correlator_data"],
      top_level_thread_ctx,
    )

    # Create the top level visibility store
    self._vis = Multiton(
      final_visibility_virtual_store,
      base_vis,
      calibration_solutions,
      telstate["chunk_info"]["correlator_data"],
      DATA_TYPE_LABELS["correlator_data"],
      top_level_thread_ctx,
    )

    # Create the top level flag store
    self._flag = Multiton(
      final_flag_store,
      self.http_backed_store("flags"),
      calibration_solutions,
      top_level_thread_ctx,
    )

  def vis(self) -> Multiton[ts.TensorStore]:
    if self._vis is None:
      self.create()

    return cast(Multiton[ts.TensorStore], self._vis)

  def weight(self) -> Multiton[ts.TensorStore]:
    if self._weight is None:
      self.create()

    return cast(Multiton[ts.TensorStore], self._weight)

  def flag(self) -> Multiton[ts.TensorStore]:
    if self._flag is None:
      self.create()

    return cast(Multiton[ts.TensorStore], self._flag)
