from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, Tuple

import numpy as np
import numpy.typing as npt
import tensorstore as ts
from xarray.backends import BackendArray
from xarray.core.indexing import (
  IndexingSupport,
  explicit_indexing_adapter,
)

if TYPE_CHECKING:
  from xarray_kat.meerkat_store_provider import MeerkatStoreProvider


class TensorstoreArray(BackendArray):
  """Wraps a tensorstore"""

  __slots__ = ("_store",)

  _store: ts.TensorStore

  def __init__(self, store: ts.TensorStore):
    self._store = store

  @property
  def shape(self):
    return self._store.shape

  @property
  def dtype(self):
    return np.dtype(self._store.dtype.type)

  def __getitem__(self, key) -> npt.NDArray:
    return explicit_indexing_adapter(
      key, self.shape, IndexingSupport.OUTER, self._getitem
    )

  def _getitem(self, key) -> npt.NDArray:
    return self._store[key].read().result()


class CorrProductMixin:
  """Mixin containing methods for reasoning about
  ``(time, frequency, corrprod)`` shaped MeerKAT archive data."""

  __slots__ = ("_cp_argsort", "_msv4_shape")

  _cp_argsort: npt.NDArray
  _msv4_shape: Tuple[int, int, int, int]

  def __init__(
    self, meerkat_shape: Tuple[int, int, int], cp_argsort: npt.NDArray, npol: int
  ):
    """Constructs a CorrProductMixin

    Args:
      meerkat_shape: The shape of the meerkat array.
        It should be associated with the ``(time, frequency, corrprod)`` dimensions.
      cp_argsort: An array sorting the ``corrprod`` dimension into a
        canonical ``(baseline_id, polarization)`` ordering.
      npol: Number of polarizations.
    """
    try:
      ntime, nfreq, ncorrprod = meerkat_shape
    except ValueError:
      raise ValueError(f"{meerkat_shape} should be (time, frequency, corrprod)")
    if len(cp_argsort) != ncorrprod:
      raise ValueError(f"{len(cp_argsort)} does not match corrprods {ncorrprod}")
    self._cp_argsort = cp_argsort
    nbl, rem = divmod(len(cp_argsort), npol)
    self._msv4_shape = (ntime, nbl, nfreq, npol)
    if rem != 0:
      raise ValueError(
        f"Number of polarizations {npol} must divide "
        f"the correlation product index {len(cp_argsort)} exactly."
      )

  @property
  def shape(self) -> Tuple[int, int, int, int]:
    return self._msv4_shape

  def _normalize_key_axis(
    self,
    key: Tuple[slice | npt.NDArray | Integral, ...],
    axis: int,
  ) -> npt.NDArray:
    """Normalises ``key[axis]`` into an numpy array"""
    if isinstance(key_item := key[axis], slice):
      return np.arange(self.shape[axis])[key_item]
    elif isinstance(key_item, Integral):
      return np.array([key_item])
    elif isinstance(key_item, np.ndarray):
      return key_item
    else:
      raise NotImplementedError(f"key_item type {type(key_item)}")

  def meerkat_key(self, msv4_key: Tuple) -> Tuple:
    """Translates an MSv4 key into a MeerKAT key.

    MSv4 arrays have ``(time, baseline_id, frequency, polarization)``
    dimensions. This method translates keys referencing the above
    dimensions into keys which operate on MeerKAT archive data with
    ``(time, frequency, corrprod)`` dimensions.
    """
    assert isinstance(msv4_key, tuple) and len(msv4_key) == 4
    time_selection = msv4_key[0]
    bl_selection = self._normalize_key_axis(msv4_key, 1)
    frequency_selection = msv4_key[2]
    pol_selection = self._normalize_key_axis(msv4_key, 3)

    bl_grid, pol_grid = np.meshgrid(bl_selection, pol_selection, indexing="ij")
    # cp_selection has shape (nbl, npol). When used in an index,
    # it has the effect of splitting the corrprod dimension
    # into baseline and polarization
    npol = self.shape[3]
    cp_selection = self._cp_argsort[bl_grid * npol + pol_grid]
    return (time_selection, frequency_selection, cp_selection)


class CorrProductTensorstore(BackendArray, CorrProductMixin):
  """Wraps a ``(time, frequency, corrprod)``` array.

  Most data in the MeerKAT archive has dimension
  ``(time, frequency, corrprod)``.
  This BackendArray reshapes the underlying data into the
  ``(time, baseline_id, frequency, polarization)`` form.
  """

  __slots__ = "_store_provider"

  _store_provider: MeerkatStoreProvider

  def __init__(
    self, store_provider: MeerkatStoreProvider, cp_argsort: npt.NDArray, npol: int
  ):
    CorrProductMixin.__init__(self, store_provider.store.shape, cp_argsort, npol)
    self._store_provider = store_provider

  @property
  def dtype(self) -> npt.DTypeLike:
    return np.dtype(self._store_provider.store.dtype.type)

  def __getitem__(self, key) -> npt.NDArray:
    return explicit_indexing_adapter(
      key, self.shape, IndexingSupport.OUTER, self._getitem
    )

  def _getitem(self, key) -> npt.NDArray:
    return (
      self._store_provider.store[self.meerkat_key(key)]
      .transpose((0, 2, 1, 3))
      .read()
      .result()
    )


class WeightArray(BackendArray, CorrProductMixin):
  """ Combines weights and channel_weights to present a unified
  WEIGHTS array in the xarray layer
  """
  _int_weight_prov: MeerkatStoreProvider
  _channel_weight_prov: MeerkatStoreProvider
  _cp_argsort: npt.NDArray
  _npol: int

  def __init__(
    self,
    int_weight_prov: MeerkatStoreProvider,
    channel_weight_prov: MeerkatStoreProvider,
    cp_argsort: npt.NDArray,
    npol: int,
  ):
    # Integer weights have a (time, frequency, corrprod) shape
    CorrProductMixin.__init__(self, int_weight_prov.store.shape, cp_argsort, npol)
    self._int_weight_prov = int_weight_prov
    self._channel_weight_prov = channel_weight_prov
    self._cp_argsort = cp_argsort
    self._npol = npol

  @property
  def dtype(self) -> npt.DTypeLike:
    return np.dtype(self._channel_weight_prov.store.dtype.type)

  def __getitem__(self, key) -> npt.NDArray:
    return explicit_indexing_adapter(
      key, self.shape, IndexingSupport.OUTER, self._getitem
    )

  def _getitem(self, key) -> npt.NDArray:
    corrprod_key = self.meerkat_key(key)
    chan_weight_key = (corrprod_key[0], None, corrprod_key[1], None)
    chan_weight_store = self._channel_weight_prov.store[chan_weight_key]
    int_weight_store = ts.cast(
      self._int_weight_prov.store[corrprod_key].transpose((0, 2, 1, 3)),
      chan_weight_store.dtype,
    )

    # Issue reads at the same time, then await their completion
    int_weight_fut = int_weight_store.read()
    chan_weight_fut = chan_weight_store.read()
    return int_weight_fut.result() * chan_weight_fut.result()
