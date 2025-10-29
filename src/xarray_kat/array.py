from numbers import Integral
from typing import Tuple

import numpy as np
import numpy.typing as npt
import tensorstore as ts
from xarray.backends import BackendArray
from xarray.core.indexing import (
  IndexingSupport,
  explicit_indexing_adapter,
)


def slice_length(s: npt.NDArray | slice, max_len) -> int:
  if isinstance(s, np.ndarray):
    if s.ndim != 1:
      raise NotImplementedError("Slicing with non-1D numpy arrays")
    return len(s)

  start, stop, step = s.indices(min(max_len, s.stop) if s.stop is not None else max_len)
  if step != 1:
    raise NotImplementedError(f"Slicing with steps {s} other than 1 not supported")
  return stop - start


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
  _cp_argsort: npt.NDArray
  _nbl: int
  _npol: int

  def __init__(self, cp_argsort: npt.NDArray, npol: int):
    self._cp_argsort = cp_argsort
    self._nbl, rem = divmod(len(cp_argsort), npol)
    assert rem == 0, f"Number of polarizations {npol} must divided {len(cp_argsort)} exactly"
    self._npol = npol

  @property
  def nbl(self):
    return self._nbl

  @property
  def npol(self):
    return self._npol

  def msv4_shape(self, shape: Tuple[int, int, int]):
    ntime, nfreq, corrprod = shape
    assert len(self._cp_argsort) == corrprod
    return (ntime, self.nbl, nfreq, self.npol)

  def _normalize_key_axis(
    self, key: Tuple[slice | npt.NDarray | Integral, ...], shape: Tuple[int, ...], axis: int
  ) -> npt.NDArray:
    key_item = key[axis]
    if isinstance(key_item, slice):
      return np.arange(shape[axis])[key_item]
    elif isinstance(key_item, Integral):
      return np.array([key_item])
    elif isinstance(key_item, np.ndarray):
      return key_item
    else:
      raise TypeError(f"Invalid key_item type {type(key_item)}")

  def _corrprod_key(self, key: Tuple, shape: Tuple[int, int, int, int]) -> Tuple:
    assert isinstance(key, tuple) and len(key) == 4
    time_selection = key[0]
    baseline_selection = self._normalize_key_axis(key, shape, 1)
    frequency_selection = key[2]
    polarization_selection = self._normalize_key_axis(key, shape, 3)

    freq_grid, pol_grid = np.meshgrid(
      baseline_selection, polarization_selection, indexing="ij"
    )
    cp_index = self._cp_argsort[freq_grid * shape[3] + pol_grid]
    return (time_selection, frequency_selection, cp_index)


class CorrProductTensorstore(BackendArray, CorrProductMixin):
  """Wraps a ``(time, frequency, corrprod)``` array.

  Most arrays in the MeerKAT archive have the form ``(time, frequency, corrprod)``.
  This reshapes the data into the ``(time, baseline_id, frequency, polarization)`` form
  """
  def __init__(self, store: ts.TensorStore, cp_argsort: npt.NDArray, npol: int):
    self._store = store
    CorrProductMixin.__init__(self, cp_argsort, npol)


class CorrProductTensorstore2(BackendArray):
  """Wraps a ``(time, frequency, corrprod)``` array.

  Most arrays in the MeerKAT archive have the form ``(time, frequency, corrprod)``.
  This reshapes the data into the ``(time, baseline_id, frequency, polarization)`` form
  """

  __slots__ = ("_store", "_cp_argsort", "_shape")

  _store: ts.TensorStore
  _cp_argsort: npt.NDArray
  _shape: Tuple[int, ...]

  def __init__(self, store: ts.TensorStore, cp_argsort: npt.NDArray, npol: int):
    self._store = store
    self._cp_argsort = cp_argsort
    assert store.domain.labels == ("time", "frequency", "corrprod")
    ntime, nfreq, ncorrprod = store.shape
    assert len(cp_argsort) == ncorrprod
    nbl, rem = divmod(len(cp_argsort), npol)
    assert rem == 0, "Polarization must divide correlation products exactly"
    self._shape = (ntime, nbl, nfreq, npol)

  @property
  def shape(self) -> Tuple[int, ...]:
    return self._shape

  @property
  def dtype(self) -> npt.DTypeLike:
    return np.dtype(self._store.dtype.type)

  def _normalize_key_axis(self, key, axis):
    key_item = key[axis]
    if isinstance(key_item, slice):
      return np.arange(self.shape[axis])[key_item]
    elif isinstance(key_item, Integral):
      return np.array([key_item])
    elif isinstance(key_item, np.ndarray):
      return key_item
    else:
      raise TypeError(f"Invalid key_item type {type(key_item)}")

  def __getitem__(self, key) -> npt.NDArray:
    return explicit_indexing_adapter(
      key, self.shape, IndexingSupport.OUTER, self._getitem
    )

  def _getitem(self, key) -> npt.NDArray:
    assert isinstance(key, tuple) and len(key) == 4
    time_selection = key[0]
    baseline_selection = self._normalize_key_axis(key, 1)
    frequency_selection = key[2]
    polarization_selection = self._normalize_key_axis(key, 3)

    freq_grid, pol_grid = np.meshgrid(
      baseline_selection, polarization_selection, indexing="ij"
    )
    nbl = len(baseline_selection)
    npol = len(polarization_selection)
    cp_index = freq_grid * self._shape[3] + pol_grid
    data = (
      self._store[(time_selection, frequency_selection, self._cp_argsort[cp_index])]
      .read()
      .result()
    )
    return data.reshape(data.shape[:2] + (nbl, npol)).transpose(0, 2, 1, 3)


class WeightTensorstore(BackendArray):
  pass
