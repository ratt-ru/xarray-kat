from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Integral
from typing import TYPE_CHECKING, Tuple

import numpy as np
import numpy.typing as npt
import tensorstore as ts
from xarray.backends import BackendArray
from xarray.core.indexing import (
  ExplicitlyIndexedNDArrayMixin,
  IndexingSupport,
  OuterIndexer,
  VectorizedIndexer,
  explicit_indexing_adapter,
)

if TYPE_CHECKING:
  from xarray_kat.multiton import Multiton


class AbstractMeerkatArchiveArray(ABC, BackendArray):
  """Require subclasses to implement ``dims`` and ``chunks`` properties.
  Note that xarray's internal API expects ``BackendArray``
  to provide ``shape`` and ``dtype`` attributes."""

  @property
  @abstractmethod
  def dims(self) -> Tuple[str, ...]:
    raise NotImplementedError

  @property
  @abstractmethod
  def chunks(self) -> Tuple[int, ...]:
    raise NotImplementedError


class CorrProductMixin:
  """Mixin containing methods for reasoning about
  ``(time, frequency, corrprod)`` shaped MeerKAT archive data.

  Implements ``dims``, ``chunks`` and ``shape`` properties
  of ``AbstractMeerkatArchiveArray``.

  The ``meerkat_key`` method produces an index
  that, when applied to a ``(time, frequency, corrprod)`` array
  produces a ``(time, frequency, baseline_id, polarization)`` array.
  This can then be transposed into canonical MSv4 ording.
  """

  __slots__ = ("_cp_argsort", "_msv4_shape", "_msv4_dims", "_msv4_chunks")

  _cp_argsort: npt.NDArray
  _msv4_shape: Tuple[int, int, int, int]
  _msv4_dims: Tuple[str, str, str, str]
  _msv4_chunks: Tuple[int, int, int, int]

  def __init__(
    self,
    meerkat_shape: Tuple[int, int, int],
    meerkat_dims: Tuple[str, str, str],
    meerkat_chunks: Tuple[int, int, int],
    cp_argsort: npt.NDArray,
    npol: int,
  ):
    """Constructs a CorrProductMixin

    Args:
      meerkat_shape: The shape of the meerkat array.
        Should be associated with the ``(time, frequency, corrprod)`` dimensions.
      meerkat_dims: The dimensions of the meerkat array.
        Should be ``(time, frequency, corrprod)``.
      meerkat_chunks: The chunking of the meerkat array.
        Should be associated with the ``(time, frequency, corrprod)`` dimensions.
      cp_argsort: An array sorting the ``corrprod`` dimension into a
        canonical ``(baseline_id, polarization)`` ordering.
      npol: Number of polarizations.
    """
    if meerkat_dims != ("time", "frequency", "corrprod"):
      raise ValueError(f"{meerkat_dims} should be (time, frequency, corrprod)")

    try:
      ntime, nfreq, ncorrprod = meerkat_shape
    except ValueError:
      raise ValueError(f"{meerkat_shape} should be (time, frequency, corrprod)")
    if len(cp_argsort) != ncorrprod:
      raise ValueError(f"{len(cp_argsort)} does not match corrprods {ncorrprod}")
    self._cp_argsort = cp_argsort
    nbl, rem = divmod(len(cp_argsort), npol)
    self._msv4_shape = (ntime, nbl, nfreq, npol)
    self._msv4_dims = (meerkat_dims[0], "baseline_id", meerkat_dims[1], "polarization")
    self._msv4_chunks = (meerkat_chunks[0], nbl, meerkat_chunks[1], npol)
    if rem != 0:
      raise ValueError(
        f"Number of polarizations {npol} must divide "
        f"the correlation product index {len(cp_argsort)} exactly."
      )

  @property
  def dims(self) -> Tuple[str, ...]:
    return self._msv4_dims

  @property
  def chunks(self) -> Tuple[int, ...]:
    return self._msv4_chunks

  @property
  def shape(self) -> Tuple[int, ...]:
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

  @property
  def transpose_axes(self) -> Tuple[int, int, int, int]:
    """Transpose (time, frequency, baseline_id, polarization) to
    (time, baseline_id, frequency, polarization)"""
    return (0, 2, 1, 3)



class DelayedTensorStore(ExplicitlyIndexedNDArrayMixin):
  """A wrapper for TensorStores that only produces new
  DelayedTensorStores when indexed"""
  __slots__ = ("array",)

  array: ts.TensorStore

  def __init__(self, array):
    self.array = array

  @property
  def dtype(self) -> npt.DTypeLike:
    return self.array.dtype.numpy_dtype

  def get_duck_array(self):
    return self.array

  async def async_get_duck_array(self):
    return self.array

  def _oindex_get(self, indexer: OuterIndexer):
    return DelayedTensorStore(self.array.oindex[indexer.tuple])

  def _vindex_get(self, indexer: VectorizedIndexer):
    return DelayedTensorStore(self.array.vindex[indexer.tuple])

  def __getitem__(self, key):
    return DelayedTensorStore(self.array[key.tuple])


class DelayedTensorStoreBackendArray(DelayedTensorStore, BackendArray):
  def __init__(self, array):
    super().__init__(array)


class ImmediateTensorStore(ExplicitlyIndexedNDArrayMixin):
  __slots__ = ("array",)

  array: ts.TensorStore

  def __init__(self, array):
    self.array = array

  @property
  def dtype(self) -> npt.DTypeLike:
    return self.array.dtype.numpy_dtype

  def get_duck_array(self):
    return self.array.read().result()

  async def async_get_duck_array(self):
    return self.array.read().result()

  def _oindex_get(self, indexer):
    return self.array.oindex[indexer.tuple].read().result()

  def _vindex_get(self, indexer):
    return self.array.vindex[indexer.tuple].read().result()

  def __getitem__(self, key):
    return self.array[key.tuple].read().result()


class ImmediateTensorBackendArray(ImmediateTensorStore, BackendArray):
  def __init__(self, array):
    super().__init__(array)



class DelayedCorrProductArray(CorrProductMixin, AbstractMeerkatArchiveArray):
  """Wraps a ``(time, frequency, corrprod)``` TensorStore.

  Most data in the MeerKAT archive has dimension
  ``(time, frequency, corrprod)``.
  This class reshapes the underlying data into the
  ``(time, baseline_id, frequency, polarization)`` form.
  """

  __slots__ = "_store"

  _store: Multiton[ts.TensorStore]

  def __init__(
    self, store: Multiton[ts.TensorStore], cp_argsort: npt.NDArray, npol: int
  ):
    CorrProductMixin.__init__(
      self,
      store.instance.shape,
      store.instance.domain.labels,
      store.instance.chunk_layout.read_chunk.shape,
      cp_argsort,
      npol,
    )
    self._store = store

  @property
  def dtype(self) -> npt.DTypeLike:
    return self._store.instance.dtype.numpy_dtype

  def __getitem__(self, key) -> DelayedTensorStore:
    return explicit_indexing_adapter(
      key, self.shape, IndexingSupport.OUTER, self._getitem
    )

  def _getitem(self, key) -> DelayedTensorStore:
    return DelayedTensorStore(
      self._store.instance[self.meerkat_key(key)]
      .transpose(self.transpose_axes)
    )

class ImmediateCorrProductArray(DelayedCorrProductArray):
  def __init__(
    self, store: Multiton[ts.TensorStore], cp_argsort: npt.NDArray, npol: int
  ):
    super().__init__(store, cp_argsort, npol)

  def _getitem(self, key) -> npt.NDArray:
    return super()._getitem(key).get_duck_array().read().result()