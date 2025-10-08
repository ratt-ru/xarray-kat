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


class BaseKatArray(BackendArray):
  """Adds required ``shape`` and ``dtype`` members"""

  __slots__ = ("shape", "dtype")

  shape: Tuple[int, ...]
  dtype: npt.DTypeLike

  def __init__(self, shape: Tuple[int, ...], dtype: npt.DTypeLike):
    self.shape = shape
    self.dtype = dtype

  def __getitem__(self, key) -> npt.NDArray:
    raise NotImplementedError


class ChunkedKatArray(BaseKatArray):
  """Adds chunk handling behaviour to BaseKatArray"""

  __slots__ = ("chunks", "chunk_offsets")

  chunks: Tuple[Tuple[int, ...]]
  chunk_offsets: Tuple[npt.NDArray[np.int32], ...]

  def __init__(self, chunks: Tuple[Tuple[int, ...]], dtype: npt.DTypeLike):
    super().__init__(tuple(sum(c) for c in chunks), dtype)
    self.chunks = chunks
    self.chunk_offsets = tuple(np.cumsum((0,) + c) for c in chunks)


class VisibilityArray(ChunkedKatArray):
  def __init__(self, chunks: Tuple[Tuple[int, ...]], dtype: npt.DTypeLike):
    super().__init__(chunks, dtype)

  def __getitem__(self, key) -> npt.NDArray:
    return explicit_indexing_adapter(
      key, self.shape, IndexingSupport.OUTER, self._getitem
    )

  def _getitem(self, key) -> npt.NDArray:
    selected_shape = tuple(slice_length(k, s) for k, s in zip(key, self.shape))
    return np.zeros(selected_shape, self.dtype)


class TensorstoreArray(BackendArray):
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
