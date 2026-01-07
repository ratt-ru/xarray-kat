import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from xarray.core.types import T_Chunks, T_DuckArray, T_NormalizedChunks
from xarray.namedarray._typing import _Chunks
from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint

from xarray_kat.utils import normalize_chunks


class MeerkatArray:
  __slots__ = ("chunks", "data")

  data: npt.ArrayLike
  chunks: T_NormalizedChunks

  def __init__(self, data: npt.ArrayLike, chunks: T_NormalizedChunks):
    self.data = data
    self.chunks = normalize_chunks(chunks, data.shape)

  @property
  def dtype(self) -> npt.DTypeLike:
    return self.data.dtype

  @property
  def ndim(self) -> int:
    return self.data.ndim

  @property
  def shape(self) -> tuple[int, ...]:
    return self.data.shape

  def rechunk(self, chunks, **kwargs):
    return MeerkatArray(self.data, chunks)

  def __array_namespace__(self, *, api_version: str | None = None):
    raise NotImplementedError

  def compute(
    self, *data: Any, **kwargs: Any
  ) -> tuple[np.ndarray[Any, npt.DTypeLike], ...]:
    return (self.data,)

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}<chunksize={self.chunks}, dtype={self.dtype}>"


class MeerKatChunkManager(ChunkManagerEntrypoint):
  def __init__(self):
    self.array_cls = MeerkatArray

  def is_chunked_array(self, data) -> bool:
    return isinstance(data, MeerkatArray)

  def chunks(self, data: MeerkatArray) -> T_NormalizedChunks:
    return data.chunks

  def normalize_chunks(
    self,
    chunks: T_Chunks | T_NormalizedChunks,
    shape: tuple[int, ...] | None = None,
    limit: int | None = None,
    dtype: np.dtype | None = None,
    previous_chunks: T_NormalizedChunks | None = None,
  ) -> T_NormalizedChunks:
    if shape is None:
      raise ValueError("shape was None")

    if limit is not None:
      warnings.warn(f"limit {limit} ignored in normalize_chunks", UserWarning)

    if previous_chunks is not None:
      warnings.warn(
        f"previous_chunks {previous_chunks} ignored in normalize_chunks", UserWarning
      )

    return normalize_chunks(chunks, shape)

  def from_array(
    self, data: T_DuckArray | npt.ArrayLike, chunks: _Chunks, **kw
  ) -> MeerkatArray:
    return MeerkatArray(data, chunks)

  def rechunk(self, data: MeerkatArray, chunks, **kwargs) -> MeerkatArray:
    return data.rechunk(chunks, **kwargs)

  def compute(self, *data: MeerkatArray, **kwargs) -> tuple[np.ndarray, ...]:
    return sum([d.compute(**kwargs) for d in data], ())

  def apply_gufunc(
    self,
    func,
    signature,
    *args,
    axes=None,
    axis=None,
    keepdims=False,
    output_dtypes=None,
    output_sizes=None,
    vectorize=None,
    allow_rechunk=False,
    meta=None,
    **kwargs,
  ):
    raise NotImplementedError
