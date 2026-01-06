from typing import Any

import numpy as np
import numpy.typing as npt
from xarray.core.types import T_Chunks, T_DuckArray, T_NormalizedChunks
from xarray.namedarray._typing import _Chunks
from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint


class MeerKatChunkedArray:
  __slots__ = ("chunks", "data")

  data: npt.ArrayLike
  chunks: T_NormalizedChunks

  def __init__(self, data: npt.ArrayLike, chunks: T_NormalizedChunks):
    if data.shape != (cs := tuple(sum(c) for c in chunks)):
      raise ValueError(f"data shape {data.shape} does match chunk shape {cs}")

    self.data = data
    self.chunks = chunks

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
    return MeerKatChunkedArray(self.data, chunks)

  def __array_namespace__(self, *, api_version: str | None = None):
    raise NotImplementedError

  def compute(
    self, *data: Any, **kwargs: Any
  ) -> tuple[np.ndarray[Any, npt.DTypeLike], ...]:
    return (self.data,)


class MeerKatChunkManager(ChunkManagerEntrypoint):
  def __init__(self):
    self.array_cls = MeerKatChunkedArray

  def is_chunked_array(self, data) -> bool:
    return isinstance(data, MeerKatChunkedArray)

  def chunks(self, data: MeerKatChunkedArray) -> T_NormalizedChunks:
    return data.chunks

  def normalize_chunks(
    self,
    chunks: T_Chunks | T_NormalizedChunks,
    shape: tuple[int, ...] | None = None,
    limit: int | None = None,
    dtype: np.dtype | None = None,
    previous_chunks: T_NormalizedChunks | None = None,
  ) -> T_NormalizedChunks:
    raise NotImplementedError

  def from_array(
    self, data: T_DuckArray | npt.ArrayLike, chunks: _Chunks, **kw
  ) -> MeerKatChunkedArray:
    raise MeerKatChunkedArray(data, chunks)

  def rechunk(self, data: MeerKatChunkedArray, chunks, **kwargs) -> MeerKatChunkedArray:
    return data.rechunk(chunks, **kwargs)

  def compute(self, *data: MeerKatChunkedArray, **kwargs) -> tuple[np.ndarray, ...]:  # type: ignore[override]
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
