import warnings
from collections import defaultdict
from functools import reduce
from itertools import pairwise, product
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

  def __getitem__(self, key):
    return self.data[key]

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
    shape: tuple[int, ...],
    limit: int | None = None,
    dtype: np.dtype | None = None,
    previous_chunks: T_NormalizedChunks | None = None,
  ) -> T_NormalizedChunks:
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
    """Overrides the ``ChunkManager.compute`` method to reify chunked arrays.

    This method attempts to load array data chunk-wise for arrays
    with the same chunking schema. This is because final
    visibilities, flags and weights (VFW) are calculated dependent on one
    another's raw inputs and it is therefore a better access pattern requires
    load their co-located chunks at the same time.

    Returns:
      tuple of result arrays
    """
    if not all(isinstance(d, MeerkatArray) for d in data):
      raise TypeError(f"Arrays {[type(d) for d in data]} are not {MeerkatArray}")

    if not all(data[0].chunks[:2] == d.chunks[:2] for d in data[1:]):
      raise ValueError(
        "Arrays do not share the same chunking schema "
        "in the (time, baseline_id) dimensions"
      )

    results = tuple(np.empty(a.shape, a.dtype) for a in data)

    # Group arrays by their chunking schema
    # Simplistic, but this will group the data arrays
    # i.e. Visibilities, Flags and Weights together as
    # they all have shape (time, baseline_id, frequency, polarization)
    # and are dependent on one another
    same_chunks = defaultdict(list)
    for d, array in enumerate(data):
      same_chunks[array.chunks].append((d, array))

    def cumsum(prev, value):
      return prev + [value + prev[-1]]

    # Iterate over groups of arrays with the same chunking
    for chunks, index_and_array in same_chunks.items():
      if len(index_and_array) == 1:
        # Singleton case, probably not necessary to ingest data chunk-wise
        index, array = index_and_array[0]
        results[index][...] = array[...]
      else:
        # Related array case
        ranges = [reduce(cumsum, c, [0]) for c in chunks]
        indices, arrays = zip(*index_and_array)

        # 1) For each chunk
        #   a) For each related array
        #     i) Do a read + write
        for dim_coord_pairs in product(*(pairwise(r) for r in ranges)):
          key = tuple(slice(s, e) for s, e in dim_coord_pairs)

          for index, array in zip(indices, arrays):
            results[index][key] = array[key]

    return tuple(results)

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
