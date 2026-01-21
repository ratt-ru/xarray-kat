import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import pairwise, product
from typing import Any, Deque, Tuple

import numpy as np
import numpy.typing as npt
import tensorstore as ts
from xarray.core.indexing import (
  BasicIndexer,
  ImplicitToExplicitIndexingAdapter,
  LazilyIndexedArray,
  OuterIndexer,
  OuterIndexerType,
  expanded_indexer,
  integer_types,
)
from xarray.core.types import T_Chunks, T_DuckArray, T_NormalizedChunks
from xarray.namedarray._typing import _Chunks
from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint

from xarray_kat.utils import normalize_chunks
from xarray_kat.utils.chunk_selection import chunk_ranges


class MeerkatArray:
  __slots__ = ("array", "_unselected_chunks", "_selected_chunk_ranges")

  array: LazilyIndexedArray
  # Chunks associated with the full shape of the array
  # i.e. with no selection applied
  _unselected_chunks: T_NormalizedChunks
  _selected_chunk_ranges: Tuple[OuterIndexerType, ...] | None

  def __init__(self, array: T_DuckArray, chunks: T_NormalizedChunks):
    if isinstance(array, ImplicitToExplicitIndexingAdapter):
      array = array.array

    if not isinstance(array, LazilyIndexedArray):
      array = LazilyIndexedArray(array)

    self.array = array
    self._unselected_chunks = normalize_chunks(chunks, array.array.shape)
    self._selected_chunk_ranges = None
    assert tuple(sum(c) for c in self._unselected_chunks) == array.array.shape

  def __getitem__(self, key):
    xkey = expanded_indexer(key, self.array.ndim)
    if any(isinstance(k, np.ndarray) for k in xkey):
      array = self.array.oindex[OuterIndexer(xkey)]
    else:
      array = self.array[BasicIndexer(xkey)]

    return MeerkatArray(array, self._unselected_chunks)

  @property
  def chunk_ranges(self):
    if self._selected_chunk_ranges is None:
      self._selected_chunk_ranges = chunk_ranges(
        self._unselected_chunks, self.array.key.tuple
      )

    return self._selected_chunk_ranges

  @property
  def chunks(self) -> T_NormalizedChunks:
    assert len(cr := self.chunk_ranges) == len(self.array.key.tuple)
    shape = tuple(sum(c) for c in self._unselected_chunks)
    result = []

    for d, dim_chunk_ranges in enumerate(cr):
      # Squeeze out integer selections
      if isinstance(dim_chunk_ranges, integer_types):
        continue

      dim_chunks = []

      for c in dim_chunk_ranges:
        if isinstance(c, slice):
          dim_chunks.append(len(range(*c.indices(shape[d]))))
        elif isinstance(c, np.ndarray):
          assert c.ndim == 2 and c.shape[0] == 2
          dim_chunks.append(c.shape[1])
        else:
          raise TypeError(f"Invalid chunk type {type(c)}")

      result.append(tuple(dim_chunks))

    return tuple(result)

  @property
  def dtype(self) -> npt.DTypeLike:
    return self.array.dtype

  @property
  def ndim(self) -> int:
    return self.array.ndim

  @property
  def shape(self) -> tuple[int, ...]:
    return self.array.shape

  def rechunk(self, chunks, **kwargs):
    return MeerkatArray(self.array, chunks)

  def __array_namespace__(self, *, api_version: str | None = None):
    raise NotImplementedError

  def compute(
    self, *data: Any, **kwargs: Any
  ) -> tuple[np.ndarray[Any, npt.DTypeLike], ...]:
    return (self.array.get_duck_array(),)

  def __repr__(self) -> str:
    return (
      f"{self.__class__.__name__}<chunksize={self.chunks}, dtype={self.dtype.name}>"
    )


@dataclass
class ReadWriteWorkItem:
  """Encapsulates an assignment of data from a destination to a source,
  possibly as part of a batch"""

  dest: npt.NDArray
  source: ts.TensorStore | npt.NDArray
  batch: ts.Batch | None

  def __post_init__(self):
    # Issue any (batched) futures
    if isinstance(self.source, ts.TensorStore):
      self.source = self.source.read(batch=self.batch)
    elif not isinstance(self.source, np.ndarray):
      warnings.warn(f"Unexpected array type {type(self.source)}", UserWarning)

  def execute(self):
    self.dest[:] = (
      self.source.result() if isinstance(self.source, ts.Future) else self.source
    )


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

    if previous_chunks is not None and previous_chunks != chunks:
      warnings.warn(
        f"previous_chunks {previous_chunks} ignored in normalize_chunks({chunks})",
        UserWarning,
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
    another's raw inputs. Therefore, an optimal access pattern loads
    their chunks one after the other.

    Returns:
      tuple of result arrays
    """
    if not all(isinstance(d, MeerkatArray) for d in data):
      raise TypeError(f"Arrays {[type(d) for d in data]} are not {MeerkatArray}")

    results = tuple(np.empty(a.shape, a.dtype) for a in data)

    # Group arrays by their chunking schema
    # Simplistic, but this will group the data arrays
    # i.e. Visibilities, Flags and Weights together as
    # they all have shape (time, baseline_id, frequency, polarization)
    # and are dependent on one another
    same_chunks = defaultdict(list)
    for index, array in enumerate(data):
      same_chunks[array.chunks].append((index, array))

    # Pipeline retrieval of data
    pipeline: Deque[ReadWriteWorkItem] = deque()
    MAX_IN_FLIGHT = 10

    # Iterate over groups of arrays with the same chunking
    for chunks, index_and_array in same_chunks.items():
      while len(pipeline) >= MAX_IN_FLIGHT:
        pipeline.popleft().execute()

      if len(index_and_array) == 1:
        # If the group contains a single array, we choose not
        # to load data chunk-wise
        index, array = index_and_array[0]
        pipeline.append(
          ReadWriteWorkItem(results[index][...], array.compute()[0], None)
        )
      else:
        # Load data chunk-wise over arrays in a group,
        ranges = [np.concatenate(([0], np.cumsum(c))) for c in chunks]
        indices, arrays = zip(*index_and_array)

        for dim_coord_pairs in product(*(pairwise(r) for r in ranges)):
          with ts.Batch() as b:
            key = tuple(slice(s, e) for s, e in dim_coord_pairs)
            for index, array in zip(indices, arrays):
              pipeline.append(
                ReadWriteWorkItem(results[index][key], array[key].compute()[0], b)
              )

    while len(pipeline) > 0:
      pipeline.popleft().execute()

    return results

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
