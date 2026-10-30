import os
import queue
import warnings
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise, product
from typing import Any, Tuple

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


#: Overrides tensorstore's default HTTP request concurrency
HTTP_REQUEST_CONCURRENCY_ENV = "TENSORSTORE_HTTP_REQUEST_CONCURRENCY"
#: tensorstore's documented default HTTP request concurrency
DEFAULT_HTTP_REQUEST_CONCURRENCY = 32
#: Chunk reads in flight per HTTP request in flight: each chunk reads
#: visibilities, weights and flags, which share their base chunks
READS_PER_REQUEST = 3


def default_max_in_flight() -> int:
  """Enough chunk reads in flight to keep tensorstore's HTTP request
  concurrency busy."""
  try:
    concurrency = int(
      os.environ.get(HTTP_REQUEST_CONCURRENCY_ENV, DEFAULT_HTTP_REQUEST_CONCURRENCY)
    )
  except ValueError:
    concurrency = DEFAULT_HTTP_REQUEST_CONCURRENCY
  return READS_PER_REQUEST * max(concurrency, 1)


class Pipeline:
  """Work items whose reads are in flight, executed as their reads complete
  rather than in the order they were issued, so that a slow read does not
  hold back the refill"""

  def __init__(self, max_in_flight: int):
    self.max_in_flight = max(max_in_flight, 1)
    self.in_flight = 0
    self.done: queue.SimpleQueue[ReadWriteWorkItem] = queue.SimpleQueue()

  def add(self, item: ReadWriteWorkItem) -> None:
    if isinstance(item.source, ts.Future):
      self.in_flight += 1
      item.source.add_done_callback(lambda _, item=item: self.done.put(item))
    else:
      item.execute()

  def wait_below(self, n: int) -> None:
    """Execute completed work items until fewer than ``n`` are in flight"""
    while self.in_flight >= n:
      self.done.get().execute()
      self.in_flight -= 1

  def make_room(self) -> None:
    self.wait_below(self.max_in_flight)

  def drain(self) -> None:
    self.wait_below(1)


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

  def compute(
    self, *data: MeerkatArray, max_in_flight: int | None = None, **kwargs
  ) -> tuple[np.ndarray, ...]:
    """Overrides the ``ChunkManager.compute`` method to reify chunked arrays.

    This method attempts to load array data chunk-wise for arrays
    with the same chunking schema. This is because final
    visibilities, flags and weights (VFW) are calculated dependent on one
    another's raw inputs. Therefore, an optimal access pattern loads
    their chunks one after the other.

    Args:
      max_in_flight: The most chunk reads in flight at once, passed as
        ``ds.load(max_in_flight=...)``. Defaults to three per HTTP request
        that tensorstore allows in flight
        (``TENSORSTORE_HTTP_REQUEST_CONCURRENCY``, 32 by default).

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
    pipeline = Pipeline(
      default_max_in_flight() if max_in_flight is None else max_in_flight
    )

    # Iterate over groups of arrays with the same chunking
    for chunks, index_and_array in same_chunks.items():
      if len(index_and_array) == 1:
        pipeline.make_room()
        # If the group contains a single array, we choose not
        # to load data chunk-wise
        index, array = index_and_array[0]
        pipeline.add(ReadWriteWorkItem(results[index][...], array.compute()[0], None))
      else:
        # Load data chunk-wise over arrays in a group,
        ranges = [np.concatenate(([0], np.cumsum(c))) for c in chunks]
        indices, arrays = zip(*index_and_array)

        for dim_coord_pairs in product(*(pairwise(r) for r in ranges)):
          # Outside the batch: its reads are only issued once it closes
          pipeline.make_room()

          with ts.Batch() as b:
            key = tuple(slice(s, e) for s, e in dim_coord_pairs)
            for index, array in zip(indices, arrays):
              pipeline.add(
                ReadWriteWorkItem(results[index][key], array[key].compute()[0], b)
              )

    pipeline.drain()

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
