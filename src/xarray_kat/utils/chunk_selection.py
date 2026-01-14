from itertools import chain, pairwise
from typing import Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
from xarray.core.indexing import OuterIndexer, OuterIndexerType, integer_types
from xarray.core.types import T_NormalizedChunks

# A chunk range can be
# 1. A single integer
# 2. A tuple of slices
# 3. A tuple of ndarrays of shape (2, index)
#    i.e. source and destination indices
T_ChunkRange: TypeAlias = int | np.integer | Tuple[slice | npt.NDArray, ...]
T_ChunkRanges = Tuple[T_ChunkRange, ...]

def _slice_ranges(
  offsets: npt.NDArray[np.integer],
  indexer: slice,
) -> Tuple[Tuple[int, int], ...]:
  """Find the chunk ranges that intersect with the indexer slice"""
  start, stop = indexer.start, indexer.stop

  # Find the range of indices in 'offsets' that fall within [start, stop]
  # left: first offset >= start; right: first offset > stop
  left = np.searchsorted(offsets, indexer.start, side="left")
  right = np.searchsorted(offsets, indexer.stop, side="right")

  # Extract the internal boundary points
  boundary_points = offsets[left : right]
  concat = [boundary_points]

  # Add the endpoints if they aren't already present
  if len(boundary_points) == 0 or boundary_points[0] != start:
    concat.insert(0, (start,))

  if len(boundary_points) == 0 or boundary_points[-1] != stop:
    concat.append((stop,))

  # Zip the points into pairs: (p0, p1), (p1, p2), ...
  return tuple(pairwise(map(int, chain(*concat))))


def _1d_chunk_ranges(
  chunks: Tuple[int, ...],
  size: int,
  indexer: OuterIndexerType
) -> T_ChunkRange:
  """For a give indexer, produces indexers for each chunk that it overlaps"""
  if len(chunks) == 0:
    raise ValueError(f"Indexing empty array")

  offsets = np.concatenate(([0], np.cumsum(chunks)))

  if isinstance(indexer, integer_types):
    if indexer > size:
      raise IndexError(f"Index {indexer} out of bounds for size {size}")
    return indexer if indexer > 0 else indexer + size
  elif isinstance(indexer, slice):
    # Normalize the indexer
    indexer = slice(*indexer.indices(size))
    return tuple(slice(s, e) for s, e in _slice_ranges(offsets, indexer))
  elif isinstance(indexer, np.ndarray):
    if np.any(oob := indexer >= size):
      raise IndexError(
        f"Indices {indexer[np.where(oob)[0]]} "
        f"are out of bounds for size {size}"
      )

    # Find the chunks to which each index applies
    indexer = np.where(indexer < 0, indexer + size, indexer)
    chunk_indices = np.searchsorted(offsets, indexer, side="right")
    dest = np.arange(chunk_indices.size)
    # Group indices by chunks
    chunks, inv = np.unique(chunk_indices, return_inverse=True)
    results = []

    # Create source and destination values for each chunk
    for c in range(len(chunks)):
      mask = c == inv
      chunk_index = indexer[mask]
      dest_index = dest[mask]
      dtype = np.min_scalar_type(max(chunk_index.max(), dest_index.max()))
      src_dest = np.empty((2, chunk_index.size), dtype)
      src_dest[0, ...] = chunk_index
      src_dest[1, ...] = dest_index
      results.append(src_dest)

    return tuple(results)
  else:
    raise TypeError(f"Invalid indexing type {type(indexer)}")

def chunk_ranges(
    chunks: T_NormalizedChunks, indexer: OuterIndexer
) -> T_ChunkRanges:
  """Given a chunking scheme and an indexer,
  return indexers for each individual chunk.

  1. Integers indexers resolve to themselves.
  2. Slice indexers resolve to a tuple of slices,
     each describing an intersecting range in each
     individual chunk.
  3. Array indexers resolve to a tuple of arrays
     of shape (2, index) describing source and
     destination indices for each chunk.

  Args:
    chunks: The chunking scheme.
    indexer: The indexer to apply.

  Returns:
    A tuple of chunk ranges describing how
    the selection applies to each chunk.
  """
  assert len(chunks) == len(indexer)
  shape = tuple(sum(c) for c in chunks)
  return tuple(_1d_chunk_ranges(c, s, i) for c, s, i in zip(chunks, shape, indexer))