import numpy as np
from numpy.testing import assert_array_equal

from xarray_kat.utils.chunk_selection import chunk_ranges


def test_chunk_ranges_1d_slice():
  # Array size 20, chunks (10, 10)
  chunks = ((10, 10),)

  # 1. Slice within first chunk
  # 0:5 -> chunk 0: 0:5
  res = chunk_ranges(chunks, (slice(0, 5),))
  assert res == ((slice(0, 5),),)

  # 2. Slice spanning chunks
  # 5:15 -> chunk 0: 5:10, chunk 1: 10:15
  # So it returns absolute ranges.

  res = chunk_ranges(chunks, (slice(5, 15),))
  assert res == ((slice(5, 10), slice(10, 15)),)

  # 3. Slice exactly one chunk
  res = chunk_ranges(chunks, (slice(10, 20),))
  assert res == ((slice(10, 20),),)


def test_chunk_ranges_1d_int():
  chunks = ((10, 10),)

  # Index 5
  res = chunk_ranges(chunks, (5,))
  # _1d_chunk_ranges returns (indexer, indexer + 1)
  # So (5, 6)
  assert res == (5,)

  # Index 15
  res = chunk_ranges(chunks, (15,))
  assert res == (15,)

  # Negative index -1 -> 19
  res = chunk_ranges(chunks, (-1,))
  assert res == (19,)


def test_chunk_ranges_1d_array():
  chunks = ((10, 10),)

  # Indices [1, 15]
  # _1d_chunk_ranges for array:
  # It returns a tuple of 2D arrays.

  idx = np.array([1, 15])
  res = chunk_ranges(chunks, (idx,))
  # Expected:
  # Chunk 0 covers index 1.
  # Chunk 1 covers index 15.
  # It loops over "contiguous runs in sorted indices".
  # Sorted: 1, 15. Diff > 1? Yes. 15-1=14.
  # Splits: [0, 1, 2]
  # Run 1: [1]. Chunk matches? 0-10.
  # src_dst for run 1: [[1], [0]] (index 1 at pos 0)
  # Run 2: [15]. Chunk matches? 10-20.
  # src_dst for run 2: [[15], [1]] (index 15 at pos 1)

  # The function returns a tuple of these 2D arrays.
  assert len(res) == 1
  chunk_res = res[0]
  assert len(chunk_res) == 2

  np.testing.assert_array_equal(chunk_res[0], [[1], [0]])
  np.testing.assert_array_equal(chunk_res[1], [[15], [1]])

  # Unsorted: [15, 1]
  idx2 = np.array([15, 1])
  res2 = chunk_ranges(chunks, (idx2,))
  # Sorted is [1, 15]. Argsort is [1, 0].
  # Run 1: index 1. Pos in result is 1.
  # Run 2: index 15. Pos in result is 0.

  assert len(res2[0]) == 2
  # Order depends on iteration over chunks.
  # The code iterates splits of sorted indices.
  # So first it handles index 1 (which is at pos 1 in original).
  # Then index 15 (which is at pos 0 in original).

  assert_array_equal(res2[0][0], [[1], [1]])
  assert_array_equal(res2[0][1], [[15], [0]])


def test_chunk_ranges_multidim():
  chunks = ((10, 10), (5, 5))
  indexer = (slice(0, 15), 2)

  res = chunk_ranges(chunks, indexer)

  # Dim 0: slice(0, 15) on chunks (10, 10) -> ((0, 10), (10, 15))
  # Dim 1: int 2 on chunks (5, 5) -> (2, 3)

  assert res == ((slice(0, 10), slice(10, 15)), 2)


def test_chunk_ranges_empty():
  chunks = ((10,),)

  # Empty slice
  res = chunk_ranges(chunks, (slice(0, 0),))
  # _slice_ranges:
  # start=0, stop=0.
  # bisect_left([0, 10], 0) -> 0
  # bisect_right([0, 10], 0) -> 0
  # boundary_points = []
  # insert 0 -> [0]
  # append 0 -> [0, 0]
  # zip -> (0, 0)
  # BUT logic:
  # if boundary_points[-1] != stop: append stop.
  # if boundary_points[0] != start: insert start.

  # If list is initially empty:
  # insert 0 -> [0].
  # last is 0. stop is 0. So no append?
  # Actually:
  # boundary_points = offsets[0:0] = []
  # if not boundary_points or boundary_points[0] != start (0!=0 False):
  # Wait, if empty, boundary_points[0] raises IndexError if checked directly?
  # Python short-circuit: if not boundary_points (True) OR ...
  # So it enters block. inserts 0.
  # boundary_points is [0].
  # boundary_points[-1] (0) != stop (0) -> False.
  # result: [0].
  # zip([0][:-1], [0][1:]) -> zip([], []) -> empty tuple.

  assert res == ((),)


def test_chunk_ranges_ndarray_contiguous():
  chunks = ((10, 10),)
  # Indices 1, 2, 3
  idx = np.array([1, 2, 3])
  res = chunk_ranges(chunks, (idx,))

  # Contiguous run 1, 2, 3.
  # One chunk (0-10) covers it.
  # One result array.
  assert len(res[0]) == 1
  assert_array_equal(res[0][0], [[1, 2, 3], [0, 1, 2]])


def test_chunk_ranges_ndarray_crossing_chunks():
  chunks = ((5, 5),)
  # Indices 3, 4, 5, 6
  # 3, 4 in chunk 0.
  # 5, 6 in chunk 1.
  idx = np.arange(3, 7)
  res = chunk_ranges(chunks, (idx,))

  # Should produce 2 arrays.
  # One for 3, 4, 5. One for 6.
  assert len(res[0]) == 2
  assert_array_equal(res[0][0], [[3, 4], [0, 1]])
  assert_array_equal(res[0][1], [[5, 6], [2, 3]])


def test_chunk_repeated_values():
  chunks = (3, 4, 5)
  res = chunk_ranges((chunks,), (np.array([3, 2, 2, 6, 5, 1]),))

  assert_array_equal(res[0][0], [[2, 2, 1], [1, 2, 5]])
  assert_array_equal(res[0][1], [[3, 6, 5], [0, 3, 4]])
