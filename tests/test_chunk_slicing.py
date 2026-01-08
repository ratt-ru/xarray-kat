import numpy as np

from xarray_kat.utils.chunk_slicing import (
  _chunks_to_mask,
  _get_partition_sizes,
  _mask_to_chunks,
  slice_chunks,
  slice_chunks1d,
)


def test_partition_sizes():
  # CHUNKS.md example 1
  chunks = (5, 10, 3)
  assert _get_partition_sizes(chunks) == [5, 10, 3]

  # CHUNKS.md example 2
  # chunks = ((-1, 3, -1), (3, -2, 5), -3, 2)
  # The example text says "four top-level chunks of 5, 10, 3 and 2 values"
  # Chunk 0: (-1, 3, -1) -> 1+3+1 = 5. Correct.
  # Chunk 1: (3, -2, 5) -> 3+2+5 = 10. Correct.
  # Chunk 2: -3 -> 3. Correct.
  # Chunk 3: 2 -> 2. Correct.
  chunks_complex = ((-1, 3, -1), (3, -2, 5), -3, 2)
  assert _get_partition_sizes(chunks_complex) == [5, 10, 3, 2]


def test_mask_conversion():
  # Simple case
  chunks = (2, 2)  # [T, T, T, T]
  mask = _chunks_to_mask(chunks)
  np.testing.assert_array_equal(mask, [True, True, True, True])

  re_chunks = _mask_to_chunks(mask, [2, 2])
  assert re_chunks == (2, 2)

  # Complex case
  # (-1, 2) -> F, T, T (size 3)
  chunks = ((-1, 2),)
  mask = _chunks_to_mask(chunks)
  np.testing.assert_array_equal(mask, [False, True, True])

  re_chunks = _mask_to_chunks(mask, [3])
  assert re_chunks == ((-1, 2),)


def test_slice_chunks1d_basic():
  # Array length 10, split (5, 5)
  chunks = (5, 5)

  # Slice first 3
  # Indices 0, 1, 2 selected.
  # Chunk 0 (0-5): 0, 1, 2 selected -> 3, -2
  # Chunk 1 (5-10): none selected -> -5
  res = slice_chunks1d(chunks, slice(0, 3))
  assert res == ((3, -2), -5)

  # Slice middle
  # slice(4, 7) -> 4, 5, 6
  # Chunk 0: index 4 selected -> -4, 1
  # Chunk 1: indices 5, 6 selected -> 2, -3
  res = slice_chunks1d(chunks, slice(4, 7))
  assert res == ((-4, 1), (2, -3))


def test_slice_chunks1d_recursive():
  # Start with complex chunks
  # Chunk 0 (5): (-1, 3, -1) -> Indices 1, 2, 3 valid.
  # Chunk 1 (5): 5 -> Indices 5, 6, 7, 8, 9 valid.
  chunks = ((-1, 3, -1), 5)

  # Logical indices available: 0->1, 1->2, 2->3, 3->5, 4->6...
  # We slice logical(0, 2) -> indices 0, 1 of the view
  # Maps to physical 1, 2.
  # New Chunk 0: 1, 2 selected -> -1, 2, -2
  # New Chunk 1: none selected -> -5

  res = slice_chunks1d(chunks, slice(0, 2))
  assert res == ((-1, 2, -2), -5)


def test_slice_chunks1d_fancy():
  chunks = (5, 5)
  # Select indices 1 and 6 using integer array
  key = np.array([1, 6])

  # Chunk 0: 1 selected -> -1, 1, -3
  # Chunk 1: 6 selected (idx 1 in chunk) -> -1, 1, -3
  res = slice_chunks1d(chunks, key)
  assert res == ((-1, 1, -3), (-1, 1, -3))

  # Verify monotonic property (reordering ignored)
  # Key [6, 1] should produce same chunks
  key2 = np.array([6, 1])
  res2 = slice_chunks1d(chunks, key2)
  assert res2 == res


def test_slice_chunks1d_step():
  chunks = (10,)
  # slice(0, 10, 2) -> 0, 2, 4, 6, 8
  # Pattern: T, F, T, F...
  res = slice_chunks1d(chunks, slice(0, 10, 2))
  # Expect (1, -1, 1, -1, 1, -1, 1, -1, 1, -1)
  assert res == ((1, -1, 1, -1, 1, -1, 1, -1, 1, -1),)


def test_slice_multidim():
  # 2D array (10, 10), chunks ((5, 5), (5, 5))
  chunks = ((5, 5), (5, 5))

  # slice [0:3, 6:8]
  # Dim 0: 0, 1, 2 -> ((3, -2), -5)
  # Dim 1: 6, 7 -> (-5, (-1, 2, -2))

  res = slice_chunks(chunks, (slice(0, 3), slice(6, 8)))
  assert res[0] == ((3, -2), -5)
  assert res[1] == (-5, (-1, 2, -2))


def test_numpy_types_handling():
  # Ensure it handles numpy integers in keys
  chunks = (5,)
  key = np.int64(2)
  res = slice_chunks1d(chunks, key)
  # Select index 2: -2, 1, -2
  assert res == ((-2, 1, -2),)
