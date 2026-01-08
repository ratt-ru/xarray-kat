from typing import Any, List, Tuple

import numpy as np

# Type aliases based on the description
# A chunk spec can be:
# - int (positive: selected size, negative: unselected size)
# - tuple of ints (sub-chunks)
ChunkSpec = int | Tuple[int, ...]
Chunks = Tuple[ChunkSpec, ...]


def _get_partition_sizes(chunks: Chunks) -> List[int]:
  """
  Calculates the size of the original partitions (top-level chunks)
  from a chunk specification.
  """
  sizes = []
  for chunk in chunks:
    if isinstance(chunk, int):
      sizes.append(abs(chunk))
    elif isinstance(chunk, tuple):
      # Sum of absolute values of the sub-chunks
      sizes.append(sum(abs(i) for i in chunk))
    else:
      raise TypeError(f"Invalid chunk spec: {chunk}")
  return sizes


def _chunk_size_sum(chunks: Chunks) -> int:
  """Return the sum of the absolute chunk sizes"""
  size = 0

  for chunk in chunks:
    if isinstance(chunk, int):
      size += abs(chunk)
    elif isinstance(chunk, tuple):
      size += sum(abs(c) for c in chunk)
    else:
      raise TypeError(f"Invalid chunk type {type(chunk)}")

  return size


def _chunks_to_mask(chunks: Chunks) -> np.ndarray:
  """
  Expands the chunk specification into a boolean mask representing
  which indices of the ORIGINAL array are currently selected.
  """
  mask_parts = np.empty(_chunk_size_sum(chunks), dtype=bool)
  current = 0

  for chunk in chunks:
    if isinstance(chunk, int):
      mask_parts[current : (current := current + abs(chunk))] = chunk > 0
    elif isinstance(chunk, tuple):
      for c in chunk:
        mask_parts[current : (current := current + abs(c))] = c > 0

  return mask_parts


def _mask_to_chunks(mask: np.ndarray, partition_sizes: List[int]) -> Chunks:
  """
  Encodes a boolean mask back into the chunk specification,
  respecting the original partition sizes.
  """
  chunks: List[int | Tuple[int, ...]] = []
  current = 0

  for size in partition_sizes:
    # Extract the segment of the mask for this partition
    segment = mask[current : (current := current + size)]

    # Optimize: Check if all True or all False
    if np.all(segment):
      chunks.append(size)
    elif not np.any(segment):
      # This also handles the empty (0 case)
      chunks.append(-size)
    else:
      value = segment[0]
      length = 1
      sub_chunks: List[int] = []

      for v in segment[1:]:
        if value == v:
          length += 1
        else:
          sub_chunks.append(length if value else -length)
          value = v
          length = 1

      sub_chunks.append(length if value else -length)
      chunks.append(tuple(sub_chunks))

  return tuple(chunks)


def slice_chunks1d(chunks: Chunks, key: Any) -> Chunks:
  """
  Slices a tuple of chunks for a single dimension.

  Args:
      chunks: The current chunk specification.
      key: The slicing key (int, slice, list, np.ndarray, etc.)

  Returns:
      The new chunk specification.
  """
  # 1. Recover structure
  partition_sizes = _get_partition_sizes(chunks)

  # 2. Decode to mask
  full_mask = _chunks_to_mask(chunks)

  # 3. Identify currently selected indices (logical indices)
  # where full_mask is True
  (selected_indices,) = np.nonzero(full_mask)

  # 4. Apply the key to the logical indices
  # This determines which of the CURRENTLY selected items remain selected.

  # Handle scalar integer key specially to preserve dimensionality if needed?
  # But usually slice_chunks is called per dimension.
  # If key is int, dimensionality is reduced in numpy.
  # But here we are returning a chunk spec for the underlying data.
  # If the dimension is dropped, xarray might handle it.
  # For this function, we just compute the new selection mask.

  try:
    # We apply the key to the array of selected indices.
    # This gives us the subset of physical indices that are now selected.
    new_subset_indices = selected_indices[key]
  except IndexError:
    raise IndexError(
      f"Index {key} out of bounds for chunked "
      f"dimension of length {len(selected_indices)}"
    )

  # 5. Create new mask
  new_mask = np.zeros_like(full_mask)

  # If the key was a single integer, new_subset_indices is a scalar (physical index).
  # If slice or array, it's an array of physical indices.
  if np.isscalar(new_subset_indices):
    new_mask[new_subset_indices] = True
  else:
    new_mask[new_subset_indices] = True

  # 6. Re-encode
  return _mask_to_chunks(new_mask, partition_sizes)


def slice_chunks(chunks: Tuple[Chunks, ...], key: Any) -> Tuple[Chunks, ...]:
  """
  Slices a tuple of chunk tuples (multi-dimensional).

  Args:
      chunks: Tuple of chunk specs, one per dimension.
      key: Slicing key. Can be a tuple for multi-dim slicing.

  Returns:
      Tuple of new chunk specs.
  """
  # Normalize key to tuple
  if not isinstance(key, tuple):
    key = (key,)

  # Handle Ellipsis and normalize key length
  # This is a simplified logic similar to xarray's or numpy's expansion
  # It assumes the key matches the dimensions.

  # Expand Ellipsis
  if Ellipsis in key:
    # Find index of Ellipsis
    e_idx = key.index(Ellipsis)
    # Calculate how many slices needed
    missing = len(chunks) - (len(key) - 1)
    new_key = key[:e_idx] + (slice(None),) * missing + key[e_idx + 1 :]
    key = new_key

  # Pad with slice(None) if key is shorter than chunks
  if len(key) < len(chunks):
    key = key + (slice(None),) * (len(chunks) - len(key))

  if len(key) != len(chunks):
    # Could happen if too many indices provided
    raise IndexError(f"Too many indices: key={key}, ndim={len(chunks)}")

  new_chunks = []
  for dim_chunks, dim_key in zip(chunks, key):
    new_chunks.append(slice_chunks1d(dim_chunks, dim_key))

  return tuple(new_chunks)
