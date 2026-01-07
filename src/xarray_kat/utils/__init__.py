from typing import Iterable, Tuple

from xarray.core.types import T_NormalizedChunks


def normalize_chunks(
  chunks: Iterable[int | Iterable[int]], shape: Iterable[int]
) -> T_NormalizedChunks:
  """
  Normalizes chunk sizes for an N-dimensional array based on a target shape.

  For each dimension:
  1. If an integer is provided, the dimension is split into uniform chunks of that
     size (the last chunk may be smaller to fit the shape).
  2. If an iterable is provided:
     - If the chunks sum to less than the shape, the last chunk value is repeated
       until the shape is covered.
     - If the chunks sum to more than the shape, the chunks are truncated at the
       point where the shape limit is reached.

  Args:
      chunks: An iterable of length L, where L is the rank of the shape.
              Each element specifies chunking for that dimension.
      shape: An iterable of integers representing the full size of each dimension.

  Returns:
      A tuple of tuples, where each inner tuple contains the explicit
      integer sizes for chunks in that dimension.
  """
  shape = tuple(shape)
  if not all(isinstance(d, int) for d in shape):
    raise TypeError(f"shape {shape} must be an Iterable[int]")

  chunks_list = list(chunks)
  if len(chunks_list) != len(shape):
    raise TypeError(
      f"chunks length {len(chunks_list)} must match shape length {len(shape)}"
    )

  normalized = []

  for c, s in zip(chunks_list, shape):
    # Case 1: Uniform integer chunk size
    if isinstance(c, int):
      if c <= 0:
        raise ValueError(f"Chunk size must be greater than 0, got {c}")

      if s == 0:
        dim_chunks = []
      else:
        n, rem = divmod(s, c)
        dim_chunks = [c] * n
        if rem > 0:
          dim_chunks.append(rem)
      normalized.append(tuple(dim_chunks))

    # Case 2: Explicit iterable of chunks
    elif isinstance(c, Iterable):
      c_list = [int(x) for x in c]
      if any(x <= 0 for x in c_list):
        raise ValueError(f"Explicit chunk sizes must be > 0: {c_list}")

      dim_chunks = []
      current_sum = 0

      if s == 0:
        normalized.append(())
        continue

      # Handle empty iterable for a positive shape (default to one big chunk)
      if not c_list:
        normalized.append((s,))
        continue

      # Iterate through provided chunks
      for chunk_val in c_list:
        if current_sum + chunk_val >= s:
          # Overfill/Exact case: Truncate and finish
          remaining = s - current_sum
          if remaining > 0:
            dim_chunks.append(remaining)
          current_sum = s
          break
        else:
          dim_chunks.append(chunk_val)
          current_sum += chunk_val

      # Underfill case: Use the last given chunk value to finish covering the shape
      if current_sum < s:
        fill_val = c_list[-1]
        while current_sum < s:
          next_chunk = min(fill_val, s - current_sum)
          dim_chunks.append(next_chunk)
          current_sum += next_chunk

      normalized.append(tuple(dim_chunks))

    else:
      raise TypeError(f"Invalid chunk type: {type(c)}. Expected int or Iterable.")

  return tuple(normalized)
