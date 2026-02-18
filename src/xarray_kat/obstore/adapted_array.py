import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from threading import Lock
from typing import Tuple, TypedDict, cast

import numpy as np
import numpy.typing as npt
from xarray.backends.common import BackendArray
from xarray.core.indexing import (
  ExplicitIndexer,
  expanded_indexer,
  integer_types,
)

from xarray_kat.xkat_types import ArchiveArrayMetadata

# A selection over (time, frequency, corrprod)
DimRangeType = Tuple[slice, slice, slice]


class PreferredChunksType(TypedDict):
  time: int
  frequency: int
  corrprod: int


class VisFlagWeightData:
  __slots__ = ("_vis", "_weight", "_flag")

  _vis: npt.NDArray | None
  _weight: npt.NDArray | None
  _flag: npt.NDArray | None

  def __init__(self):
    self._vis = None
    self._weight = None
    self._flag = None


class VisFlagWeightGrid:
  def __init__(
    self,
    vis_meta: ArchiveArrayMetadata,
    weight_meta: ArchiveArrayMetadata,
    channel_weight_meta: ArchiveArrayMetadata,
    flag_meta: ArchiveArrayMetadata,
    preferred_chunks: PreferredChunksType,
  ):
    self._vis_meta = vis_meta
    self._weight_meta = weight_meta
    self._channel_weight_meta = channel_weight_meta
    self._flag_meta = flag_meta
    self._preferred_chunks = preferred_chunks
    self._pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())

    if not (vis_meta.shape == weight_meta.shape == flag_meta.shape) or not (
      vis_meta.shape[:2] == channel_weight_meta.shape
    ):
      raise ValueError("Archive Array shapes don't match")

    self.shape = vis_meta.shape
    ntime, nfreq, ncorrprod = self.shape

    array_chunks = [
      vis_meta.chunks,
      weight_meta.chunks,
      channel_weight_meta.chunks + (-1,),
      flag_meta.chunks,
    ]

    time_chunks, freq_chunks, cp_chunks = (max(c) for c in zip(*array_chunks))
    time_chunks = max(preferred_chunks.get("time", time_chunks), time_chunks)
    freq_chunks = max(preferred_chunks.get("frequency", freq_chunks), freq_chunks)
    cp_chunks = max(preferred_chunks.get("corrprod", cp_chunks), cp_chunks)
    self.chunks = (time_chunks, freq_chunks, cp_chunks)

    ntime_chunks, rem = divmod(ntime, time_chunks)
    ntime_chunks += int(rem != 0)
    nfreq_chunks, rem = divmod(nfreq, freq_chunks)
    nfreq_chunks += int(rem != 0)
    ncorrprod_chunks, rem = divmod(ncorrprod, cp_chunks)
    ncorrprod_chunks += int(rem != 0)

    shape = (ntime_chunks, nfreq_chunks, ncorrprod_chunks)
    nelements = ntime_chunks * nfreq_chunks * ncorrprod_chunks
    grid = np.asarray([VisFlagWeightData()] * nelements).reshape(shape)  # noqa: F841
    locks = np.asarray([Lock()] * nelements).reshape(shape)  # noqa: F841

  def _chunk_indexer(self, key):
    ndim = len(self.shape)
    indexer = (
      expanded_indexer(key, ndim) if not isinstance(key, ExplicitIndexer) else key
    )

    new_indexer = []

    for index, chunk, size in zip(indexer, self.chunks, self.shape):
      if isinstance(index, integer_types):
        new_indexer.append([index if index >= 0 else index + size])
      elif isinstance(index, slice):
        if index.step not in {None, 1}:
          raise NotImplementedError(
            f"slice steps {index.step} other than 1 are not currently supported"
          )

        if (index_start := 0 if index.start is None else index.start) < 0:
          index_start += size

        if (index_stop := size if index.stop is None else index.stop) < 0:
          index_stop += size

        start_chunk, start_rem = divmod(index_start, chunk)
        end_chunk, end_rem = divmod(index_stop, chunk)

        # The index addresses a single chunk in any case
        if start_chunk == end_chunk:
          new_indexer.append([index])
        else:
          new_index = [
            slice(index_start, index_start + (chunk if start_rem == 0 else start_rem))
          ]

          for c in range(start_chunk + 1, end_chunk):
            new_index.append(slice(c * chunk, c * chunk + chunk))

          if end_rem > 0:
            new_index.append(slice(index_stop - 1, index_stop + end_rem - 1))

          new_indexer.append(new_index)
      elif isinstance(index, np.ndarray):
        # Convert negative indices
        index = np.where(index >= 0, index, index + size)
        argsort = np.argsort(index)
        sorted_index = index[argsort]
        index_chunks = sorted_index // chunk
        splits = np.where(np.ediff1d(index_chunks, to_begin=0) != 0)[0]
        # Compute indices within each chunk
        source_indices = np.split(sorted_index - (index_chunks * chunk), splits)
        # Compute target indices for each chunk
        target_indices = np.split(np.arange(argsort.size)[argsort], splits)
        new_indexer.append(
          list(np.vstack(pair) for pair in zip(source_indices, target_indices))
        )
      else:
        raise TypeError(f"{type(index)} was not an integer, slice or numpy array")

    return new_indexer

  def __getitem__(self, key):
    for index in product(*self._chunk_indexer(key)):
      print(index)


class VFWAdapter(BackendArray):
  def __init__(self, array: str):
    self.array = array


if __name__ == "__main__":
  from xarray_kat.utils import normalize_chunks

  prefix = "12345-sdp-l0"
  ntime = 100
  nfreq = 32
  ncorrprod = (7 * 7 // 2) * 4
  chunks = normalize_chunks((2, 8, ncorrprod), (ntime, nfreq, ncorrprod))
  time_chunks, freq_chunks, _ = chunks

  chunk_info = {
    "correlator_data": {
      "prefix": prefix,
      "dtype": "<c8",  # complex64 little-endian
      "shape": (ntime, nfreq, ncorrprod),
      "chunks": (time_chunks, freq_chunks, (ncorrprod,)),
    },
    "flags": {
      "prefix": prefix,
      "dtype": "|u1",  # uint8
      "shape": (ntime, nfreq, ncorrprod),
      "chunks": (time_chunks, freq_chunks, (ncorrprod,)),
    },
    "weights": {
      "prefix": prefix,
      "dtype": "|u1",  # uint8
      "shape": (ntime, nfreq, ncorrprod),
      "chunks": (time_chunks, freq_chunks, (ncorrprod,)),
    },
    "weights_channel": {
      "prefix": prefix,
      "dtype": "<f4",  # float32 little-endian
      "shape": (ntime, nfreq),
      "chunks": (time_chunks, freq_chunks),
    },
  }

  dim_labels = ("time", "frequency", "corrprod")

  meta = {
    name: ArchiveArrayMetadata(
      name,
      0,
      dim_labels[: len(value["shape"])],
      cast(str, value["prefix"]),
      cast(Tuple[Tuple[int, ...], ...], value["chunks"]),
      cast(str, value["dtype"]),
    )
    for name, value in chunk_info.items()
  }

  grid = VisFlagWeightGrid(
    meta["correlator_data"],
    meta["weights"],
    meta["weights_channel"],
    meta["flags"],
    {"time": 2, "frequency": 8, "corrprod": 4},
  )

  print(grid[slice(5), np.array([6, 11, 7, 7, 10, 11, 8, 12])])
