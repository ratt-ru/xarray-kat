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

  @property
  def has_data(self) -> bool:
    return self._vis is not None and self._weight is not None and self._flag is not None


class VisFlagWeightGrid:
  _vis_meta: ArchiveArrayMetadata
  _weight_meta: ArchiveArrayMetadata
  _channel_weight_meta: ArchiveArrayMetadata
  _flag_meta: ArchiveArrayMetadata
  _preferred_chunks: PreferredChunksType
  _pool: ThreadPoolExecutor
  _grid: npt.NDArray
  _locks: npt.NDArray

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
    self._grid = np.asarray([VisFlagWeightData()] * nelements).reshape(shape)  # noqa: F841
    self._locks = np.asarray([Lock()] * nelements).reshape(shape)  # noqa: F841

  def _chunk_indexer(self, key):
    """Returns an indexer of the form (chunk_id, source_indexer, target_indexer)
    for each dimension in the grid"""
    ndim = len(self.shape)
    indexer = (
      expanded_indexer(key, ndim) if not isinstance(key, ExplicitIndexer) else key
    )

    new_indexer = []

    for index, chunk, size in zip(indexer, self.chunks, self.shape):
      if isinstance(index, integer_types):
        if index < 0:
          index += size

        chunk_index, source_index = divmod(index, chunk)

        new_indexer.append([(chunk_index, source_index, index)])
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

        # The index addresses a single chunk
        if start_chunk == end_chunk:
          new_indexer.append([(start_chunk, slice(0, index_stop - index_start), index)])
        else:
          # Multiple chunks case

          # Add the start chunk
          new_index = [
            (
              start_chunk,
              slice(start_rem, chunk),
              slice(index_start, (next_chunk := start_chunk + 1) * chunk),
            )
          ]

          # Middle chunks
          for c in range(next_chunk, end_chunk):
            new_index.append((c, slice(0, chunk), slice(c * chunk, (c + 1) * chunk)))

          if end_rem > 0:
            # Final chunk
            new_index.append(
              (
                end_chunk,
                slice(0, end_rem),
                slice(end_chunk * chunk, index_stop),
              )
            )

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
          (c[0].item(), si, ti)
          for c, si, ti in zip(
            np.split(index_chunks, splits), source_indices, target_indices
          )
        )
      else:
        raise TypeError(f"{type(index)} was not an integer, slice or numpy array")

    return new_indexer

  def __getitem__(self, key):
    for index in product(*self._chunk_indexer(key)):
      chunk, source_indices, target_indices = zip(*index)
      self._maybe_retrieve_chunk(chunk, source_indices, target_indices)

  def _maybe_retrieve_chunk(self, chunk, source_index, target_index):
    def archive_array_paths(chunk_extents, meta: ArchiveArrayMetadata) -> list[str]:
      paths = []
      chunk_starts = (
        tuple(range(s, e, c)) for (s, e), c in zip(chunk_extents, meta.chunks)
      )
      for chunk_start in product(*chunk_starts):
        path_parts = "_".join(f"{c:05d}" for c in chunk_start)
        path = f"{meta.prefix}/{meta.name}/{path_parts}.npy"
        paths.append(path)
      return paths

    with self._locks[chunk]:
      if not cast(VisFlagWeightData, self._grid[chunk]).has_data:
        chunk_extents = tuple((c * s, (c + 1) * s) for c, s in zip(chunk, self.chunks))

        paths = []
        paths.extend(archive_array_paths(chunk_extents, self._flag_meta))
        paths.extend(archive_array_paths(chunk_extents, self._vis_meta))
        paths.extend(archive_array_paths(chunk_extents, self._weight_meta))
        paths.extend(archive_array_paths(chunk_extents, self._channel_weight_meta))
        print(paths)


class VFWAdapter(BackendArray):
  def __init__(self, array: str):
    self.array = array


if __name__ == "__main__":
  from xarray_kat.utils import normalize_chunks

  prefix = "12345-sdp-l0"
  ntime = 100
  nfreq = 32
  ncorrprod = (7 * 7 // 2) * 4

  chunk_info = {
    "correlator_data": {
      "prefix": prefix,
      "dtype": "<c8",  # complex64 little-endian
      "shape": (ntime, nfreq, ncorrprod),
      "chunks": normalize_chunks((1, 8, ncorrprod), (ntime, nfreq, ncorrprod)),
    },
    "flags": {
      "prefix": prefix,
      "dtype": "|u1",  # uint8
      "shape": (ntime, nfreq, ncorrprod),
      "chunks": normalize_chunks((8, 8, ncorrprod), (ntime, nfreq, ncorrprod)),
    },
    "weights": {
      "prefix": prefix,
      "dtype": "|u1",  # uint8
      "shape": (ntime, nfreq, ncorrprod),
      "chunks": normalize_chunks((8, 8, ncorrprod), (ntime, nfreq, ncorrprod)),
    },
    "weights_channel": {
      "prefix": prefix,
      "dtype": "<f4",  # float32 little-endian
      "shape": (ntime, nfreq),
      "chunks": normalize_chunks((8, 8), (ntime, nfreq)),
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

  print(grid[slice(10, 28), np.array([6, 11, 7, 7, 10, 11, 8, 12])])
