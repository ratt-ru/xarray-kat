import asyncio
import io
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from threading import Lock
from typing import Any, Tuple, TypedDict, cast

import numpy as np
import numpy.typing as npt
from xarray.backends.common import BackendArray
from xarray.core.indexing import (
  ExplicitIndexer,
  expanded_indexer,
  integer_types,
)

from xarray_kat.async_loop import AsyncLoopSingleton
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


async def _fetch_component(
  store: Any,
  paths_and_starts: list[tuple[str, tuple[int, ...]]],
  meta: ArchiveArrayMetadata,
  chunk_extents: tuple[tuple[int, int], ...],
) -> npt.NDArray:
  """Download and assemble one array component from archive chunks concurrently."""

  async def _get_bytes(path: str) -> bytes:
    result = await store.get_async(path)
    return bytes(await result.bytes_async())

  raw_bytes_list = await asyncio.gather(*(_get_bytes(p) for p, _ in paths_and_starts))

  out_shape = tuple(e - s for s, e in chunk_extents)
  out = np.empty(out_shape, dtype=meta.dtype)

  for (_, chunk_start), raw_bytes in zip(paths_and_starts, raw_bytes_list):
    file_arr = np.load(io.BytesIO(raw_bytes))
    out_slices = []
    file_slices = []
    for d, (cs, (ext_s, ext_e)) in enumerate(zip(chunk_start, chunk_extents)):
      file_size = file_arr.shape[d]
      overlap_s = max(cs, ext_s)
      overlap_e = min(cs + file_size, ext_e)
      out_slices.append(slice(overlap_s - ext_s, overlap_e - ext_s))
      file_slices.append(slice(overlap_s - cs, overlap_e - cs))
    out[tuple(out_slices)] = file_arr[tuple(file_slices)]

  return out


class VisFlagWeightGrid:
  _vis_meta: ArchiveArrayMetadata
  _weight_meta: ArchiveArrayMetadata
  _channel_weight_meta: ArchiveArrayMetadata
  _flag_meta: ArchiveArrayMetadata
  _preferred_chunks: PreferredChunksType
  _pool: ThreadPoolExecutor
  _store: Any
  _grid: npt.NDArray
  _locks: npt.NDArray

  def __init__(
    self,
    vis_meta: ArchiveArrayMetadata,
    weight_meta: ArchiveArrayMetadata,
    channel_weight_meta: ArchiveArrayMetadata,
    flag_meta: ArchiveArrayMetadata,
    preferred_chunks: PreferredChunksType,
    store: Any,
  ):
    self._vis_meta = vis_meta
    self._weight_meta = weight_meta
    self._channel_weight_meta = channel_weight_meta
    self._flag_meta = flag_meta
    self._preferred_chunks = preferred_chunks
    self._store = store
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

    dim_names = ("time", "frequency", "corrprod")
    a_chunks = t_chunks, f_chunks, cp_chunks = tuple(max(c) for c in zip(*array_chunks))
    pt_chunks, pf_chunks, pcp_chunks = tuple(
      preferred_chunks.get(d, p) for d, p in zip(dim_names, a_chunks)
    )

    def archive_chunk_mul(preferred, archive):
      """Find the archive chunk multiplier nearest to the preferred chunk"""
      return max(1, round(preferred / archive)) * archive

    self.chunks = (
      time_chunks := archive_chunk_mul(pt_chunks, t_chunks),
      freq_chunks := archive_chunk_mul(pf_chunks, f_chunks),
      cp_chunks := archive_chunk_mul(pcp_chunks, cp_chunks),
    )

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

  async def _retrieve_chunk_async(
    self, chunk_extents: tuple[tuple[int, int], ...]
  ) -> VisFlagWeightData:
    """Fetch all four components for one virtual chunk concurrently."""

    def paths_and_starts(
      meta: ArchiveArrayMetadata, extents: tuple[tuple[int, int], ...]
    ) -> list[tuple[str, tuple[int, ...]]]:
      starts = [range((s // c) * c, e, c) for (s, e), c in zip(extents, meta.chunks)]
      result = []
      for cs in product(*starts):
        path_parts = "_".join(f"{i:05d}" for i in cs)
        result.append((f"{meta.prefix}/{meta.name}/{path_parts}.npy", cs))
      return result

    # Clip virtual chunk extents to actual array boundaries
    clipped: tuple[tuple[int, int], ...] = tuple(
      (s, min(e, self.shape[d])) for d, (s, e) in enumerate(chunk_extents)
    )

    vis_arr, weight_arr, cw_arr, flag_arr = await asyncio.gather(
      _fetch_component(
        self._store, paths_and_starts(self._vis_meta, clipped), self._vis_meta, clipped
      ),
      _fetch_component(
        self._store,
        paths_and_starts(self._weight_meta, clipped),
        self._weight_meta,
        clipped,
      ),
      _fetch_component(
        self._store,
        paths_and_starts(self._channel_weight_meta, clipped[:2]),
        self._channel_weight_meta,
        clipped[:2],
      ),
      _fetch_component(
        self._store,
        paths_and_starts(self._flag_meta, clipped),
        self._flag_meta,
        clipped,
      ),
    )

    data = VisFlagWeightData()
    data._vis = vis_arr
    data._weight = weight_arr * cw_arr[..., np.newaxis]
    data._flag = flag_arr
    return data

  def _maybe_retrieve_chunk(self, chunk, source_index, target_index):
    with self._locks[chunk]:
      if not cast(VisFlagWeightData, self._grid[chunk]).has_data:
        chunk_extents = tuple((c * s, (c + 1) * s) for c, s in zip(chunk, self.chunks))
        loop = AsyncLoopSingleton().instance
        future = asyncio.run_coroutine_threadsafe(
          self._retrieve_chunk_async(chunk_extents), loop
        )
        self._grid[chunk] = future.result()


class VFWAdapter(BackendArray):
  def __init__(self, array: str):
    self.array = array


if __name__ == "__main__":
  import sys
  from pathlib import Path as _Path

  sys.path.insert(0, str(_Path(__file__).parent.parent.parent.parent))

  from tests.conftest import SyntheticObservation, setup_mock_archive_server

  ntime = 8
  nfreq = 16
  nants = 4

  obs = SyntheticObservation("1234567890", ntime=ntime, nfreq=nfreq, nants=nants)
  obs.add_scan(range(0, 8), "track", "PKS1934")
  archive_path = _Path("/tmp/synthobs")
  obs.save_to_directory(archive_path)

  capture_block_id = obs.capture_block_id
  chunk_info = obs.create_telstate_dict()["chunk_info"]
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

  from obstore.store import HTTPStore
  from pytest_httpserver import HTTPServer

  httpserver = HTTPServer()
  httpserver.start()
  try:
    token = setup_mock_archive_server(
      httpserver, archive_path, capture_block_id, require_auth=False
    )
    assert token is None

    store = HTTPStore.from_url(
      httpserver.url_for("/"), client_options={"allow_http": True}
    )

    grid = VisFlagWeightGrid(
      meta["correlator_data"],
      meta["weights"],
      meta["weights_channel"],
      meta["flags"],
      {"time": 2, "frequency": 8, "corrprod": 4},
      store,
    )

    print(grid[slice(0, 6), np.array([0, 5, 3, 3, 4, 7, 2, 6])])
  finally:
    httpserver.clear()
    httpserver.stop()
