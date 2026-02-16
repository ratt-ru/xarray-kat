from typing import Any, Dict, TypedDict, Tuple
import numpy as np
import numpy.typing as npt
from xarray.backends.common import BackendArray

from xarray_kat.multiton import Multiton
from xarray_kat.xkat_types import ArchiveArrayMetadata

# A selection over (time, frequency, corrprod)
DimRangeT = Tuple[slice, slice, slice]

class PreferredChunksT(TypedDict):
  time: int
  frequency: int
  corrprod: int


class VisFlagWeightData:
  __slots__ = ("vis", "weight", "flag")

  vis: npt.NDArray
  weight: npt.NDArray
  flag: npt.NDArray

  def __init__(self):
    vis = np.ones(10)
    weight = np.ones(10)
    flag = np.ones(10)


class VisFlagWeightGrid:
  def __init__(
    self,
    vis_meta: ArchiveArrayMetadata,
    weight_meta: ArchiveArrayMetadata,
    channel_weight_meta: ArchiveArrayMetadata,
    flag_meta: ArchiveArrayMetadata,
    preferred_chunks: PreferredChunksT,
  ):
    self._vis_meta = vis_meta
    self._weight_meta = weight_meta
    self._channel_weight_meta = channel_weight_meta
    self._flag_meta = flag_meta
    self._preferred_chunks = preferred_chunks

    assert vis_meta.shape == weight_meta.shape == flag_meta.shape, "Archive Array shapes don't match"
    assert vis_meta.shape[:2] == channel_weight_meta.shape
    ntime, nfreq, ncorrprod = vis_meta.shape

    chunks = [
      vis_meta.chunks,
      weight_meta.chunks,
      channel_weight_meta.chunks + (-1,),
      flag_meta.chunks
    ]

    time_chunks, freq_chunks, cp_chunks = (max(c) for c in zip(*chunks))
    time_chunks = max(preferred_chunks.get("time", time_chunks), time_chunks)
    freq_chunks = max(preferred_chunks.get("frequency", freq_chunks), freq_chunks)
    cp_chunks = max(preferred_chunks.get("corrprod", cp_chunks), cp_chunks)

    ntime, rem = divmod(ntime, time_chunks)
    ntime += int(rem != 0)
    nfreq, rem = divmod(nfreq, freq_chunks)
    nfreq += int(rem != 0)
    ncorrprod, rem = divmod(ncorrprod, cp_chunks)
    ncorrprod += int(rem != 0)

    import time
    start = time.time()
    grid = np.asarray([VisFlagWeightData()] * (ntime*nfreq*ncorrprod)).reshape((ntime, nfreq, ncorrprod))




class VFWAdapter(BackendArray):
  def __init__(self, array: str):
    self.array = array

if __name__ == "__main__":
  prefix = "12345-sdp-l0"
  time_chunks = (1,) * 10
  freq_chunks = (256,) * 4
  ntime = sum(time_chunks)
  nfreq = sum(freq_chunks)
  ncorrprod = (7 * 6 // 2) * 4
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
      dim_labels[:len(value["shape"])],
      value["prefix"],
      value["chunks"],
      value["dtype"]
    ) for name, value in chunk_info.items()
  }

  grid = VisFlagWeightGrid(
    meta["correlator_data"],
    meta["weights"],
    meta["weights_channel"],
    meta["flags"],
    {"time": 2, "frequency": 512, "corrprod": 4})