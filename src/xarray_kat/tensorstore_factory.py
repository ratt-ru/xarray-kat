import asyncio
import logging
from functools import reduce
from io import BytesIO
from operator import mul
from typing import Any, Collection, Dict, Tuple

import numpy as np
import numpy.typing as npt
import tensorstore as ts
from katsdptelstate import TelescopeState
from numpy.lib.format import read_array_header_1_0, read_array_header_2_0, read_magic

from xarray_kat.async_loop import AsyncLoopSingleton

FULL_POLARIZATIONS = ["HH", "HV", "VH", "VV"]
MISSING_VALUES = {
  "correlator_data": np.nan + np.nan * 1j,
  "flags": 1,
  "weights": 0,
  "weights_channel": 0.0,
}
DATA_TYPE_LABELS = {
  "correlator_data": ("time", "frequency", "corrprod"),
  "flags": ("time", "frequency", "corrprod"),
  "weights": ("time", "frequency", "corrprod"),
  "weights_channel": ("time", "frequency"),
}


log = logging.getLogger(__name__)


def http_spec_factory(endpoint: str, token: str | None) -> Dict[str, Collection[str]]:
  spec: Dict[str, Collection[str]] = {
    "driver": "http",
    "base_url": endpoint,
  }

  if token:
    spec["headers"] = [f"Authorization: Bearer {token}"]

  return spec


def virtual_chunked_store(
  http_store: ts.KvStore,
  data_type: str,
  data_schema: Dict[str, Any],
  dim_labels: Tuple[str, ...],
  missing_value: Any,
  context: ts.Context,
) -> ts.KvStore:
  dtype = data_schema["dtype"]
  prefix = data_schema["prefix"]
  chunks = data_schema["chunks"]
  shape = tuple(sum(dc) for dc in chunks)
  if not all(all(dc[0] == c for c in dc[1:-1]) for dc in chunks):
    raise ValueError(f"{chunks} are not homogenous")

  domain = ts.IndexDomain(
    [ts.Dim(size=s, label=ll) for s, ll in zip(shape, dim_labels)]
  )
  chunk_layout = ts.ChunkLayout(chunk_shape=[c[0] for c in chunks])

  def read_chunk(
    domain: ts.IndexDomain, array: np.ndarray, params: ts.VirtualChunkedReadParameters
  ) -> ts.KvStore.TimestampedStorageGeneration:
    """Reads the MeerKAT archive npy file associated with the domain,
    strips off the header and then assigns the resulting data
    into array
    {cbid}/correlator-data/00000_00000_00000.npy
    {cbid}/correlator-data/00001_00000_00000.npy
    {cbid}/correlator-data/00000_00032_00000.npy
    {cbid}/flags/00000_00032_00000.npy

    """
    key_parts = [f"{o:05}" for o in domain.origin]
    key = f"{prefix}/{data_type}/{'_'.join(key_parts)}.npy"

    async def key_reader(store, key):
      return await store.read(key)

    def read_array(raw_bytes: bytes) -> npt.NDArray | None:
      """Attempts to reconstruct the NumPy array from raw bytes
      Returns None if the bytes are truncated or otherwise corrupt"""
      if len(raw_bytes) == 0:
        return None

      fp = BytesIO(raw_bytes)
      try:
        version = read_magic(fp)
        read_header = (
          read_array_header_1_0 if version == (1, 0) else read_array_header_2_0
        )
        shape, fortran_order, dtype = read_header(fp)
      except ValueError:
        return None

      # Are there enough bytes for a reconstruction?
      if len(array_bytes := raw_bytes[fp.tell() :]) != reduce(
        mul, shape, dtype.itemsize
      ):
        return None

      assert not fortran_order, "Data should always be C-ordered"
      return np.frombuffer(array_bytes, dtype=dtype).reshape(shape)

    for attempt in range(3):
      log.debug("%d Read %s into domain %s", attempt, key, domain)
      read_result = asyncio.run_coroutine_threadsafe(
        key_reader(http_store, key), AsyncLoopSingleton().instance
      ).result()

      if (data := read_array(read_result.value)) is not None:
        log.debug("Completed read %s into domain %s", key, domain)
        break

    # Couldn't read a reasonable value, bail out
    if data is None:
      log.warning("Substituting %s for truncated key %s", missing_value, key)
      array[...] = missing_value
      return read_result.stamp

    assert array.shape == data.shape

    chunk_domain = domain.translate_backward_by[domain.origin]
    array[...] = data[chunk_domain.index_exp]
    return read_result.stamp

  return ts.virtual_chunked(
    read_function=read_chunk,
    rank=len(shape),
    domain=domain,
    dtype=dtype,
    chunk_layout=chunk_layout,
    context=context,
  )


class StoreFactory:
  @classmethod
  def make(
    cls, telstate: TelescopeState, endpoint: str, token: str | None = None
  ) -> Dict[str, ts.KvStore]:
    assert (
      endpoint.startswith("https") and isinstance(token, str)
    ) or endpoint.startswith("http")
    chunk_info = telstate["chunk_info"]
    http_store_spec = http_spec_factory(endpoint, token)
    http_store = ts.KvStore.open(http_store_spec).result()

    virtual_store_spec = ts.Context.Spec({"data_copy_concurrency": {"limit": 12}})
    virtual_store_context = ts.Context(virtual_store_spec)
    tensorstores = {}

    for data_type, data_schema in chunk_info.items():
      store = virtual_chunked_store(
        http_store,
        data_type,
        data_schema,
        DATA_TYPE_LABELS[data_type],
        MISSING_VALUES[data_type],
        virtual_store_context,
      )
      tensorstores[data_type] = store

    return tensorstores
