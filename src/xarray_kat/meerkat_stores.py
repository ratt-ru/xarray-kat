import asyncio
import logging
from functools import reduce
from io import BytesIO
from operator import mul
from typing import Any, Collection, Dict, Tuple

import numpy as np
import numpy.typing as npt
import tensorstore as ts
from numpy.lib.format import read_array_header_1_0, read_array_header_2_0, read_magic

from xarray_kat.async_loop import AsyncLoopSingleton
from xarray_kat.multiton import Multiton

log = logging.getLogger(__name__)


def http_spec(
  endpoint: str, path: str, token: str | None
) -> Dict[str, Collection[str]]:
  """Creates a spec defining an http specification for accessing
  the MeerKAT HTTP archive.

  Args:
    endpoint: the http(s) endpoint
    path: Relative path from the endpoint
    token: The JWT token, if available

  Returns:
    Tensorstore kvstore specification
  """
  spec: Dict[str, Collection[str]] = {
    "driver": "http",
    "base_url": endpoint,
    "path": path,
  }

  if token:
    spec["headers"] = [f"Authorization: Bearer {token}"]

  return spec


def http_store_factory(
  endpoint: str,
  path: str,
  token: str | None = None,
  context: ts.Context | None = None,
) -> ts.TensorStore:
  """Creates an http(s) tensorstore referencing the specified
  endpoint and path.

  A jwt token is required if the endpoint is an https endpoint."""
  spec = http_spec(endpoint, path, token)
  return ts.KvStore.open(spec, context=context).result()


def read_array(raw_bytes: bytes) -> npt.NDArray | None:
  """Attempts to reconstruct the NumPy array from raw bytes
  Returns None if the bytes are truncated or otherwise corrupt

  Args:
    raw_bytes: bytes defining an array

  Returns:
    A numpy array, or None if reconstruction was not possible.
  """
  if len(raw_bytes) == 0:
    return None

  fp = BytesIO(raw_bytes)
  try:
    version = read_magic(fp)
    read_header = read_array_header_1_0 if version == (1, 0) else read_array_header_2_0
    shape, fortran_order, dtype = read_header(fp)
  except ValueError:
    return None

  # Are there enough bytes for a reconstruction?
  if len(array_bytes := raw_bytes[fp.tell() :]) != reduce(mul, shape, dtype.itemsize):
    return None

  assert not fortran_order, "Data should always be C-ordered"
  result = np.ndarray(shape, dtype, array_bytes)
  result.shape = shape
  return result


def virtual_chunked_store(
  http_store: Multiton[ts.TensorStore],
  data_type: str,
  data_schema: Dict[str, Any],
  dim_labels: Tuple[str, ...],
  missing_value: Any,
  context: ts.Context,
) -> ts.TensorStore:
  dtype = data_schema["dtype"]
  chunks = data_schema["chunks"]
  shape = tuple(sum(dc) for dc in chunks)
  if not all(all(dc[0] == c for c in dc[1:-1]) for dc in chunks):
    raise ValueError(f"{chunks} are not homogenous")

  def read_chunk(
    domain: ts.IndexDomain, array: np.ndarray, params: ts.VirtualChunkedReadParameters
  ) -> ts.KvStore.TimestampedStorageGeneration:
    """Reads the MeerKAT archive npy file associated with the domain,
    strips off the header and then assigns the resulting data
    into array:

    /correlator-data/00000_00000_00000.npy
    /correlator-data/00001_00000_00000.npy
    /correlator-data/00000_00032_00000.npy
    /flags/00000_00032_00000.npy
    """
    key_parts = [f"{o:05}" for o in domain.origin]
    key = f"/{data_type}/{'_'.join(key_parts)}.npy"

    async def key_reader(store, key: str):
      return await store.read(key)

    for attempt in range(3):
      log.debug("%d Read %s into domain %s", attempt, key, domain)
      coro = key_reader(http_store.instance, key)
      read_result = asyncio.run_coroutine_threadsafe(
        coro, AsyncLoopSingleton().instance
      ).result()

      if read_result.state == "missing":
        data = None
        break
      elif (
        read_result.state == "value"
        and (data := read_array(read_result.value)) is not None
      ):
        log.debug("Completed reading %s into domain %s", key, domain)
        break

    # Fill with defaults if retrieval failed
    if data is None:
      log.warning("Substituting %s for missing key %s", missing_value, key)
      array[...] = missing_value
      return read_result.stamp

    assert array.shape == data.shape
    chunk_domain = domain.translate_backward_by[domain.origin]
    array[...] = data[chunk_domain.index_exp]
    return read_result.stamp

  return ts.virtual_chunked(
    read_function=read_chunk,
    rank=len(shape),
    domain=ts.IndexDomain(
      [ts.Dim(size=s, label=ll) for s, ll in zip(shape, dim_labels)]
    ),
    dtype=dtype,
    chunk_layout=ts.ChunkLayout(chunk_shape=[c[0] for c in chunks]),
    context=context,
  )
