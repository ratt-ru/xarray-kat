import asyncio
import logging
from functools import reduce
from io import BytesIO
from operator import mul
from threading import Lock
from typing import Any, Collection, Dict, Tuple

import numpy as np
import numpy.typing as npt
import tensorstore as ts
from numpy.lib.format import read_array_header_1_0, read_array_header_2_0, read_magic

from xarray_kat.async_loop import AsyncLoopSingleton

log = logging.getLogger(__name__)

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


def http_spec_factory(
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
  http_store: ts.KvStore,
  data_type: str,
  data_schema: Dict[str, Any],
  dim_labels: Tuple[str, ...],
  missing_value: Any,
  context: ts.Context,
) -> ts.KvStore:
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

    async def key_reader(store, key):
      return await store.read(key)

    for attempt in range(3):
      log.debug("%d Read %s into domain %s", attempt, key, domain)
      read_result = asyncio.run_coroutine_threadsafe(
        key_reader(http_store, key), AsyncLoopSingleton().instance
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

    # Fill with defaults if it wasn't possible to retrieve something
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


class MeerkatStoreProvider:
  """Provides a tensorstore accessing MeerKAT chunk data.

  This pickleable class is necessary because a ``tensorstore.virtual_chunked``
  tensorstore is used to access MeerKAT chunks and the underlying
  ``read_chunk`` closure is not pickleable.

  Use as follows, for example:

  .. code-block:: python

    provider = MeerkatStoreProvider(
      "https://archive-gw-1.kat.ac.za", "eYabcdefg",
      "correlator-data", {...}, {...}, {...})

    provider.store
  """

  # Immutable members comprehensively defining
  # the creation of the underlying tensorstore
  _endpoint: str
  _token: str | None
  _data_type: str
  _chunk_schema: Dict[str, Any]
  _http_context: ts.Context | None
  _virtual_context: ts.Context | None

  # Cached virtual store and protecting lock
  _virtual_store: ts.TensorStore | None
  _lock: Lock

  def __init__(
    self,
    data_type: str,
    chunk_schema: Dict[str, Any],
    endpoint: str,
    token: str | None = None,
    http_context: Dict[str, Collection[str]] | ts.Context | None = None,
    virtual_context: Dict[str, Collection[str]] | ts.Context | None = None,
  ):
    f"""Creates a MeerkatStoreProvider

    Args:
      data_type: The type of MeerKAT archive data.
        Should be a key of ``telstate['chunk_info']``
        and should be one of {list(DATA_TYPE_LABELS.keys())}.
      chunk_schema: The schema describing chunked MeerKAT archive data.
        Should be the value in ``telstate['chunk_info']``
        associated with ``data_type``.
      endpoint: The http(s) endpoint of the MeerKAT archive.
      token: The JWT token granting access to the endpoint.
        Required if the endpoint is an https endpoint.
    """
    self._endpoint = endpoint
    self._token = token
    self._data_type = data_type
    self._chunk_schema = chunk_schema

    if isinstance(http_context, dict):
      self._http_context = ts.Context(ts.Context.Spec(http_context))
    elif http_context is None or isinstance(http_context, ts.Context):
      self._http_context = http_context
    else:
      raise TypeError(f"http_context {http_context}")

    if isinstance(virtual_context, dict):
      self._virtual_context = ts.Context(ts.Context.Spec(virtual_context))
    elif http_context is None or isinstance(virtual_context, ts.Context):
      self._virtual_context = virtual_context
    else:
      raise TypeError(f"virtual_context {virtual_context}")

    self._virtual_store = None
    self._lock = Lock()

  def __reduce__(self):
    return (
      MeerkatStoreProvider,
      (
        self._data_type,
        self._chunk_schema,
        self._endpoint,
        self._token,
        self._http_context,
        self._virtual_context,
      ),
    )

  @property
  def store(self):
    """Returns a tensorstore defined by this StoreProvider"""
    with self._lock:
      if self._virtual_store is None:
        try:
          dim_labels = DATA_TYPE_LABELS[self._data_type]
          missing_value = MISSING_VALUES[self._data_type]
        except KeyError:
          raise ValueError(
            f"{self._data_type} is not a valid meerkat archive data type: "
            f"{list(DATA_TYPE_LABELS.keys())}"
          )

        http_store_spec = http_spec_factory(
          self._endpoint, self._chunk_schema["prefix"], self._token
        )
        http_store = ts.KvStore.open(
          http_store_spec, context=self._http_context
        ).result()
        self._virtual_store = virtual_chunked_store(
          http_store,
          self._data_type,
          self._chunk_schema,
          dim_labels,
          missing_value,
          self._virtual_context,
        )

      return self._virtual_store
