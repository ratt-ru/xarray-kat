import asyncio
from io import BytesIO
from typing import Any, Collection, Dict, Tuple

import numpy as np
import numpy.typing as npt
import tensorstore as ts
from katsdptelstate import TelescopeState
from numpy.lib.format import read_array_header_1_0, read_array_header_2_0, read_magic

from xarray_kat.async_loop import AsyncLoopSingleton

FULL_POLARIZATIONS = ["HH", "HV", "VH", "VV"]
DATA_TYPE_LABELS = {
  "correlator_data": (
    ("time", "frequency", "corrprod"),
    ("time", "baseline", "frequency", "polarization"),
  ),
  "flags": (
    ("time", "frequency", "corrprod"),
    ("time", "baseline", "frequency", "polarization"),
  ),
  "weights": (
    ("time", "frequency", "corrprod"),
    ("time", "baseline", "frequency", "polarization"),
  ),
  "weights_channel": (("time", "frequency"), ("time", "frequency")),
}


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
  cp_argsort: npt.NDArray,
  dim_labels: Tuple[Tuple[str, ...], Tuple[str, ...]],
  context: ts.Context,
) -> ts.KvStore:
  dtype = data_schema["dtype"]
  prefix = data_schema["prefix"]
  chunks = data_schema["chunks"]
  shape = tuple(sum(dc) for dc in chunks)
  if not all(all(dc[0] == c for c in dc[1:-1]) for dc in chunks):
    raise ValueError(f"{chunks} are not homogenous")

  source_labels, dest_labels = dim_labels

  # Rework the domain shape and chunks if we have
  # corrprods to reflect the transpose inside read_chunk
  if corrprods_transpose := (source_labels == ("time", "frequency", "corrprod")):
    if len(cp_argsort) % len(FULL_POLARIZATIONS) != 0:
      raise ValueError(
        f"corrprod {len(cp_argsort)} is not a multiple "
        f"of polarizations {len(FULL_POLARIZATIONS)}"
      )

    ncp = shape[-1]
    assert chunks[-1] == (ncp,), f"corrprod chunks {chunks[-1]} != {ncp}"
    nbl = ncp // len(FULL_POLARIZATIONS)
    npol = len(FULL_POLARIZATIONS)
    chunks = (chunks[0], (nbl,), chunks[1], (npol,))
    shape = tuple(sum(dc) for dc in chunks)

  time_index = dest_labels.index("time")
  freq_index = dest_labels.index("frequency")

  domain = ts.IndexDomain(
    [ts.Dim(size=s, label=ll) for s, ll in zip(shape, dest_labels)]
  )
  chunk_layout = ts.ChunkLayout(chunk_shape=[c[0] for c in chunks])

  def read_chunk(
    domain: ts.IndexDomain, array: np.ndarray, params: ts.VirtualChunkedReadParameters
  ) -> ts.KvStore.TimestampedStorageGeneration:
    """Reads the MeerKAT archive npy file associated with the domain,
    strips off the header and then assigns the resulting data
    into array"""

    # Hacky calculation of the key for the corrprod case
    # assumes a single corrprod chunk (which seems to be the case)
    if corrprods_transpose:
      key_parts = [f"{domain.origin[i]:05}" for i in (time_index, freq_index)] + [
        f"{0:05}"
      ]
    else:
      key_parts = [f"{o:05}" for o in domain.origin]

    key = f"{prefix}/{data_type}/{'_'.join(key_parts)}.npy"
    # print(f"Initiating read {key} into domain {domain}")

    async def key_reader(store, key):
      return await store.read(key)

    read_result = asyncio.run_coroutine_threadsafe(
      key_reader(http_store, key), AsyncLoopSingleton().instance
    ).result()

    # print(f"Completed read {key} into domain {domain}")
    raw_bytes = read_result.value
    fp = BytesIO(raw_bytes)
    version = read_magic(fp)
    read_header = read_array_header_1_0 if version == (1, 0) else read_array_header_2_0
    shape, fortran_order, dtype = read_header(fp)
    data = np.frombuffer(raw_bytes[fp.tell() :], dtype=dtype)
    if fortran_order:
      data.shape = shape[::-1]
      data = data.transpose()
    else:
      data.shape = shape

    # TODO: Improve the efficiency of this
    # 1. Use numba to do it (as in katdal)
    # 2. Build a C++ wheel to do it
    # 3. Wait for tensorstore to implement kvstore reshape
    if corrprods_transpose:
      data = (
        data[..., cp_argsort].reshape(shape[:2] + (nbl, npol)).transpose(0, 2, 1, 3)
      )

    assert array.shape == data.shape

    chunk_domain = domain.translate_backward_by[domain.origin]
    tensor_data = ts.array(data)
    array[...] = tensor_data[chunk_domain]
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
    corrprods = telstate["bls_ordering"]
    cp_argsort = np.array(
      sorted(range(len(corrprods)), key=lambda i: tuple(corrprods[i]))
    )
    http_store_spec = http_spec_factory(endpoint, token)
    http_store = ts.KvStore.open(http_store_spec).result()

    virtual_store_spec = ts.Context.Spec({"data_copy_concurrency": {"limit": 12}})
    virtual_store_context = ts.Context(virtual_store_spec)

    tensorstores = {}

    for data_type, data_schema in chunk_info.items():
      dim_labels = DATA_TYPE_LABELS[data_type]
      store = virtual_chunked_store(
        http_store,
        data_type,
        data_schema,
        cp_argsort,
        dim_labels,
        virtual_store_context,
      )
      tensorstores[data_type] = store

    return tensorstores
