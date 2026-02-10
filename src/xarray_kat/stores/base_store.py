import logging
from functools import reduce
from io import BytesIO
from operator import mul

import numpy as np
import numpy.typing as npt
import tensorstore as ts
from numpy.lib.format import read_array_header_1_0, read_array_header_2_0, read_magic

from xarray_kat.multiton import Multiton
from xarray_kat.xkat_types import ArchiveArrayMetadata

log = logging.getLogger(__name__)


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


def base_virtual_store(
  http_store: Multiton[ts.TensorStore],
  array_meta: ArchiveArrayMetadata,
  context: ts.Context,
) -> ts.TensorStore:
  """Creates a virtual_chunked TensorStore over a set of keys in an http_store.

  Args:
    http_store: A http KvStore. It's path should be set to
    data_schema: Dictionary derived from telstate described the
      schema of this array, in particular it's data type (dtype)
      and chunking (chunks).
    dim_labels: The labels associated with each dimension of
      the data
    missing_value: Value substituted for the portion of the
      store corresponding to a missing key in the http KvStore.
    context: TensorStore context to associated with the returned
      store.
  """

  missing_value = array_meta.default

  def read_chunk(
    domain: ts.IndexDomain, array: np.ndarray, params: ts.VirtualChunkedReadParameters
  ) -> ts.KvStore.TimestampedStorageGeneration:
    """Reads the MeerKAT archive npy file associated with the domain,
    strips off the header and then assigns the resulting data
    into array:

    00000_00000_00000.npy
    00001_00000_00000.npy
    00000_00032_00000.npy
    """
    key_parts = [f"{o:05}" for o in domain.origin]
    key = f"{'_'.join(key_parts)}.npy"
    log.debug("%d Read %s into domain %s", key, domain)
    data = None

    if (result := http_store.instance.read(key).result()).state == "value" and (
      data := read_array(result.value)
    ) is not None:
      log.debug("Read %s into domain %s", key, domain)

    # Fill with defaults if retrieval failed
    if data is None:
      log.warning("Defaulting to %s for missing key %s", missing_value, key)
      array[...] = missing_value
      return result.stamp

    assert array.shape == data.shape
    chunk_domain = domain.translate_backward_by[domain.origin]
    array[...] = data[chunk_domain.index_exp]
    return result.stamp

  return ts.virtual_chunked(
    read_function=read_chunk,
    rank=len(array_meta.shape),
    domain=ts.IndexDomain(
      [
        ts.Dim(size=s, label=ll)
        for s, ll in zip(array_meta.shape, array_meta.dim_labels)
      ]
    ),
    dtype=array_meta.dtype,
    chunk_layout=ts.ChunkLayout(chunk_shape=array_meta.chunks),
    context=context,
  )
