import logging
from typing import Any, Dict, Tuple

import numpy as np
import tensorstore as ts

from xarray_kat.katdal_types import AutoCorrelationIndices
from xarray_kat.multiton import Multiton
from xarray_kat.stores.base_store import read_array
from xarray_kat.third_party.vendored.katdal.applycal_minimal import apply_vis_correction
from xarray_kat.third_party.vendored.katdal.van_vleck import autocorr_lookup_table
from xarray_kat.types import VanVleckLiteralType

log = logging.getLogger(__name__)

# Default MeerKAT F-engine levels
MEERKAT_F_ENGINE_OUTPUT_LEVELS = np.arange(-127.0, 128.0)
LOOKUP_TABLES = Multiton(autocorr_lookup_table, MEERKAT_F_ENGINE_OUTPUT_LEVELS)


def base_visibility_virtual_store(
  http_store: Multiton[ts.TensorStore],
  data_schema: Dict[str, Any],
  dim_labels: Tuple[str, ...],
  missing_value: Any,
  autocorrs: Multiton[AutoCorrelationIndices],
  van_vleck: VanVleckLiteralType,
  context: ts.Context | None,
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
    corrprods: Correlation product array derived from telstate
    van_vleck: Literal controlling application of Van Vleck Corrections.
    context: TensorStore context to associate with the returned store.

  Returns:
    A TensorStore representing visibility data, possibly with
    Van Vleck Corrections applied.
  """
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
      data = array
    else:
      assert array.shape == data.shape
      chunk_domain = domain.translate_backward_by[domain.origin]
      array[...] = data[chunk_domain.index_exp]

    # Apply van vleck corrections
    if van_vleck == "autocorr":
      auto_indices = autocorrs.instance.auto_indices
      array[..., auto_indices] = np.interp(
        data[..., auto_indices].real,
        LOOKUP_TABLES.instance.quantised,
        LOOKUP_TABLES.instance.true,
      )
    elif van_vleck != "off":
      raise ValueError(f"Invalid van_vleck value {van_vleck}")

    return result.stamp

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


def final_visibility_virtual_store(
  base_vis_store: Multiton[ts.TensorStore],
  calibration_solution_store: Multiton[ts.TensorStore] | None,
  data_schema: Dict[str, Any],
  dim_labels: Tuple[str, ...],
  context: ts.Context | None,
):
  """Creates a virtual_chunked TensorStore that derives its
  values from a base visibility store and, possibly a calibration solution store. """
  dtype = data_schema["dtype"]
  chunks = data_schema["chunks"]
  shape = tuple(sum(dc) for dc in chunks)
  if not all(all(dc[0] == c for c in dc[1:-1]) for dc in chunks):
    raise ValueError(f"{chunks} are not homogenous")

  def read_chunk(
    domain: ts.IndexDomain, array: np.ndarray, params: ts.VirtualChunkedReadParameters
  ) -> ts.KvStore.TimestampedStorageGeneration:
    vis_rr = base_vis_store.instance[domain].read()
    vis_rr.force()

    if calibration_solution_store is not None:
      cal_rr = calibration_solution_store.instance[domain].read()
      cal_rr.force()

    array[:] = vis_rr.result()

    if calibration_solution_store is not None:
      apply_vis_correction(array, cal_rr.result())

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
