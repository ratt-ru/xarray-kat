from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy as npt
import tensorstore as ts

from xarray_kat.katdal_types import AutoCorrelationIndices
from xarray_kat.multiton import Multiton
from xarray_kat.stores.base_store import read_array
from xarray_kat.third_party.vendored.katdal.applycal_minimal import (
  apply_vis_correction,
  calc_correction_per_corrprod,
)
from xarray_kat.third_party.vendored.katdal.van_vleck import autocorr_lookup_table
from xarray_kat.xkat_types import VanVleckLiteralType

if TYPE_CHECKING:
  from xarray_kat.katdal_types import TelstateDataProducts
  from xarray_kat.types import ArchiveArrayMetadata

log = logging.getLogger(__name__)

# Default MeerKAT F-engine levels
MEERKAT_F_ENGINE_OUTPUT_LEVELS = np.arange(-127.0, 128.0)
LOOKUP_TABLES = Multiton(autocorr_lookup_table, MEERKAT_F_ENGINE_OUTPUT_LEVELS)


def base_visibility_virtual_store(
  http_store: Multiton[ts.TensorStore],
  array_metadata: ArchiveArrayMetadata,
  autocorrs: Multiton[AutoCorrelationIndices],
  van_vleck: VanVleckLiteralType,
  context: ts.Context | None,
) -> ts.TensorStore:
  """Creates a virtual_chunked TensorStore over a set of keys in an http_store.

  Args:
    http_store: A http KvStore. It's path should be set to
    array_metadata: Archive array metadata
    corrprods: Correlation product array derived from telstate
    van_vleck: Literal controlling application of Van Vleck Corrections.
    context: TensorStore context to associate with the returned store.

  Returns:
    A TensorStore representing visibility data, possibly with
    Van Vleck Corrections applied.
  """
  default_value = array_metadata.default

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

    if (result := http_store.instance.read(key, batch=params.batch).result()).state == "value" and (
      data := read_array(result.value)
    ) is not None:
      log.debug("Read %s into domain %s", key, domain)

    # Fill with defaults if retrieval failed
    if data is None:
      log.warning("Defaulting to %s for missing key %s", default_value, key)
      array[...] = default_value
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

  return ts.virtual_chunked(
    read_function=read_chunk,
    rank=array_metadata.rank,
    domain=ts.IndexDomain(
      [
        ts.Dim(size=s, label=ll)
        for s, ll in zip(array_metadata.shape, array_metadata.dim_labels)
      ]
    ),
    dtype=array_metadata.dtype,
    chunk_layout=ts.ChunkLayout(chunk_shape=array_metadata.chunks),
    context=context,
  )


def final_visibility_virtual_store(
  base_vis_store: Multiton[ts.TensorStore],
  array_metadata: ArchiveArrayMetadata,
  data_products: Multiton[TelstateDataProducts],
  context: ts.Context | None,
):
  """Creates a virtual_chunked TensorStore that derives its
  values from a base visibility store and, possibly a calibration solution store."""

  def read_chunk(
    domain: ts.IndexDomain, array: np.ndarray, params: ts.VirtualChunkedReadParameters
  ) -> ts.KvStore.TimestampedStorageGeneration:
    # Issue visibility data future
    (vis_future := base_vis_store.instance[domain].read(batch=params.batch)).force()

    # Prepare calibration solutions while waiting for visibilities
    cal_solutions: npt.NDArray | None = None

    if (cal_params := data_products.instance.calibration_params) is not None:
      cal_solutions = np.empty_like(array, dtype=np.complex64)
      slices = domain.index_exp
      for n, dump in enumerate(range(slices[0].start, slices[0].stop)):
        cal_solutions[n] = calc_correction_per_corrprod(dump, slices[1], cal_params)

    # Read visibilities
    array[:] = vis_future.result()

    # Possibly apply calibration solutions
    if cal_solutions is not None:
      apply_vis_correction(array, cal_solutions)

  return ts.virtual_chunked(
    read_function=read_chunk,
    rank=array_metadata.rank,
    domain=ts.IndexDomain(
      [
        ts.Dim(size=s, label=ll)
        for s, ll in zip(array_metadata.shape, array_metadata.dim_labels)
      ]
    ),
    dtype=array_metadata.dtype,
    chunk_layout=ts.ChunkLayout(chunk_shape=array_metadata.chunks),
    context=context,
  )
