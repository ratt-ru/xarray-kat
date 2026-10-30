from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, get_args

import numpy as np
import numpy as npt
import tensorstore as ts
from rarg_python_patterns import Multiton

from xarray_kat.async_loop import AsyncLoopSingleton
from xarray_kat.katdal_types import AutoCorrelationIndices
from xarray_kat.stores.base_store import read_array
from xarray_kat.third_party.vendored.katdal.applycal_minimal import (
  apply_vis_correction,
  calc_correction_per_corrprod,
)
from xarray_kat.third_party.vendored.katdal.van_vleck import autocorr_lookup_table
from xarray_kat.xkat_types import VanVleckLiteralType

if TYPE_CHECKING:
  from xarray_kat.katdal_types import TelstateDataProducts
  from xarray_kat.xkat_types import ArchiveArrayMetadata

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

  def assign_chunk(
    domain: ts.IndexDomain, array: np.ndarray, key: str, result: ts.KvStore.ReadResult
  ) -> None:
    data = None

    if result.state == "value" and (data := read_array(result.value)) is not None:
      log.debug("Read %s into domain %s", key, domain)

    # Fill with defaults if retrieval failed
    if data is None:
      log.warning("Defaulting to %s for missing key %s", default_value, key)
      array[...] = default_value
      data = array
    else:
      assert array.shape == data.shape
      array[...] = data

    # Apply van vleck corrections
    if van_vleck == "autocorr":
      auto_indices = autocorrs.instance.auto_indices
      array[..., auto_indices] = np.interp(
        data[..., auto_indices].real,
        LOOKUP_TABLES.instance.quantised,
        LOOKUP_TABLES.instance.true,
      )
    elif van_vleck != "off":
      raise ValueError(
        f"Invalid van_vleck value {van_vleck} "
        f"Should be one of {get_args(VanVleckLiteralType)}"
      )

  async def read_chunk(
    domain: ts.IndexDomain, array: np.ndarray, params: ts.VirtualChunkedReadParameters
  ) -> None:
    """Reads the MeerKAT archive npy file associated with the domain,
    strips off the header and then assigns the resulting data
    into array:

    00000_00000_00000.npy
    00001_00000_00000.npy
    00000_00032_00000.npy

    The http read is awaited rather than blocked on: a read function
    that blocks on another TensorStore operation can deadlock
    tensorstore's thread pool.
    """
    key_parts = [f"{o:05}" for o in domain.origin]
    key = f"{'_'.join(key_parts)}.npy"
    result = await http_store.instance.read(key, batch=params.batch)
    # Decode, copy and correct off the event loop thread
    await asyncio.to_thread(assign_chunk, domain, array, key, result)

  return ts.virtual_chunked(
    read_function=read_chunk,
    loop=AsyncLoopSingleton().instance,
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
  values from a base visibility store and, possibly a calibration solution store.

  The read function awaits the base store rather than blocking on it:
  a read function that blocks on another TensorStore read can deadlock
  tensorstore's thread pool."""

  def assign_chunk(domain: ts.IndexDomain, array: np.ndarray, vis: npt.NDArray) -> None:
    array[:] = vis

    # Possibly apply calibration solutions
    if (cal_params := data_products.instance.calibration_params) is not None:
      cal_solutions = np.empty_like(array, dtype=np.complex64)
      slices = domain.index_exp
      for n, dump in enumerate(range(slices[0].start, slices[0].stop)):
        cal_solutions[n] = calc_correction_per_corrprod(dump, slices[1], cal_params)
      array[:] = apply_vis_correction(array, cal_solutions)

  async def read_chunk(
    domain: ts.IndexDomain, array: np.ndarray, params: ts.VirtualChunkedReadParameters
  ) -> None:
    vis = await base_vis_store.instance[domain].read(batch=params.batch)
    # Copy and calibrate off the event loop thread
    await asyncio.to_thread(assign_chunk, domain, array, vis)

  return ts.virtual_chunked(
    read_function=read_chunk,
    loop=AsyncLoopSingleton().instance,
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
