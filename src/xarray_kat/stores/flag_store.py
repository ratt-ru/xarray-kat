from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import tensorstore as ts

from xarray_kat.async_loop import AsyncLoopSingleton
from xarray_kat.third_party.vendored.katdal.applycal_minimal import (
  apply_flags_correction,
  calc_correction_per_corrprod,
)

if TYPE_CHECKING:
  import numpy.typing as npt
  from rarg_python_patterns.multiton import Multiton

  from xarray_kat.katdal_types import TelstateDataProducts
  from xarray_kat.xkat_types import ArchiveArrayMetadata


def final_flag_store(
  base_flag_store: Multiton[ts.TensorStore],
  data_products: Multiton[TelstateDataProducts],
  array_metadata: ArchiveArrayMetadata,
  context: ts.Context,
):
  """Creates a TensorStore representing flagged values.
  These are inherited from the base flag store, with optional
  calibration solutions applied.

  The read function awaits the base store rather than blocking on it:
  a read function that blocks on another TensorStore read can deadlock
  tensorstore's thread pool."""

  def assign_chunk(
    domain: ts.IndexDomain, array: np.ndarray, flags: npt.NDArray
  ) -> None:
    array[:] = flags

    if (cal_params := data_products.instance.calibration_params) is not None:
      cal_solutions = np.empty(array.shape, np.complex64)
      slices = domain.index_exp
      for n, dump in enumerate(range(slices[0].start, slices[0].stop)):
        cal_solutions[n] = calc_correction_per_corrprod(dump, slices[1], cal_params)
      array[:] = apply_flags_correction(array, cal_solutions)

  async def read_chunk(
    domain: ts.IndexDomain, array: np.ndarray, params: ts.VirtualChunkedReadParameters
  ) -> None:
    flags = await base_flag_store.instance[domain].read(batch=params.batch)
    # Copy and calibrate off the event loop thread
    await asyncio.to_thread(assign_chunk, domain, array, flags)

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
