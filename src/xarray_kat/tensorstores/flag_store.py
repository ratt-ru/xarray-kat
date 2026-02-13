from __future__ import annotations

from typing import TYPE_CHECKING

import tensorstore as ts

from xarray_kat.third_party.vendored.katdal.applycal_minimal import (
  apply_flags_correction,
  calc_correction_per_corrprod,
)

if TYPE_CHECKING:
  import numpy as np
  import numpy.typing as npt

  from xarray_kat.katdal_types import TelstateDataProducts
  from xarray_kat.multiton import Multiton
  from xarray_kat.xkat_types import ArchiveArrayMetadata


def final_flag_store(
  base_flag_store: Multiton[ts.TensorStore],
  data_products: Multiton[TelstateDataProducts],
  array_metadata: ArchiveArrayMetadata,
  context: ts.Context,
):
  """Creates a TensorStore representing flagged values.
  These are inherited from the base flag store, with optional
  calibration solutions applied"""

  def read_chunk(
    domain: ts.IndexDomain, array: np.ndarray, params: ts.VirtualChunkedReadParameters
  ) -> ts.KvStore.TimestampedStorageGeneration:
    # Issue flag read future
    (flags_future := base_flag_store.instance[domain].read(batch=params.batch)).force()

    # Prepare calibration solutions while waiting for data
    cal_solutions: npt.NDArray | None = None

    if (cal_params := data_products.instance.calibration_params) is not None:
      cal_solutions = np.empty(array.shape, np.complex64)
      slices = domain.index_exp
      for n, dump in enumerate(range(slices[0].start, slices[0].stop)):
        cal_solutions[n] = calc_correction_per_corrprod(dump, slices[1], cal_params)

    array[:] = flags_future.result()

    if cal_solutions is not None:
      apply_flags_correction(array, cal_solutions)

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
