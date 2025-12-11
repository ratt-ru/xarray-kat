from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
import tensorstore as ts

if TYPE_CHECKING:
  from xarray_kat.katdal_types import TelstateDataProducts
  from xarray_kat.multiton import Multiton


def calibration_solutions_store(
  data_products: Multiton[TelstateDataProducts],
  dim_labels: Tuple[str, ...],
  context: ts.Context,
):
  """A virtual array of complex calibration solutions"""
  chunk_info = data_products.instance.telstate["chunk_info"]
  vis_chunks = chunk_info["correlator_data"]["chunks"]
  dtype = chunk_info["correlator_data"]["dtype"]
  vis_shape = tuple(sum(dc) for dc in vis_chunks)
  if not all(all(dc[0] == c for c in dc[1:-1]) for dc in vis_chunks):
    raise ValueError(f"Visibility {vis_chunks} are not homogenous")

  # Defer the import
  from xarray_kat.third_party.vendored.katdal.applycal_minimal import (
    calc_correction_per_corrprod,
  )

  def read_chunk(
    domain: ts.IndexDomain, array: np.ndarray, params: ts.VirtualChunkedReadParameters
  ) -> ts.KvStore.TimestampedStorageGeneration:
    """This function is a variant of applycal_minimal._correction_block"""
    assert (cal_params := data_products.instance.calibration_params) is not None

    slices = domain.index_exp
    for n, dump in enumerate(range(slices[0].start, slices[0].stop)):
      array[n] = calc_correction_per_corrprod(dump, slices[1], cal_params)

  return ts.virtual_chunked(
    read_function=read_chunk,
    rank=len(vis_shape),
    domain=ts.IndexDomain(
      [ts.Dim(size=s, label=ll) for s, ll in zip(vis_shape, dim_labels)]
    ),
    dtype=np.dtype(dtype),
    chunk_layout=ts.ChunkLayout(chunk_shape=[c[0] for c in vis_chunks]),
    context=context,
  )
