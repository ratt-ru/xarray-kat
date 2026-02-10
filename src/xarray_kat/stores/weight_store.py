from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import tensorstore as ts

from xarray_kat.third_party.vendored.katdal.applycal_minimal import (
  apply_weights_correction,
  calc_correction_per_corrprod,
)
from xarray_kat.third_party.vendored.katdal.vis_flags_weights_minimal import (
  weight_power_scale,
)

if TYPE_CHECKING:
  from xarray_kat.katdal_types import AutoCorrelationIndices, TelstateDataProducts
  from xarray_kat.multiton import Multiton
  from xarray_kat.xkat_types import ArchiveArrayMetadata

def scaled_weight_store(
  int_weights_store: Multiton[ts.TensorStore],
  channel_weights_store: Multiton[ts.TensorStore],
  vis_store: Multiton[ts.TensorStore],
  data_products: Multiton[TelstateDataProducts],
  autocorrs: Multiton[AutoCorrelationIndices],
  array_metadata: ArchiveArrayMetadata,
  apply_scaling: bool,
  context: ts.Context | None,
):
  """Combines weights and channel_weights and scales the weights
  by the visibility auto-correlations

  Args:
    int_weights_store: uint8 weights TensorStore of shape
      ``(time, frequency, corrprod)``.
    channel_weights_store: float32 weights TensorStore of shape ``(time, frequency)``.
    vis_store: complex visibility TensorStore of shape ``(time, frequency, corrprod)``.
    data_products: Telescope State Data Products
    autocorrs: Autocorrelation indices mapping.
    chunk_info: telstate chunk info dictionary
    apply_scaling: bool whether scaling should be applied or reversed.
      Should be derived from telstate["needs_weight_power_scale"].
    dim_labels: Dimension labels applied to the returned store.
      Should be ``(time, frequency, corrprod)``.
    context: TensorStore context associated with the returned store.

  Returns:
    A TensorStore representing the weights scaled by the visibilities.

  """

  def read_chunk(
    domain: ts.IndexDomain, array: np.ndarray, params: ts.VirtualChunkedReadParameters
  ) -> ts.KvStore.TimestampedStorageGeneration:
    cws = channel_weights_store.instance
    iws = ts.cast(int_weights_store.instance, cws.dtype)

    # Issue reads to the underlying stores
    (int_weights_future := iws[domain].read(batch=params.batch)).force()
    (channel_weights_future := cws[domain[:-1]].read(batch=params.batch)).force()
    (vis_future := vis_store.instance[domain].read(batch=params.batch)).force()

    # Prepare calibration solutions while waiting for data
    cal_solutions: npt.NDArray | None = None

    if (cal_params := data_products.instance.calibration_params) is not None:
      cal_solutions = np.empty_like(array.shape, np.complex64)
      slices = domain.index_exp
      for n, dump in enumerate(range(slices[0].start, slices[0].stop)):
        cal_solutions[n] = calc_correction_per_corrprod(dump, slices[1], cal_params)

    weights = int_weights_future.result()
    weights *= channel_weights_future.result()[..., None]

    array[...] = weight_power_scale(
      vis_future.result(),
      weights,
      autocorrs.instance.auto_indices,
      autocorrs.instance.index1,
      autocorrs.instance.index2,
      divide=apply_scaling,
    )

    if cal_solutions is not None:
      apply_weights_correction(array, cal_solutions)

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
