from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import tensorstore as ts

from xarray_kat.async_loop import AsyncLoopSingleton
from xarray_kat.third_party.vendored.katdal.applycal_minimal import (
  apply_weights_correction,
  calc_correction_per_corrprod,
)
from xarray_kat.third_party.vendored.katdal.vis_flags_weights_minimal import (
  weight_power_scale,
)

if TYPE_CHECKING:
  from rarg_python_patterns.multiton import Multiton

  from xarray_kat.katdal_types import AutoCorrelationIndices, TelstateDataProducts
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

    The read function awaits the underlying stores rather than blocking on
    them: a read function that blocks on another TensorStore read can
    deadlock tensorstore's thread pool.
  """

  def assign_chunk(
    domain: ts.IndexDomain,
    array: np.ndarray,
    weights: npt.NDArray,
    channel_weights: npt.NDArray,
    autocorr_vis: npt.NDArray,
  ) -> None:
    weights *= channel_weights[..., np.newaxis]

    # autocorr_vis holds only the autocorrelations, in auto_indices order
    weight_power_scale(
      autocorr_vis,
      weights,
      np.arange(autocorr_vis.shape[-1]),
      autocorrs.instance.index1,
      autocorrs.instance.index2,
      out=array,
      divide=apply_scaling,
    )

    if (cal_params := data_products.instance.calibration_params) is not None:
      cal_solutions = np.empty(array.shape, np.complex64)
      slices = domain.index_exp
      for n, dump in enumerate(range(slices[0].start, slices[0].stop)):
        cal_solutions[n] = calc_correction_per_corrprod(dump, slices[1], cal_params)
      array[:] = apply_weights_correction(array, cal_solutions)

  async def read_chunk(
    domain: ts.IndexDomain, array: np.ndarray, params: ts.VirtualChunkedReadParameters
  ) -> None:
    cws = channel_weights_store.instance
    iws = ts.cast(int_weights_store.instance, cws.dtype)

    # Issue reads to the underlying stores
    (int_weights_future := iws[domain].read(batch=params.batch)).force()
    (channel_weights_future := cws[domain[:-1]].read(batch=params.batch)).force()
    # Only the autocorrelations scale the weights: reading every
    # baseline would copy the whole visibility chunk
    autocorr_vis = vis_store.instance[domain][..., autocorrs.instance.auto_indices]
    (vis_future := autocorr_vis.read(batch=params.batch)).force()

    weights = await int_weights_future
    channel_weights = await channel_weights_future
    vis = await vis_future
    # Scale and calibrate off the event loop thread
    await asyncio.to_thread(assign_chunk, domain, array, weights, channel_weights, vis)

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
