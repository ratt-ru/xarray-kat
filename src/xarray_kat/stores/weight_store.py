from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np
import tensorstore as ts

from xarray_kat.third_party.vendored.katdal.vis_flags_weights_minimal import (
  weight_power_scale,
)

if TYPE_CHECKING:
  from xarray_kat.katdal_types import AutoCorrelationIndices
  from xarray_kat.multiton import Multiton


def scaled_weight_store(
  int_weights_store: Multiton[ts.TensorStore],
  channel_weights_store: Multiton[ts.TensorStore],
  vis_store: Multiton[ts.TensorStore],
  autocorrs: Multiton[AutoCorrelationIndices],
  chunk_info: Dict[str, Any],
  apply_scaling: bool,
  dim_labels: Tuple[str, ...],
  context: ts.Context,
):
  """Combines weights and channel_weights and scales the weights
  by the visibility auto-correlations

  Args:
    int_weights_store: uint8 weights TensorStore of shape ``(time, frequency, corrprod)``.
    channel_weights_store: float32 weights TensorStore of shape ``(time, frequency)``.
    vis_store: complex visibility TensorStore of shape ``(time, frequency, corrprod)``.
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
  vis_chunks = chunk_info["correlator_data"]["chunks"]
  vis_shape = tuple(sum(dc) for dc in vis_chunks)
  if not all(all(dc[0] == c for c in dc[1:-1]) for dc in vis_chunks):
    raise ValueError(f"Visibility {vis_chunks} are not homogenous")

  def read_chunk(
    domain: ts.IndexDomain, array: np.ndarray, params: ts.VirtualChunkedReadParameters
  ) -> ts.KvStore.TimestampedStorageGeneration:
    cws = channel_weights_store.instance
    iws = ts.cast(int_weights_store.instance, cws.dtype)

    # Issue reads to the underlying stores
    int_weights_rr = iws[domain].read()
    channel_weights_rr = cws[domain[:-1]].read()
    vis_rr = vis_store.instance[domain].read()
    int_weights_rr.force()
    channel_weights_rr.force()
    vis_rr.force()
    weights = int_weights_rr.result()
    weights *= channel_weights_rr.result()[..., None]

    array[...] = weight_power_scale(
      vis_rr.result(),
      weights,
      autocorrs.instance.auto_indices,
      autocorrs.instance.index1,
      autocorrs.instance.index2,
      divide=apply_scaling,
    )

  return ts.virtual_chunked(
    read_function=read_chunk,
    rank=len(vis_shape),
    domain=ts.IndexDomain(
      [ts.Dim(size=s, label=ll) for s, ll in zip(vis_shape, dim_labels)]
    ),
    dtype=np.dtype(chunk_info["weights_channel"]["dtype"]),
    chunk_layout=ts.ChunkLayout(chunk_shape=[c[0] for c in vis_chunks]),
    context=context,
  )
