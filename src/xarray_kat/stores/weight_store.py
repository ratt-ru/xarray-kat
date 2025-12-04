from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
import tensorstore as ts

from xarray_kat.third_party.vendored.katdal.vis_flags_weights_minimal import (
  weight_power_scale,
)

if TYPE_CHECKING:
  from xarray_kat.katdal_types import AutoCorrelationIndices, TelstateDataSource
  from xarray_kat.multiton import Multiton


def scaled_weight_store(
  int_weights_store: Multiton[ts.TensorStore],
  channel_weights_store: Multiton[ts.TensorStore],
  vis_store: Multiton[ts.TensorStore],
  autocorrs: Multiton[AutoCorrelationIndices],
  datasource: Multiton[TelstateDataSource],
  dim_labels: Tuple[str, ...],
  context: ts.Context,
):
  telstate = datasource.instance.telstate
  chunk_info = telstate["chunk_info"]
  vis_chunks = chunk_info["correlator_data"]["chunks"]
  vis_shape = tuple(sum(dc) for dc in vis_chunks)
  if not all(all(dc[0] == c for c in dc[1:-1]) for dc in vis_chunks):
    raise ValueError(f"Visibility {vis_chunks} are not homogenous")

  store_weights_are_scaled = not telstate.get("need_weights_power_scale", False)
  divide = not store_weights_are_scaled

  def read_chunk(
    domain: ts.IndexDomain, array: np.ndarray, params: ts.VirtualChunkedReadParameters
  ) -> ts.KvStore.TimestampedStorageGeneration:
    cws = channel_weights_store.instance
    iws = ts.cast(int_weights_store.instance, cws.dtype)

    int_weights_rr = iws[domain].read()
    channel_weights_rr = cws[domain[:-1]].read()
    vis_rr = vis_store.instance[domain].read()
    weights = int_weights_rr.result() * channel_weights_rr.result()[..., None]

    array[...] = weight_power_scale(
      vis_rr.result(),
      weights,
      autocorrs.instance.auto_indices,
      autocorrs.instance.index1,
      autocorrs.instance.index2,
      divide=divide,
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
