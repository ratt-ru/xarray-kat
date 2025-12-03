from typing import Any, Dict, Tuple

import numpy as np
import tensorstore as ts

from xarray_kat.multiton import Multiton


def scaled_weight_store(
  int_weights_store: Multiton[ts.TensorStore],
  channel_weights_store: Multiton[ts.TensorStore],
  chunk_info: Dict[str, Any],
  dim_labels: Tuple[str, ...],
  context: ts.Context,
  van_vleck_vis_store: Multiton[ts.TensorStore] | None = None,
):
  vis_chunks = chunk_info["correlator_data"]["chunks"]
  vis_shape = tuple(sum(dc) for dc in vis_chunks)
  if not all(all(dc[0] == c for c in dc[1:-1]) for dc in vis_chunks):
    raise ValueError(f"Visibility {vis_chunks} are not homogenous")

  def read_chunk(
    domain: ts.IndexDomain, array: np.ndarray, params: ts.VirtualChunkedReadParameters
  ) -> ts.KvStore.TimestampedStorageGeneration:
    int_weights = int_weights_store.instance.read(domain)
    channel_weights = channel_weights_store.instance.read(domain)
    array[...] = int_weights.result().value * channel_weights.result().value

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
