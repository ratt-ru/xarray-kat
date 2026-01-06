from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import tensorstore as ts

from xarray_kat.multiton import Multiton
from xarray_kat.third_party.vendored.katdal.applycal_minimal import (
  apply_flags_correction,
  apply_vis_correction,
  apply_weights_correction,
  calc_correction_per_corrprod,
)
from xarray_kat.third_party.vendored.katdal.flags import DATA_LOST
from xarray_kat.third_party.vendored.katdal.vis_flags_weights_minimal import (
  weight_power_scale,
)

if TYPE_CHECKING:
  from xarray_kat.katdal_types import AutoCorrelationIndices, TelstateDataProducts
  from xarray_kat.types import VanVleckLiteralType


def interleaved_vfw_store(
  vis_store: Multiton[ts.TensorStore],
  int_weights_store: Multiton[ts.TensorStore],
  channel_weights_store: Multiton[ts.TensorStore],
  flag_store: Multiton[ts.TensorStore],
  data_products: Multiton[TelstateDataProducts],
  autocorrs: Multiton[AutoCorrelationIndices],
  apply_scaling: bool,
  van_vleck: VanVleckLiteralType,
  context: ts.Context,
):
  """Produces a tensorstore containing the interleaved values produced by
  combining visibilities, weights and flags

  Raw MeerKAT visibilities, weights and flags (vfw) need to be combined to produce
  final vfw by applying transformations such as Van Vleck transforms,
  weight scaling and flagging missing data.

  This function produces a tensorstore which applies the above transforms
  and returns the vfw concatenated with one another as raw bytes

  Returns:
    A tensorstore
  """
  vis_dtype = vis_store.instance.dtype.numpy_dtype()
  vis_bytes = vis_dtype.itemsize
  weight_dtype = channel_weights_store.instance.dtype.numpy_dtype()
  weight_bytes = weight_dtype.itemsize
  flag_dtype = flag_store.instance.dtype.numpy_dtype()
  flag_bytes = flag_dtype.itemsize
  vfw_bytes = vis_bytes + weight_bytes + flag_bytes
  store_shape = (vfw_bytes,) + vis_store.instance.shape
  store_labels = ("bytes",) + vis_store.instance.dim_labels
  store_chunks = (vfw_bytes,) + vis_store.instance.chunk_layout.read_chunk.shape

  def read_chunk(
    domain: ts.IndexDomain, array: np.ndarray, params: ts.VirtualChunkedReadParameters
  ) -> ts.KvStore.TimestampedStorageGeneration:
    cws = channel_weights_store.instance
    iws = ts.cast(int_weights_store.instance, cws.dtype)

    # Sanity check store shapes
    if not (
      (vis_store.instance.shape == iws.shape == flag_store.instance.shape)
      and cws.shape == iws.shape[:-1]
    ):
      raise ValueError(
        f"Store shape mismatch "
        f"visibilities: {vis_store.instance.shape} "
        f"int weights: {iws.shape} "
        f"channel weights: {cws.shape} "
        f"flags: {flag_store.instance.shape}"
      )

    # Issue data retrieval futures in increasing order of size
    (flag_future := flag_store.instance[domain].read()).force()
    (iw_future := iws[domain].read()).force()
    (cw_future := cws[domain[:-1]].read()).force()
    (vis_future := vis_store.instance[domain].read()).force()

    # Possibly prepare calibration solutions
    # while I/O operations are in flight
    cal_solutions: None | npt.NDArray = None
    data_shape = array.shape[1:]

    if (cal_params := data_products.instance.calibration_params) is not None:
      slices = domain.index_exp
      cal_solutions = np.empty(data_shape, dtype=vis_dtype)

      for n, dump in enumerate(range(slices[0].start, slices[0].stop)):
        cal_solutions[n] = calc_correction_per_corrprod(dump, slices[1], cal_params)

    # Establish higher level views over the raw byte output array
    vis_slice = slice(0, pos := vis_bytes)
    weight_slice = slice(pos, pos := pos + weight_bytes)
    flag_slice = slice(pos, pos := pos + flag_bytes)

    vis_view = array[vis_slice].ravel().view(vis_dtype).reshape(data_shape)
    weight_view = array[weight_slice].ravel().view(weight_dtype).reshape(data_shape)
    flag_view = array[flag_slice].ravel().view(flag_dtype).reshape(data_shape)

    assert vis_view.base is array, "View doesn't reference base array"
    assert weight_view.base is array, "View doesn't reference base array"
    assert flag_view.base is array, "View doesn't reference base array"

    # Read results into the result array in order of increasing size
    if (flags := flag_future.result()).state != "value":
      raise RuntimeError(f"Reading flags failed {flags.result().state}")

    flag_view[:] = flags

    if (weights := iw_future.result()).state != "value":
      raise RuntimeError(f"Reading integer weights failed {iw_future.result().state}")

    weight_view[:] = weights

    if (cweights := cw_future.result()).state != "value":
      raise RuntimeError(f"Reading channel weights failed {cweights.result().state}")

    weight_view[:] *= cweights[..., None]
    flag_view[:] |= np.where(weight_view == 0, DATA_LOST, 0)

    if (vis := vis_future.result()).state != "value":
      raise RuntimeError(f"Visibility future failed {vis_future.result().state}")

    vis_view[:] = vis
    flag_view[:] |= np.where(vis_view == 0j, DATA_LOST, 0)

    # Apply weight scaling
    weight_power_scale(
      vis_view,
      weight_view,
      autocorrs.instance.auto_indices,
      autocorrs.instance.index1,
      autocorrs.instance.index2,
      out=weight_view,
      divide=apply_scaling,
    )

    # Apply any calibration solutions
    if cal_solutions is not None:
      apply_weights_correction(weight_view, cal_solutions)
      apply_vis_correction(vis_view, cal_solutions)
      apply_flags_correction(flag_view, cal_solutions)

  return ts.virtual_chunked(
    read_function=read_chunk,
    rank=len(store_shape),
    domain=ts.IndexDomain(
      [ts.Dim(size=s, label=ll) for s, ll in zip(store_shape, store_labels)]
    ),
    dtype=np.uint8,
    chunk_layout=ts.ChunkLayout(chunk_shape=store_chunks),
    context=context,
  )
