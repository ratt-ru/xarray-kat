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
  vis_dtype = vis_store.instance.dtype.numpy_dtype
  vis_bytes = vis_dtype.itemsize
  weight_dtype = channel_weights_store.instance.dtype.numpy_dtype
  weight_bytes = weight_dtype.itemsize
  flag_dtype = flag_store.instance.dtype.numpy_dtype
  flag_bytes = flag_dtype.itemsize
  vfw_bytes = vis_bytes + weight_bytes + flag_bytes
  store_shape = (vfw_bytes,) + vis_store.instance.shape
  store_labels = ("bytes",) + vis_store.instance.domain.labels
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

    # Strip out the bytes dimension
    subdomain = domain[1:]

    # Issue data retrieval futures in increasing order of size
    (flag_future := flag_store.instance[subdomain].read()).force()
    (iw_future := iws[subdomain].read()).force()
    (cw_future := cws[subdomain[:-1]].read()).force()
    (vis_future := vis_store.instance[subdomain].read()).force()

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

    def _view(a, dt):
      return a.ravel().view(dt).reshape(data_shape)

    if (vis_view := _view(array[vis_slice], vis_dtype)).base is not array:
      raise RuntimeError("Visibility view doesn't reference base array")

    if (weight_view := _view(array[weight_slice], weight_dtype)).base is not array:
      raise RuntimeError("Weight view doesn't reference base array")

    if (flag_view := _view(array[flag_slice], flag_dtype)).base is not array:
      raise RuntimeError("Flag view doesn't reference base array")

    # Read results into the result array in order of increasing size
    flag_view[:] = flag_future.result()
    data_lost = flag_dtype.type(DATA_LOST)
    zero_flag = flag_dtype.type(0)

    weight_view[:] = iw_future.result()
    weight_view[:] *= cw_future.result()[:, :, None]
    flag_view[:] |= np.where(weight_view == 0, data_lost, zero_flag)

    vis_view[:] = vis_future.result()
    flag_view[:] |= np.where(vis_view == 0j, data_lost, zero_flag)

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
