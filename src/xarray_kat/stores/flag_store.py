from __future__ import annotations

from typing import TYPE_CHECKING

import tensorstore as ts

from xarray_kat.third_party.vendored.katdal.applycal_minimal import (
  apply_flags_correction,
)

if TYPE_CHECKING:
  import numpy as np

  from xarray_kat.multiton import Multiton


def final_flag_store(
  base_flag_store: Multiton[ts.TensorStore],
  cal_solutions_store: Multiton[ts.TensorStore] | None,
  context: ts.Context,
):
  def read_chunk(
    domain: ts.IndexDomain, array: np.ndarray, params: ts.VirtualChunkedReadParameters
  ) -> ts.KvStore.TimestampedStorageGeneration:
    flags_rr = base_flag_store.instance[domain].read()
    flags_rr.force()
    if cal_solutions_store is not None:
      cal_rr = cal_solutions_store.instance[domain].read()
      cal_rr.force()

    array[:] = flags_rr.result()

    if cal_solutions_store is not None:
      apply_flags_correction(array, cal_rr.result())

  return ts.virtual_chunked(
    read_function=read_chunk,
    rank=base_flag_store.instance.rank,
    domain=base_flag_store.instance.domain,
    dtype=base_flag_store.instance.dtype,
    chunk_layout=base_flag_store.instance.chunk_layout,
    context=context,
  )
