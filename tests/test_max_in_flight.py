"""MeerkatArray finds open_datatree's max_in_flight once, below xarray's adapters."""

import numpy as np
from xarray.core.indexing import (
  CopyOnWriteArray,
  ImplicitToExplicitIndexingAdapter,
  MemoryCachedArray,
  NumpyIndexingAdapter,
)

from xarray_kat.meerkat_chunk_manager import MeerkatArray


class Backend(NumpyIndexingAdapter):
  """A numpy-backed array carrying max_in_flight, as the backend arrays do"""

  __slots__ = ("max_in_flight",)

  def __init__(self, array, max_in_flight):
    super().__init__(array)
    self.max_in_flight = max_in_flight


def test_max_in_flight_is_found_below_xarray_adapters_and_kept_by_indexing():
  backend = Backend(np.zeros((25, 45)), max_in_flight=7)
  # As xarray hands it over, plus a further adapter it might add
  wrapped = ImplicitToExplicitIndexingAdapter(
    CopyOnWriteArray(MemoryCachedArray(backend))
  )

  array = MeerkatArray(wrapped, (10, 20))

  assert array.max_in_flight == 7
  assert array[2:12, 5:].max_in_flight == 7
  assert array.rechunk((5, 5)).max_in_flight == 7


def test_arrays_not_from_the_backend_have_no_max_in_flight():
  assert MeerkatArray(np.zeros((25, 45)), (10, 20)).max_in_flight is None
