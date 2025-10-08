import numpy as np
import pytest
from xarray import Variable
from xarray.core.indexing import LazilyIndexedArray

from xarray_kat.array import VisibilityArray


@pytest.mark.parametrize(
  "dims, chunks",
  [
    (("time", "frequency", "corrprod"), ((32, 3), (1024, 1024, 1024, 1024), (60,))),
  ],
)
def test_chunked_kat_array(dims, chunks):
  assert len(dims) == len(chunks)
  array = LazilyIndexedArray(VisibilityArray(chunks, np.int32))
  var = Variable(dims, array)

  assert var[[1, 3, 29], slice(0, 3), slice(0, 4)].shape == (
    3,
    3,
    4,
  )
  None
