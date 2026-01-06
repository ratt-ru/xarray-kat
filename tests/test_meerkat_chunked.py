import numpy as np
import pytest
import xarray
from xarray.namedarray.parallelcompat import (
  get_chunked_array_type,
  guess_chunkmanager,
  list_chunkmanagers,
)

from xarray_kat.meerkat_chunk_manager import MeerKatChunkedArray, MeerKatChunkManager



@pytest.fixture
def register_meerkat_chunkmanager(monkeypatch):
  """
  Mocks the registering of an additional ChunkManagerEntrypoint.

  This preserves the presence of the existing DaskManager, so a test that relies on this and DaskManager both being
  returned from list_chunkmanagers() at once would still work.

  The monkeypatching changes the behavior of list_chunkmanagers when called inside xarray.namedarray.parallelcompat,
  but not when called from this tests file.
  """
  # Should include DaskManager iff dask is available to be imported
  preregistered_chunkmanagers = list_chunkmanagers()

  monkeypatch.setattr(
    "xarray.namedarray.parallelcompat.list_chunkmanagers",
    lambda: {"meerkat": MeerKatChunkManager()} | preregistered_chunkmanagers,
  )
  yield


def test_get_chunkmanager(register_meerkat_chunkmanager):
  assert isinstance(guess_chunkmanager("meerkat"), MeerKatChunkManager)

def test_chunkmanager_chunked_array(register_meerkat_chunkmanager):
  array = MeerKatChunkedArray(np.zeros((25, 45), np.float64), ((10, 10, 5), (20, 20, 5)))
  manager = get_chunked_array_type(array)
  assert isinstance(manager, MeerKatChunkManager)

def test_load_chunked_array(register_meerkat_chunkmanager):
  ds = xarray.Dataset({
    "A": (("x", "y"), MeerKatChunkedArray(np.zeros((10, 10), np.float64), ((5, 5), (5, 5)))),
    "B": (("x", "y"), MeerKatChunkedArray(np.ones((10, 10), np.int32), ((5, 5), (5, 5)))),
  })

  ds.load()

