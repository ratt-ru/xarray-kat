import numpy as np
import pytest
import xarray
from xarray.namedarray.parallelcompat import (
  get_chunked_array_type,
  guess_chunkmanager,
  list_chunkmanagers,
)

from xarray_kat.meerkat_chunk_manager import MeerkatArray, MeerKatChunkManager


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
  array = MeerkatArray(np.zeros((25, 45), np.float64), ((10, 10, 5), (20, 20, 5)))
  manager = get_chunked_array_type(array)
  assert isinstance(manager, MeerKatChunkManager)


def test_load_chunked_array(register_meerkat_chunkmanager):
  ntime = 10
  nbl = 7 * (7 - 1) // 2
  nfreq = 16
  npol = 4
  tbl = (ntime, nbl)
  all = tbl + (nfreq, npol)
  tblc = (2, 3)
  allc = tblc + (4, 4)
  tbldims = ("time", "baseline_id")
  alldims = tbldims + ("frequency", "polarization")

  ds = xarray.Dataset(
    {
      "UVW": (
        tbldims + ("uvw_label",),
        MeerkatArray(np.zeros(tbl + (3,)), tblc + (3,)),
      ),
      "FLAG": (alldims, MeerkatArray(np.ones(all, np.uint8), allc)),
      "WEIGHT": (alldims, MeerkatArray(np.ones(all, np.float64), allc)),
      "DATA": (alldims, MeerkatArray(np.ones(all, np.complex64), allc)),
    }
  )

  assert isinstance(ds.UVW.data, MeerkatArray)
  assert isinstance(ds.FLAG.data, MeerkatArray)
  assert isinstance(ds.WEIGHT.data, MeerkatArray)
  assert isinstance(ds.DATA.data, MeerkatArray)

  ds.load()

  assert isinstance(ds.UVW.data, np.ndarray)
  assert isinstance(ds.FLAG.data, np.ndarray)
  assert isinstance(ds.WEIGHT.data, np.ndarray)
  assert isinstance(ds.DATA.data, np.ndarray)
