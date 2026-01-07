import numpy as np
import pytest
import xarray
from xarray.backends import BackendArray
from xarray.core.indexing import (
  IndexingSupport,
  LazilyIndexedArray,
  explicit_indexing_adapter,
)
from xarray.namedarray.parallelcompat import (
  get_chunked_array_type,
  guess_chunkmanager,
  list_chunkmanagers,
)

from xarray_kat.meerkat_chunk_manager import MeerkatArray, MeerKatChunkManager


class DummyArray(BackendArray):
  def __init__(self, data):
    self.data = data

  @property
  def shape(self):
    return self.data.shape

  @property
  def dtype(self):
    return self.data.dtype

  def __getitem__(self, key):
    return explicit_indexing_adapter(
      key, self.shape, IndexingSupport.OUTER, self._getitem
    )

  def _getitem(self, key):
    return self.data[key]


@pytest.fixture
def register_meerkat_chunkmanager(monkeypatch):
  """Mock registration of a ChunkManagerEntrypoint"""
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


NTIME = 10
NBL = 7 * (7 - 1) // 2
NFREQ = 16
NPOL = 4
TBL = (NTIME, NBL)
ALL = TBL + (NFREQ, NPOL)
TBLC = (2, 3)
ALLC = TBLC + (4, 4)
TBLDIMS = ("time", "baseline_id")
ALLDIMS = TBLDIMS + ("frequency", "polarization")


@pytest.fixture
def small_meerkat_ds(request):
  A = lambda a: LazilyIndexedArray(DummyArray(a))

  return xarray.Dataset(
    {
      "UVW": (TBLDIMS + ("uvw_label",), A(np.zeros(TBL + (3,)))),
      "FLAG": (ALLDIMS, A(np.ones(ALL, np.uint8))),
      "WEIGHT": (ALLDIMS, A(np.ones(ALL, np.float64))),
      "DATA": (ALLDIMS, A(np.ones(ALL, np.complex64))),
    }
  )


def test_load_backend_arrays(register_meerkat_chunkmanager, small_meerkat_ds):
  ds = small_meerkat_ds
  assert (mgr := guess_chunkmanager("meerkat")) is not None
  shape = (ds.sizes[d] for d in ("time", "baseline_id", "frequency", "polarization"))
  chunks = mgr.normalize_chunks(ALLC, shape)
  uvw_chunks = chunks[:2] + ((3,),)

  assert isinstance(ds.UVW.variable._data, LazilyIndexedArray)
  assert isinstance(ds.FLAG.variable._data, LazilyIndexedArray)
  assert isinstance(ds.WEIGHT.variable._data, LazilyIndexedArray)
  assert isinstance(ds.DATA.variable._data, LazilyIndexedArray)

  ds = small_meerkat_ds.chunk(
    {
      "time": ALLC[0],
      "baseline_id": ALLC[1],
      "frequency": ALLC[2],
      "polarization": ALLC[3],
    },
    chunked_array_type="meerkat",
  )

  assert isinstance(uvw := ds.UVW.data, MeerkatArray) and uvw.chunks == uvw_chunks
  assert isinstance(flag := ds.FLAG.data, MeerkatArray) and flag.chunks == chunks
  assert isinstance(weight := ds.WEIGHT.data, MeerkatArray) and weight.chunks == chunks
  assert isinstance(data := ds.DATA.data, MeerkatArray) and data.chunks == chunks

  ds.load()

  assert isinstance(ds.UVW.data, np.ndarray)
  assert isinstance(ds.FLAG.data, np.ndarray)
  assert isinstance(ds.WEIGHT.data, np.ndarray)
  assert isinstance(ds.DATA.data, np.ndarray)
