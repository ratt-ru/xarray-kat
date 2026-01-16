import numpy as np
import numpy.typing as npt
import pytest
import tensorstore as ts
import xarray
from numpy.testing import assert_array_equal
from xarray.backends import BackendArray, BackendEntrypoint
from xarray.core.indexing import (
  BasicIndexer,
  ExplicitlyIndexedNDArrayMixin,
  IndexingSupport,
  LazilyIndexedArray,
  OuterIndexer,
  VectorizedIndexer,
  explicit_indexing_adapter,
)
from xarray.namedarray.parallelcompat import (
  get_chunked_array_type,
  guess_chunkmanager,
  list_chunkmanagers,
)

from xarray_kat.meerkat_chunk_manager import MeerkatArray, MeerKatChunkManager


@pytest.fixture(params=[{"shape": (4, 5), "dtype": np.complex64}])
def ramp_data(request):
  shape = request.param["shape"]
  dtype = request.param["dtype"]
  return np.arange(np.prod(shape)).reshape(shape).astype(dtype)


class DummyArray(BackendArray):
  """Wraps a numpy array for testing"""

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
  array = MeerkatArray(np.zeros((25, 45), np.float64), (10, 20))
  manager = get_chunked_array_type(array)
  assert isinstance(manager, MeerKatChunkManager)


NTIME = 10
NBL = 7 * (7 - 1) // 2
NFREQ = 16
NPOL = 4
TBL = (NTIME, NBL)
ALL = TBL + (NFREQ, NPOL)
TBL_CHUNKS = (2, 3)
UVW_CHUNKS = TBL_CHUNKS + (3,)
ALL_CHUNKS = TBL_CHUNKS + (4, 4)
TBL_DIMS = ("time", "baseline_id")
UVW_DIMS = TBL_DIMS + ("uvw_label",)
ALL_DIMS = TBL_DIMS + ("frequency", "polarization")


@pytest.fixture
def small_ds(request):
  def Array(a):
    return LazilyIndexedArray(DummyArray(a))

  ramp = np.arange(np.prod(ALL)).astype(np.float32).reshape(ALL)

  return xarray.Dataset(
    {
      "UVW": (UVW_DIMS, Array(np.zeros(TBL + (3,)))),
      "FLAG": (ALL_DIMS, Array(np.ones(ALL, np.uint8))),
      "WEIGHT": (ALL_DIMS, Array(ramp)),
      "DATA": (ALL_DIMS, Array(ramp + ramp * 1j)),
    }
  )


def test_load_backend_arrays(register_meerkat_chunkmanager, small_ds):
  ds = small_ds
  assert (mgr := guess_chunkmanager("meerkat")) is not None
  shape = tuple(ds.sizes[d] for d in ALL_DIMS)
  chunks = mgr.normalize_chunks(ALL_CHUNKS, shape)
  uvw_chunks = chunks[:2] + ((3,),)

  assert isinstance(ds.UVW.variable._data, LazilyIndexedArray)
  assert isinstance(ds.FLAG.variable._data, LazilyIndexedArray)
  assert isinstance(ds.WEIGHT.variable._data, LazilyIndexedArray)
  assert isinstance(ds.DATA.variable._data, LazilyIndexedArray)

  ds = small_ds.chunk(dict(zip(ALL_DIMS, ALL_CHUNKS)), chunked_array_type="meerkat")

  assert isinstance(uvw := ds.UVW.data, MeerkatArray) and uvw.chunks == uvw_chunks
  assert isinstance(flag := ds.FLAG.data, MeerkatArray) and flag.chunks == chunks
  assert isinstance(weight := ds.WEIGHT.data, MeerkatArray) and weight.chunks == chunks
  assert isinstance(data := ds.DATA.data, MeerkatArray) and data.chunks == chunks

  assert dict(ds.chunks) == dict(zip(ALL_DIMS + ("uvw_label",), chunks + ((3,),)))

  ds.load()

  assert isinstance(ds.UVW.data, np.ndarray)
  assert isinstance(ds.FLAG.data, np.ndarray)
  assert isinstance(ds.WEIGHT.data, np.ndarray)
  assert isinstance(ds.DATA.data, np.ndarray)

  # Expected data has been populated during the load
  ramp = np.arange(np.prod(ds.WEIGHT.shape)).reshape(ds.WEIGHT.shape)
  np.testing.assert_array_equal(ramp, ds.WEIGHT.data)
  np.testing.assert_array_equal(ramp + ramp * 1j, ds.DATA.data)


def test_load_backend_arrays_isel(register_meerkat_chunkmanager, small_ds):
  ds = small_ds
  assert (mgr := guess_chunkmanager("meerkat")) is not None
  shape = tuple(ds.sizes[d] for d in ALL_DIMS)
  chunks = mgr.normalize_chunks(ALL_CHUNKS, shape)
  uvw_chunks = chunks[:2] + ((3,),)

  ramp = np.arange(np.prod(ds.WEIGHT.shape)).reshape(ds.WEIGHT.shape)

  assert isinstance(ds.UVW.variable._data, LazilyIndexedArray)
  assert isinstance(ds.FLAG.variable._data, LazilyIndexedArray)
  assert isinstance(ds.WEIGHT.variable._data, LazilyIndexedArray)
  assert isinstance(ds.DATA.variable._data, LazilyIndexedArray)

  ds = small_ds.chunk(dict(zip(ALL_DIMS, ALL_CHUNKS)), chunked_array_type="meerkat")

  assert isinstance(uvw := ds.UVW.data, MeerkatArray) and uvw.chunks == uvw_chunks
  assert isinstance(flag := ds.FLAG.data, MeerkatArray) and flag.chunks == chunks
  assert isinstance(weight := ds.WEIGHT.data, MeerkatArray) and weight.chunks == chunks
  assert isinstance(data := ds.DATA.data, MeerkatArray) and data.chunks == chunks

  assert dict(ds.chunks) == dict(zip(ALL_DIMS + ("uvw_label",), chunks + ((3,),)))

  sel = {"time": slice(2, 20), "baseline_id": [1, 3, 5, 6, 7], "frequency": slice(4, 8)}
  ds = ds.isel(**sel)
  ds.load()

  assert isinstance(ds.UVW.data, np.ndarray)
  assert isinstance(ds.FLAG.data, np.ndarray)
  assert isinstance(ds.WEIGHT.data, np.ndarray)
  assert isinstance(ds.DATA.data, np.ndarray)

  key = tuple((sel["time"], sel["baseline_id"], sel["frequency"], slice(None)))

  # Expected data has been populated during the load
  # that matches the selection
  assert_array_equal(ramp[key], ds.WEIGHT.data)
  assert_array_equal((ramp + ramp * 1j)[key], ds.DATA.data)


class WrappedTensorStore(ExplicitlyIndexedNDArrayMixin):
  __slots__ = ("array",)

  @property
  def dtype(self) -> npt.DTypeLike:
    return self.array.dtype.numpy_dtype

  def __init__(self, array):
    self.array = array

  def get_duck_array(self):
    return self.array

  async def async_get_duck_array(self):
    return self.array

  def _oindex_get(self, indexer: OuterIndexer):
    return WrappedTensorStore(self.array.oindex[indexer.tuple])

  def _vindex_get(self, indexer: VectorizedIndexer):
    return WrappedTensorStore(self.array.vindex[indexer.tuple])

  def __getitem__(self, indexer):
    return WrappedTensorStore(self.array[indexer.tuple])


class TensorstoreBackendArray(WrappedTensorStore, BackendArray):
  def __init__(self, array):
    super().__init__(array)

  def __getitem__(self, key):
    return explicit_indexing_adapter(
      key, self.shape, IndexingSupport.OUTER, self._getitem
    )

  def _getitem(self, key):
    return WrappedTensorStore(self.array[key])


def test_tensorstore_backend_array(register_meerkat_chunkmanager, ramp_data):
  A = TensorstoreBackendArray(ts.array(ramp_data))
  M = MeerkatArray(A, (2, 3))
  L = LazilyIndexedArray(A)
  V = L[BasicIndexer((slice(1, 3), slice(2, 4)))]
  assert isinstance(V.get_duck_array(), ts.TensorStore)

  ds = xarray.Dataset({"M": (("x", "y"), M)})
  key = {"x": slice(1, 3), "y": [1, 4]}
  ds = ds.isel(**key)
  ds.load()

  assert_array_equal(ds.M.data, ramp_data[key["x"], key["y"]])


class DummyBackendEntryPoint(BackendEntrypoint):
  description = "Dummy Testing Backend"
  supports_groups = True

  def open_datatree(self, filename_or_obj, *, drop_variables=None):
    return xarray.DataTree.from_dict(
      self.open_groups_as_dict(filename_or_obj, drop_variables=drop_variables)
    )

  def open_groups_as_dict(self, filename_or_obj, *, drop_variables=None):
    def Array(a):
      return TensorstoreBackendArray(ts.array(a))

    ramp = np.arange(np.prod(ALL)).astype(np.float32).reshape(ALL)

    ds = xarray.Dataset(
      {
        "UVW": (UVW_DIMS, Array(np.zeros(TBL + (3,)))),
        "FLAG": (ALL_DIMS, Array(np.ones(ALL, np.uint8))),
        "WEIGHT": (ALL_DIMS, Array(ramp)),
        "VISIBILITY": (ALL_DIMS, Array(ramp + ramp * 1j)),
      }
    )

    return {"a": ds, "b": ds}


@pytest.fixture
def register_dummy_engine(monkeypatch):
  """Mock registration of a ChunkManagerEntrypoint"""
  from xarray.backends.plugins import list_engines

  engines = list_engines()

  monkeypatch.setattr(
    "xarray.backends.plugins.list_engines",
    lambda: engines | {"test-backend": DummyBackendEntryPoint()},
  )
  yield


def test_tensorstore_arrays_open_datatree(
  register_meerkat_chunkmanager, register_dummy_engine
):
  """Tests that opening tensorstore backend arrays with the MeerKatArray
  Chunked Array type works"""
  dt = xarray.open_datatree(
    "don't-care", engine="test-backend", chunked_array_type="meerkat", chunks={}
  )
  shape = dt["a"].FLAG.shape
  ramp = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
  key = {"time": slice(2, 4), "baseline_id": [1, 3, 4], "frequency": slice(4, 12)}
  assert isinstance(dt["a"].WEIGHT.data, MeerkatArray)
  assert isinstance(dt["a"].VISIBILITY.data, MeerkatArray)
  assert isinstance(dt["b"].WEIGHT.data, MeerkatArray)
  assert isinstance(dt["b"].VISIBILITY.data, MeerkatArray)

  dt = dt.isel(**key)
  dt.load()

  sel_ramp = ramp[tuple(key.values())]

  assert isinstance(dt["a"].WEIGHT.data, np.ndarray)
  assert isinstance(dt["a"].VISIBILITY.data, np.ndarray)
  assert isinstance(dt["b"].WEIGHT.data, np.ndarray)
  assert isinstance(dt["b"].VISIBILITY.data, np.ndarray)

  assert_array_equal(dt["a"].WEIGHT, sel_ramp)
  assert_array_equal(dt["a"].VISIBILITY, sel_ramp + sel_ramp * 1j)
  assert_array_equal(dt["b"].WEIGHT, sel_ramp)
  assert_array_equal(dt["b"].VISIBILITY, sel_ramp + sel_ramp * 1j)


def test_tensorstore_wrapped_array(register_meerkat_chunkmanager, ramp_data):
  A = WrappedTensorStore(ts.array(ramp_data))
  assert A.dtype == np.complex64
  assert A.shape == (4, 5)
  key = (slice(1, 3), slice(2, 4))
  assert_array_equal(A[BasicIndexer(key)], ramp_data[key])
