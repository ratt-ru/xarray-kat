import inspect
from types import FrameType
from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt
import pytest
import tensorstore as ts
import xarray
from numpy.testing import assert_array_equal
from xarray.backends import BackendArray, BackendEntrypoint
from xarray.backends.api import open_datatree, open_groups
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
from xarray_kat.utils import normalize_chunks


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


def test_chunk_backend_arrays(register_meerkat_chunkmanager, small_ds):
  """Tests rechunking into chunked MeerKatArrays"""
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
  """Tests rechunking into chunked MeerKatArray following by an isel"""
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


class DelayedTensorStore(ExplicitlyIndexedNDArrayMixin):
  __slots__ = ("array",)

  array: ts.TensorStore

  def __init__(self, array):
    self.array = array

  @property
  def dtype(self) -> npt.DTypeLike:
    return self.array.dtype.numpy_dtype

  def get_duck_array(self):
    return self.array

  async def async_get_duck_array(self):
    return self.array

  def _oindex_get(self, indexer: OuterIndexer):
    return DelayedTensorStore(self.array.oindex[indexer.tuple])

  def _vindex_get(self, indexer: VectorizedIndexer):
    return DelayedTensorStore(self.array.vindex[indexer.tuple])

  def __getitem__(self, key):
    return DelayedTensorStore(self.array[key.tuple])


class DelayedTensorStoreBackendArray(DelayedTensorStore, BackendArray):
  def __init__(self, array):
    super().__init__(array)


class ImmediateTensorStore(ExplicitlyIndexedNDArrayMixin):
  __slots__ = ("array",)

  array: ts.TensorStore

  def __init__(self, array):
    self.array = array

  @property
  def dtype(self) -> npt.DTypeLike:
    return self.array.dtype.numpy_dtype

  def get_duck_array(self):
    return self.array.read().result()

  async def async_get_duck_array(self):
    return self.array.read().result()

  def _oindex_get(self, indexer):
    return self.array.oindex[indexer.tuple].read().result()

  def _vindex_get(self, indexer):
    return self.array.vindex[indexer.tuple].read().result()

  def __getitem__(self, key):
    return self.array[key.tuple].read().result()


class ImmediateTensorBackendArray(ImmediateTensorStore, BackendArray):
  def __init__(self, array):
    super().__init__(array)


def test_tensorstore_backend_array(register_meerkat_chunkmanager, ramp_data):
  A = DelayedTensorStoreBackendArray(ts.array(ramp_data))
  M = MeerkatArray(A, (2, 3))
  L = LazilyIndexedArray(A)
  V = L[BasicIndexer((slice(1, 3), slice(2, 4)))]
  assert isinstance(V.get_duck_array(), DelayedTensorStore)

  ds = xarray.Dataset({"M": (("x", "y"), M)})
  key = {"x": slice(1, 3), "y": [1, 4]}
  ds = ds.isel(**key)
  ds.load()

  assert_array_equal(ds.M.data, ramp_data[key["x"], key["y"]])


class DummyBackendEntryPoint(BackendEntrypoint):
  description = "Dummy Testing Backend"
  supports_groups = True

  @staticmethod
  def infer_api_chunking(
    frame: FrameType | None, depth: int = 10
  ) -> Tuple[Dict[str, int] | None, str | None]:
    chunks = None
    array_type = None

    while frame and depth > 0 and chunks is None and array_type is None:
      if frame.f_code in {open_groups.__code__, open_datatree.__code__}:
        chunks = chunks or frame.f_locals.get("chunks")
        array_type = array_type or frame.f_locals.get("chunked_array_type")

      depth -= 1
      frame = frame.f_back

    return chunks, array_type

  def open_datatree(self, filename_or_obj, *, drop_variables=None):
    return xarray.DataTree.from_dict(
      self.open_groups_as_dict(filename_or_obj, drop_variables=drop_variables)
    )

  def open_groups_as_dict(self, filename_or_obj, *, drop_variables=None):
    ramp = np.arange(np.prod(ALL)).astype(np.float32).reshape(ALL)
    chunks, _ = self.infer_api_chunking(inspect.currentframe().f_back)

    def Array(a):
      if chunks is not None:
        return DelayedTensorStoreBackendArray(ts.array(a))
      return LazilyIndexedArray(ImmediateTensorBackendArray(ts.array(a)))

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


def test_tensorstore_arrays_open_datatree(register_dummy_engine):
  """Test opening the datatree without chunking"""
  dt = xarray.open_datatree("dont-care", engine="test-backend")

  def recurse_isinstance(data, types):
    if isinstance(data, types):
      return True

    return recurse_isinstance(getattr(data, "array", None), types)

  assert recurse_isinstance(dt["a"].WEIGHT.variable._data, ImmediateTensorBackendArray)
  assert recurse_isinstance(
    dt["a"].VISIBILITY.variable._data, ImmediateTensorBackendArray
  )
  assert recurse_isinstance(dt["b"].WEIGHT.variable._data, ImmediateTensorBackendArray)
  assert recurse_isinstance(
    dt["b"].VISIBILITY.variable._data, ImmediateTensorBackendArray
  )

  dt.load()

  assert recurse_isinstance(dt["a"].WEIGHT.variable._data, np.ndarray)
  assert recurse_isinstance(dt["a"].VISIBILITY.variable._data, np.ndarray)
  assert recurse_isinstance(dt["b"].WEIGHT.variable._data, np.ndarray)
  assert recurse_isinstance(dt["b"].VISIBILITY.variable._data, np.ndarray)


@pytest.mark.parametrize(
  "chunks",
  [{}, {"time": 1, "frequency": 4}, {"time": 2, "baseline_id": 3, "frequency": 8}],
)
def test_tensorstore_arrays_open_datatree_meerkat_chunks(
  register_meerkat_chunkmanager, register_dummy_engine, chunks
):
  """Tests that opening tensorstore backend arrays with the MeerKatArray
  Chunked Array type works"""
  dt = xarray.open_datatree(
    "don't-care", engine="test-backend", chunked_array_type="meerkat", chunks=chunks
  )

  flag = dt["a"].FLAG
  expected_chunks = normalize_chunks(
    tuple(chunks.get(d, c) for (d, c) in zip(flag.dims, flag.chunks)), flag.shape
  )

  shape = dt["a"].FLAG.shape
  ramp = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
  key = {"time": slice(2, 4), "baseline_id": [1, 3, 4], "frequency": slice(4, 12)}
  assert isinstance(dt["a"].WEIGHT.data, MeerkatArray)
  assert isinstance(dt["a"].VISIBILITY.data, MeerkatArray)
  assert dt["a"].WEIGHT.data.chunks == expected_chunks
  assert dt["a"].VISIBILITY.data.chunks == expected_chunks
  assert isinstance(dt["b"].WEIGHT.data, MeerkatArray)
  assert isinstance(dt["b"].VISIBILITY.data, MeerkatArray)
  assert dt["b"].WEIGHT.data.chunks == expected_chunks
  assert dt["b"].VISIBILITY.data.chunks == expected_chunks

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


@pytest.mark.parametrize(
  "chunks",
  [{}, {"time": 1, "frequency": 4}, {"time": 2, "baseline_id": 3, "frequency": 8}],
)
def test_tensorstore_arrays_open_datatree_dask_chunks(
  register_meerkat_chunkmanager, register_dummy_engine, chunks
):
  """Tests that opening tensorstore backend arrays with the MeerKatArray
  Chunked Array type works"""
  da = pytest.importorskip("dask.array")
  dt = xarray.open_datatree(
    "s3://remote-url-to-mollify-mtime",
    engine="test-backend",
    chunked_array_type="dask",
    chunks=chunks,
  )

  flag = dt["a"].FLAG
  expected_chunks = normalize_chunks(
    tuple(chunks.get(d, c) for (d, c) in zip(flag.dims, flag.chunks)), flag.shape
  )

  shape = dt["a"].FLAG.shape
  ramp = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
  key = {"time": slice(2, 4), "baseline_id": [1, 3, 4], "frequency": slice(4, 12)}
  assert isinstance(dt["a"].WEIGHT.data, da.Array)
  assert isinstance(dt["a"].VISIBILITY.data, da.Array)
  assert dt["a"].WEIGHT.data.chunks == expected_chunks
  assert dt["a"].VISIBILITY.data.chunks == expected_chunks
  assert isinstance(dt["b"].WEIGHT.data, da.Array)
  assert isinstance(dt["b"].VISIBILITY.data, da.Array)
  assert dt["b"].WEIGHT.data.chunks == expected_chunks
  assert dt["b"].VISIBILITY.data.chunks == expected_chunks

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


def test_tensorstore_delayed_array(register_meerkat_chunkmanager, ramp_data):
  A = DelayedTensorStore(ts.array(ramp_data))
  assert A.dtype == np.complex64
  assert A.shape == (4, 5)
  key = (slice(1, 3), slice(2, 4))
  assert_array_equal(A[BasicIndexer(key)], ramp_data[key])
