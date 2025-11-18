import gc
import pickle
from dataclasses import dataclass

import cloudpickle
import dill
import pytest

from xarray_kat.multiton import Multiton


@dataclass
class Data:
  a: float
  b: float


def test_multiton_arg_normalisation():
  """Test that factory keywords are correctly normalised into args"""
  m1 = Multiton(Data, 2.0, b=3.0)
  m2 = Multiton(Data, 2.0, 3.0)
  assert m1.instance is m2.instance


@pytest.mark.parametrize("method", [pickle, cloudpickle, dill])
def test_multiton_pickle(method):
  """Tests multiton pickling with difference pickle implementations"""
  m = Multiton(Data, 2.0, b=3.0)
  datum = {"d": m, "e": {"f": m}}
  udatum = method.loads(method.dumps(datum))
  assert datum["d"].instance is datum["e"]["f"].instance
  assert udatum["d"].instance is udatum["e"]["f"].instance
  assert datum["d"].instance is udatum["d"].instance


def test_multiton_release():
  """Tests that releasing the multiton leads to
  eviction of the instance from the cache"""
  # Two multiton that resolve to the same instance
  m1 = Multiton(Data, 1.0, b=3.0)
  m2 = Multiton(Data, 1.0, 3.0)
  assert m1.instance in Multiton._INSTANCE_CACHE.values()
  assert m1.instance is m2.instance
  assert len(Multiton._INSTANCE_CACHE) == 1
  m2.release()
  gc.collect()
  assert m1.instance in Multiton._INSTANCE_CACHE.values()
  assert len(Multiton._INSTANCE_CACHE) == 1
  m1.release()
  gc.collect()
  assert len(Multiton._INSTANCE_CACHE) == 0
