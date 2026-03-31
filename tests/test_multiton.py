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


class DataFactory:
  @classmethod
  def create(cls, a: float, b: float = 3.0) -> Data:
    return Data(a, b)


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


def test_multiton_reentrant():
  """Tests RLock works"""

  def inner_factory(m: Multiton[Data]) -> Data:
    return m.instance

  def outer_factory(a: int, m: Multiton[Data]) -> Data:
    return inner_factory(m)

  om = Multiton(outer_factory, 2, Multiton(Data, 1.0, b=2.0))
  assert om.instance.a == 1.0
  assert om.instance.b == 2.0


def test_multiton_classmethod_normalisation():
  """Test that normalise_args correctly handles classmethods (skips bound cls)"""
  m1 = Multiton(DataFactory.create, 2.0, b=3.0)
  m2 = Multiton(DataFactory.create, 2.0, 3.0)
  assert m1.instance is m2.instance


def test_multiton_classmethod_pickle():
  """Test that a Multiton with a classmethod factory round-trips through pickle.

  This exercises the normalise_args fix: after unpickling, the reconstructed
  Multiton must produce the same key and call the factory with the same args
  as the original, without cls being counted as a positional slot.

  stdlib pickle serialises classmethods by reference, so the deserialized key
  is identical and we get cache sharing (m.instance is m2.instance).
  cloudpickle/dill may reconstruct a new bound-method object whose hash differs,
  so we only assert value equality for those.
  """
  m = Multiton(DataFactory.create, 2.0, b=3.0)
  m2 = pickle.loads(pickle.dumps(m))
  # stdlib pickle: key must be identical → cache hit → same object
  assert m.instance is m2.instance
  assert m2.instance == Data(2.0, 3.0)

  for method in [cloudpickle, dill]:
    m3 = method.loads(method.dumps(m))
    # cloudpickle/dill may not preserve bound-method identity across streams,
    # but the factory must still be callable with the correct arguments.
    assert m3.instance == Data(2.0, 3.0)


def test_multiton_classmethod_default_not_duplicated():
  """Test that a defaulted kwarg isn't appended twice when unpickling.

  Before the fix, the chunk_store-style bug would cause normalise_args to
  append the default value again on reconstruction, producing a key mismatch
  and a TypeError when calling the factory with too many positional args.
  """
  # Pass chunk_store-equivalent arg explicitly (overriding its default)
  m = Multiton(DataFactory.create, 2.0, b=5.0)
  m2 = pickle.loads(pickle.dumps(m))
  # Keys must be equal so the cache is shared
  assert m == m2
  # Factory must be callable without TypeError
  assert m2.instance == Data(2.0, 5.0)


def test_multiton_engaged():
  """Tests that a multiton instance is automatically engaged,
  if it has already been engaged by another Multiton"""
  m1 = Multiton(Data, 1.0, b=3.0)
  assert m1._instance is None
  assert m1.instance == Data(1.0, 3.0)

  m2 = Multiton(Data, 1.0, b=3.0)
  assert m2._instance == Data(1.0, 3.0)
