import weakref
from itertools import pairwise, product
from threading import Lock
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt
from xarray.backends.common import BackendArray


class LifeCycleManager:
  def __init__(self):
    # Keeps the object alive as long as it's in this set
    self._keep_alive = {}
    # Tracks which fields have been touched
    self._touch_log = {}
    # Protect access to the manager
    self._lock = Lock()

  def register(self, obj: Any, required_fields: Iterable[str]):
    obj_id = id(obj)
    with self._lock:
      self._keep_alive[obj_id] = obj
      self._touch_log[obj_id] = set(required_fields)
      # print(f"Registered {obj_id}: Persistence active.")

  def touch(self, obj: Any, field: str):
    obj_id = id(obj)
    with self._lock:
      if obj_id in self._touch_log:
        self._touch_log[obj_id].discard(field)

        # If no fields are left, drop the strong reference
        if not self._touch_log[obj_id]:
          # print(f"All fields accessed for {obj_id}. Releasing to GC.")
          del self._keep_alive[obj_id]
          del self._touch_log[obj_id]


# The Global Manager
_MANAGER = LifeCycleManager()


class VisFlagWeight:
  __slots__ = ("_vis", "_flag", "_weight", "_lock", "__weakref__")

  _lock: Lock
  _vis: npt.NDArray | None
  _flag: npt.NDArray | None
  _weight: npt.NDArray | None

  def __init__(self):
    _MANAGER.register(self, ["vis", "flags", "weights"])
    self._lock = Lock()
    self._vis = None
    self._weight = None
    self._flag = None

  def _retrieve(self):
    with self._lock:
      pass

  @property
  def vis(self) -> npt.NDArray:
    _MANAGER.touch(self, "vis")
    return self._vis

  @property
  def weight(self) -> npt.NDArray:
    _MANAGER.touch(self, "weights")
    return self._weight

  @property
  def flag(self) -> npt.NDArray:
    _MANAGER.touch(self, "flags")
    return self._flag


vfw: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
vfw_lock: Lock = Lock()


class VFWArray(BackendArray):
  def __init__(self, shape, dtype, chunks):
    self.shape = shape
    self.dtype = dtype
    from xarray_kat.utils import normalize_chunks

    self.chunks = normalize_chunks(chunks, shape)
    ranges = [np.concatenate(([0], np.cumsum(c))).tolist() for c in self.chunks]

    for dim_coord_pairs in product(*(pairwise(r) for r in ranges)):
      key = tuple(dim_coord_pairs)
      with vfw_lock:
        vfw.setdefault(key, VisFlagWeight())


if __name__ == "__main__":
  print(len(vfw))

  def func():
    A = VFWArray((100, 16, 4), np.float64, ((11,), (5,), (3,)))  # noqa: F841
    B = VFWArray((100, 16, 4), np.bool, ((11,), (5,), (3,)))  # noqa: F841
    print(len(vfw))
    D = next(iter(vfw.values()))
    D.vis
    D.flag
    D.weight

  func()
  print(len(vfw))
  import gc

  gc.collect()
  print(len(vfw))

  def clear_fn():
    for v in vfw.values():
      v.vis
      v.flag
      v.weight

  clear_fn()
  print(len(vfw))
