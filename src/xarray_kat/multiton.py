from __future__ import annotations

from threading import Lock
from typing import Any, Callable, ClassVar, Dict, Generic, Tuple, TypeVar
from weakref import WeakValueDictionary

from xarray_kat.utils.serialisation import FrozenKey, normalise_args

T = TypeVar("T")


class Multiton(Generic[T]):
  """Implementation of the Multiton pattern.

  See https://en.wikipedia.org/wiki/Multiton_pattern for an overview.

  Multiton's are hashable, equality-comparable and pickleable as long
  as the supplied arguments also support these properties.

  .. code-block:: python

    # Factory function creating a resource
    def open_connection(url: str, timeout: float = 1.0) -> Connection:
      ...

    # Create a multiton representing a resource
    resource = Multiton(open_connection, "https://www.python.org", timeout=10.0)

    # The resource is only created when the instance attribute is accessed
    response = resource.instance.request("GET", "/foo/bar.html")
  """

  # Class variables
  _INSTANCE_CACHE: ClassVar[WeakValueDictionary[FrozenKey, Any]] = WeakValueDictionary()
  _INSTANCE_LOCK: Lock = Lock()

  __slots__ = ("_factory", "_args", "_kw", "_key", "_instance")

  # Instance variables
  _factory: Callable[..., T]
  _args: Tuple[Any, ...]
  _kw: Dict[str, Any]
  _key: FrozenKey
  _instance: T | None

  def __init__(self, factory: Callable[..., T], *args, **kw):
    """Create a Multiton with the factory function and arguments
    necessary for creating the underlying object instance.

    Arguments:
      factory: A factory function
      args: Arguments passed to the factory function
      kw: Keyword arguments passed to the factory function
    """
    self._factory = factory
    self._args, self._kw = normalise_args(factory, args, kw)
    self._key = FrozenKey(factory, *self._args, **self._kw)
    self._instance = None

  @staticmethod
  def from_reduce_args(factory: Callable[..., T], args, kw) -> Multiton[T]:
    """Helper method for reconstructing a Multiton from arg and kw objects"""
    return Multiton[T](factory, *args, **kw)

  def __reduce__(self) -> Tuple[Callable, Tuple[Any, ...]]:
    return (Multiton.from_reduce_args, (self._factory, self._args, self._kw))

  def __hash__(self) -> int:
    return hash(self._key)

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, Multiton):
      return NotImplemented
    return self._key == other._key

  @property
  def instance(self) -> T:
    """Returns the instance defined by this Multiton"""
    if self._instance is not None:
      return self._instance

    with self._INSTANCE_LOCK:
      if self._instance is not None:
        return self._instance

      try:
        self._instance = self._INSTANCE_CACHE[self._key]
      except KeyError:
        self._instance = self._factory(*self._args, **self._kw)
        self._INSTANCE_CACHE[self._key] = self._instance

      return self._instance

  def release(self) -> None:
    """Release the instance held by this Multiton."""
    self._instance = None
