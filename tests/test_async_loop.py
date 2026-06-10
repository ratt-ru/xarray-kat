import asyncio
import gc
import threading
import weakref

import pytest

from xarray_kat.async_loop import (
  AsyncLoopSingleton,
  Singleton,
  _close_state,
  _LoopState,
)


@pytest.fixture(autouse=True)
def reset_singleton():
  """Give every test a fresh AsyncLoopSingleton.

  AsyncLoopSingleton lives in the Singleton metaclass cache, so without this
  one test's close()/restart would leak loop+thread state into the next.
  """

  def _reset():
    instance = Singleton._instances.pop(AsyncLoopSingleton, None)
    if instance is not None:
      instance.close()

  _reset()
  yield
  _reset()


# ---------------------------------------------------------------------------
# Singleton metaclass
# ---------------------------------------------------------------------------


def test_singleton_returns_same_instance():
  assert AsyncLoopSingleton() is AsyncLoopSingleton()
  assert AsyncLoopSingleton().instance is AsyncLoopSingleton().instance


def test_concurrent_construction_yields_single_instance():
  """The double-checked lock in Singleton.__call__ must hand every racing
  thread the same instance (and start exactly one loop thread)."""
  n = 8
  barrier = threading.Barrier(n)
  results: list[AsyncLoopSingleton] = []
  lock = threading.Lock()

  def make():
    barrier.wait()
    inst = AsyncLoopSingleton()
    with lock:
      results.append(inst)

  threads = [threading.Thread(target=make) for _ in range(n)]
  for t in threads:
    t.start()
  for t in threads:
    t.join()

  assert len(results) == n
  assert len({id(r) for r in results}) == 1
  # Exactly one background loop thread was spawned.
  loop_threads = [t for t in threading.enumerate() if t.name == "AsyncLoopThread"]
  assert len(loop_threads) == 1


# ---------------------------------------------------------------------------
# start() / running loop
# ---------------------------------------------------------------------------


def test_instance_is_a_running_loop_on_construction():
  """start() waits on the running event, so .instance is never observed
  before run_forever() is actually driving the loop."""
  singleton = AsyncLoopSingleton()
  assert isinstance(singleton.instance, asyncio.AbstractEventLoop)
  assert singleton.instance.is_running()
  assert singleton._state.running.is_set()


def test_loop_runs_in_dedicated_daemon_thread():
  singleton = AsyncLoopSingleton()
  thread = singleton._state.thread
  assert thread is not None
  assert thread.is_alive()
  assert thread.daemon
  assert thread.name == "AsyncLoopThread"
  assert thread is not threading.current_thread()


def test_start_is_idempotent_while_running():
  """Calling start() again on a live loop is a no-op: same loop, same thread,
  no extra thread spawned."""
  singleton = AsyncLoopSingleton()
  loop, thread = singleton.instance, singleton._state.thread

  singleton.start()

  assert singleton.instance is loop
  assert singleton._state.thread is thread
  loop_threads = [t for t in threading.enumerate() if t.name == "AsyncLoopThread"]
  assert len(loop_threads) == 1


# ---------------------------------------------------------------------------
# coroutine execution
# ---------------------------------------------------------------------------


def test_run_coroutine_returns_result():
  async def add_one(a):
    return a + 1

  loop = AsyncLoopSingleton().instance
  assert asyncio.run_coroutine_threadsafe(add_one(2), loop).result(timeout=5) == 3


def test_coroutine_exception_propagates():
  async def boom():
    raise ValueError("nope")

  loop = AsyncLoopSingleton().instance
  fut = asyncio.run_coroutine_threadsafe(boom(), loop)
  with pytest.raises(ValueError, match="nope"):
    fut.result(timeout=5)


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


def test_close_stops_thread_and_clears_state():
  singleton = AsyncLoopSingleton()
  thread = singleton._state.thread

  singleton.close()

  thread.join(timeout=5)
  assert not thread.is_alive()
  assert singleton.instance is None
  assert singleton._state.thread is None
  assert not singleton._state.running.is_set()


def test_close_is_idempotent():
  singleton = AsyncLoopSingleton()
  singleton.close()
  singleton.close()  # must not raise
  assert singleton.instance is None


def test_close_state_no_op_when_never_started():
  """_close_state on fresh state (loop/thread still None) returns early."""
  state = _LoopState()
  _close_state(state)  # must not raise
  assert state.loop is None
  assert state.thread is None


def test_restart_after_close():
  singleton = AsyncLoopSingleton()
  old_loop = singleton.instance
  singleton.close()
  assert singleton.instance is None

  singleton.start()
  new_loop = singleton.instance
  assert new_loop is not None
  assert new_loop is not old_loop
  assert new_loop.is_running()

  async def answer():
    return 42

  assert asyncio.run_coroutine_threadsafe(answer(), new_loop).result(timeout=5) == 42


def test_close_from_loop_thread_does_not_deadlock():
  """Closing from a coroutine running on the loop thread must not self-join
  deadlock while holding the state lock."""
  singleton = AsyncLoopSingleton()
  loop = singleton.instance

  async def close_from_inside():
    singleton.close()

  fut = asyncio.run_coroutine_threadsafe(close_from_inside(), loop)
  fut.result(timeout=5)  # hangs (and times out) if the self-join deadlock regresses

  assert singleton.instance is None


# ---------------------------------------------------------------------------
# finalizer
# ---------------------------------------------------------------------------


def test_finalizer_collects_instance_and_closes_loop():
  """The finalizer targets a free function over _LoopState, not a bound
  method, so the instance is collectable; collection then tears down the
  loop thread."""
  singleton = AsyncLoopSingleton()
  thread = singleton._state.thread
  ref = weakref.ref(singleton)

  # Drop the only strong references (cache + local) so the instance can die.
  Singleton._instances.pop(AsyncLoopSingleton, None)
  del singleton
  gc.collect()

  # Instance was actually collected (pre-fix regression: leaked via bound method).
  assert ref() is None
  thread.join(timeout=5)
  assert not thread.is_alive()  # finalizer ran and tore down the loop
