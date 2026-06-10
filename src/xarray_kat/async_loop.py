from __future__ import annotations

import asyncio
import logging
import threading
import weakref
from typing import Any, Dict

log = logging.getLogger(__name__)


class Singleton(type):
  _instances: Dict[type, Any] = {}
  _instance_lock = threading.Lock()

  def __call__(cls, *args, **kwargs):
    if cls not in cls._instances:
      with cls._instance_lock:
        if cls not in cls._instances:
          cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

    return cls._instances[cls]


def _run_loop_in_thread(
  loop: asyncio.AbstractEventLoop, running: threading.Event
) -> None:
  asyncio.set_event_loop(loop)
  running.set()

  try:
    loop.run_forever()
  finally:
    log.debug("Loop stops")
    running.clear()
    log.debug("Shutting down async generators")
    loop.run_until_complete(loop.shutdown_asyncgens())
    log.debug("Shutting down default executors")
    loop.run_until_complete(loop.shutdown_default_executor())

    log.debug("Closing the loop")
    loop.close()
    log.debug("Done")


class _LoopState:
  """Mutable loop/thread state shared between the singleton and its finalizer.

  Kept separate from ``AsyncLoopSingleton`` so the ``weakref.finalize`` callback
  can reference this state instead of a bound method of the instance (which would
  create a strong self-reference and prevent the instance from being collected).
  """

  __slots__ = ("loop", "thread", "lock", "running")

  def __init__(self) -> None:
    self.loop: asyncio.AbstractEventLoop | None = None
    self.thread: threading.Thread | None = None
    self.lock = threading.Lock()
    self.running = threading.Event()


def _close_state(state: _LoopState) -> None:
  with state.lock:
    thread, loop = state.thread, state.loop
    if not thread or not loop:
      return

    if loop.is_running():
      loop.call_soon_threadsafe(loop.stop)

    # A thread cannot join itself; calling close() from a coroutine running on
    # the loop thread would otherwise deadlock (while holding state.lock).
    if thread is not threading.current_thread():
      thread.join()

    state.thread = None
    state.loop = None


class AsyncLoopSingleton(metaclass=Singleton):
  _state: _LoopState

  def __init__(self):
    self._state = _LoopState()
    weakref.finalize(self, _close_state, self._state)
    self.start()

  @property
  def instance(self):
    return self._state.loop

  def start(self) -> None:
    state = self._state
    with state.lock:
      if state.thread and state.thread.is_alive():
        return

      state.running.clear()
      state.loop = asyncio.new_event_loop()
      state.thread = threading.Thread(
        target=_run_loop_in_thread,
        args=(state.loop, state.running),
        daemon=True,
        name="AsyncLoopThread",
      )
      state.thread.start()
      # Don't return until run_forever() is actually driving the loop, so
      # callers of .instance never see a not-yet-running loop.
      state.running.wait()

  def close(self) -> None:
    _close_state(self._state)
