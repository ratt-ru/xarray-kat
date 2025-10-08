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


class AsyncLoopSingleton(metaclass=Singleton):
  _loop: asyncio.AbstractEventLoop | None
  _thread: threading.Thread | None
  _lock: threading.Lock
  _running: threading.Event

  def __init__(self):
    self._loop = None
    self._thread = None
    self._lock = threading.Lock()
    self._running = threading.Event()
    weakref.finalize(self, self.close)
    self.start()

  @property
  def instance(self):
    return self._loop

  def start(self) -> None:
    with self._lock:
      if self._thread and self._thread.is_alive():
        return

      self._loop = asyncio.new_event_loop()
      self._thread = threading.Thread(
        target=_run_loop_in_thread,
        args=(self._loop, self._running),
        daemon=True,
        name="AsyncLoopThread",
      )
      self._thread.start()

  def close(self) -> None:
    with self._lock:
      if not self._thread or not self._loop:
        return

      if self._loop.is_running():
        self._loop.call_soon_threadsafe(self._loop.stop)

      self._thread.join()
      self._thread = None
      self._loop = None
