import asyncio
import gc

from xarray_kat.async_loop import AsyncLoopSingleton


def test_async_loop_manager_singleton():
  assert AsyncLoopSingleton().instance is AsyncLoopSingleton().instance


def test_sync_coroutine():
  async def f(a):
    return a + 1

  def test():
    assert (
      asyncio.run_coroutine_threadsafe(f(2), AsyncLoopSingleton().instance).result()
      == 3
    )

  test()
  gc.collect()
