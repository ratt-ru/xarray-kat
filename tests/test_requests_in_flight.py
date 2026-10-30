"""Loading keeps as many reads in flight as tensorstore allows.

The chunk manager pipelines reads, issuing those of many chunks before waiting
on any. A pipeline of fixed depth caps the requests in flight below
tensorstore's HTTP request concurrency, and waiting on the oldest read first
lets one slow request stall every refill behind it.
"""

import functools
import http.server
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import xarray

#: Every chunk request takes at least this long, so that latency rather than
#: the client's issue rate decides how many requests are in flight
DELAY = 0.1
SLOW_CHUNK = "correlator_data/00002_00000_00000.npy"
SLOW_DELAY = 3.0


class Counter:
  def __init__(self):
    self.lock = threading.Lock()
    self.now = self.peak = 0
    self.started = 0
    self.started_during_slow: int | None = None


class DelayedHandler(http.server.SimpleHTTPRequestHandler):
  counter: Counter

  def log_message(self, format, *args):
    pass

  def do_GET(self):
    if not self.path.endswith(".npy"):
      return super().do_GET()
    c = self.counter
    with c.lock:
      c.now += 1
      c.started += 1
      c.peak = max(c.peak, c.now)
      started = c.started
    slow = self.path.endswith(SLOW_CHUNK)
    time.sleep(SLOW_DELAY if slow else DELAY)
    try:
      super().do_GET()
    finally:
      with c.lock:
        c.now -= 1
        if slow:
          c.started_during_slow = c.started - started


class Server(http.server.ThreadingHTTPServer):
  # Python's default listen backlog of 5 would throttle new connections
  request_queue_size = 1024


@pytest.fixture
def delayed_archive(synthetic_observation, tmp_path):
  """An observation of ~1000 chunk objects, served by a threaded server that
  delays each request"""
  obs = synthetic_observation(ntime=32, nfreq=128, nants=4)
  obs.add_scan(range(0, obs.ntime), "track", "MockTarget")
  obs.save_to_directory(tmp_path)
  counter = Counter()
  handler = type("Handler", (DelayedHandler,), {"counter": counter})
  server = Server(("127.0.0.1", 0), functools.partial(handler, directory=tmp_path))
  threading.Thread(target=server.serve_forever, daemon=True).start()
  rdb = f"{obs.capture_block_id}_sdp_l0.full.rdb"
  yield f"http://127.0.0.1:{server.server_port}/{rdb}", counter
  server.shutdown()


LOAD = """
import sys, xarray
dt = xarray.open_datatree(
  sys.argv[1], engine="xarray-kat", chunked_array_type="xarray-kat", chunks={}
)
for node in dt.subtree:
  if "VISIBILITY" in node.ds.data_vars:
    node.ds[["VISIBILITY", "WEIGHT", "FLAG"]].load()
"""


def test_requests_in_flight_follow_the_http_request_concurrency(delayed_archive):
  url, counter = delayed_archive
  # In a fresh process, so that tensorstore reads the concurrency when it starts
  env = {**os.environ, "TENSORSTORE_HTTP_REQUEST_CONCURRENCY": "256"}

  subprocess.run([sys.executable, "-c", LOAD, url], env=env, check=True, timeout=300)

  # A fixed pipeline of 100 chunk reads (0.0.7) keeps at most ~130 requests in
  # flight here, where each chunk's reads fetch four objects
  assert counter.peak > 180


def open_scan(url, ntime=None):
  dt = xarray.open_datatree(
    url, engine="xarray-kat", chunked_array_type="xarray-kat", chunks={}
  )
  [node] = [node for node in dt.subtree if "VISIBILITY" in node.ds.data_vars]
  return node.ds[["VISIBILITY", "WEIGHT", "FLAG"]].isel(time=slice(0, ntime))


def test_max_in_flight_bounds_the_chunk_reads_in_flight(delayed_archive):
  url, counter = delayed_archive
  expected = open_scan(url, ntime=8).load()
  counter.peak = 0

  limited = open_scan(url, ntime=8).load(max_in_flight=3)

  # One chunk's VISIBILITY, WEIGHT and FLAG reads fetch at most four base
  # objects (visibilities, flags, weights and channel weights), and the next
  # chunk's are issued as the last of these completes; unbounded, 32 are
  assert counter.peak <= 8
  for name in ("VISIBILITY", "WEIGHT", "FLAG"):
    np.testing.assert_array_equal(limited[name].values, expected[name].values)


def test_a_slow_read_does_not_stall_the_reads_behind_it(delayed_archive):
  url, counter = delayed_archive

  open_scan(url).load()

  # Waiting on the oldest read first, only the pipeline's depth (tens of requests)
  # could start while the slow request was outstanding, of ~1000
  assert counter.started_during_slow is not None
  assert counter.started_during_slow > 300
