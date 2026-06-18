"""Tests that opening a mock archive with applycal yields a valid CorrectionParams.

A mock katdal archive is built with synthetic calibration solutions and opened
through the public ``xarray.open_datatree`` path. The ``TelstateDataProducts``
instance created along the way is recovered from ``Multiton._INSTANCE_CACHE``
and its ``calibration_params`` (a :class:`CorrectionParams`) is validated.
"""

from collections import defaultdict

import numpy as np
import pytest
import xarray
from pytest_httpserver import HTTPServer
from rarg_python_patterns import Multiton
from xarray.core.indexing import LazilyIndexedArray

from tests.conftest import (
  SyntheticObservation,
  setup_mock_archive_server,
)
from xarray_kat.array import CalibrationBackendArray
from xarray_kat.calibration import calc_correction_per_antenna
from xarray_kat.katdal_types import TelstateDataProducts, TelstateDataSource
from xarray_kat.third_party.vendored.katdal.applycal_minimal import (
  CorrectionParams,
  calc_correction_per_corrprod,
)
from xarray_kat.utils import ANTENNA_RECEPTOR_REGEX

CBID = "1234567890"
NTIME = 25
NFREQ = 16
NANTS = 4


def _build_archive(httpserver: HTTPServer, tmp_path, *, applycal, with_cal):
  """Build a mock archive, and serve it"""
  obs = SyntheticObservation(CBID, ntime=NTIME, nfreq=NFREQ, nants=NANTS)
  obs.add_scan(range(0, 8), "track", "PKS1934")
  obs.add_scan(range(8, 20), "scan", "3C286")
  obs.add_scan(range(20, NTIME), "track", "PKS1934")
  if with_cal:
    obs.add_calibration_solutions()
  obs.save_to_directory(tmp_path)

  setup_mock_archive_server(httpserver, tmp_path, CBID, require_auth=False)
  return obs, f"{httpserver.url_for('/')}{CBID}/{CBID}_sdp_l0.full.rdb"


def _open_archive(httpserver: HTTPServer, tmp_path, *, applycal, with_cal):
  """Build a mock archive, serve it, open it and return the TelstateDataProducts."""
  obs, rdb_url = _build_archive(
    httpserver, tmp_path, applycal=applycal, with_cal=with_cal
  )

  # Open via the public entrypoint; this constructs (and caches) the
  # TelstateDataProducts as a side effect.
  xarray.open_datatree(rdb_url, engine="xarray-kat", applycal=applycal)

  # Recover the TelstateDataProducts instance from the multiton cache. The
  # clear_multitons autouse fixture wipes the cache around every test, so the
  # only instance present is the one created by the open_datatree call above.
  products = [
    entry[0]
    for entry in Multiton._INSTANCE_CACHE.values()
    if isinstance(entry[0], TelstateDataProducts)
  ]
  assert len(products) == 1, f"expected one TelstateDataProducts, found {len(products)}"
  return obs, products[0]


def test_cal_params(httpserver: HTTPServer, tmp_path):
  """Opening a mock archive with applycal='all' builds a valid CorrectionParams."""
  obs, tdp = _open_archive(httpserver, tmp_path, applycal="all", with_cal=True)
  params = tdp.calibration_params

  assert isinstance(params, CorrectionParams)

  # Cal products were discovered and recorded (l1.{G,K,B} in some order).
  applied = tdp._dataset.applycal_products
  assert applied, "no cal products were applied"
  assert all(p.startswith("l1.") for p in applied)

  # inputs: one entry per antenna-pol (h/v) feed, sorted.
  inputs = params.inputs
  assert inputs == sorted(inputs)
  assert len(inputs) == NANTS * 2
  assert all(inp[:-1] in obs.ant_names and inp[-1] in ("h", "v") for inp in inputs)

  # input1_index / input2_index: integer indices into `inputs`, one per
  # correlation product, all within range.
  n_corrprods = len(obs.bls_ordering)
  for idx in (params.input1_index, params.input2_index):
    idx = np.asarray(idx)
    assert idx.shape == (n_corrprods,)
    assert np.issubdtype(idx.dtype, np.integer)
    assert idx.min() >= 0 and idx.max() < len(inputs)

  # corrections / channel_maps are keyed by the applied products; each
  # corrections entry has one (dump-indexable) sequence per input, and each
  # channel map is callable.
  assert set(params.corrections) == set(applied)
  assert set(params.channel_maps) == set(applied)
  for product in applied:
    per_input = params.corrections[product]
    assert len(per_input) == len(inputs)
    first = per_input[0]
    assert first[0] is not None  # indexable by dump
    assert callable(params.channel_maps[product])

  # Functional smoke test: a per-corrprod correction for the first dump over
  # the full band has the expected shape and dtype.
  gains = calc_correction_per_corrprod(0, slice(0, NFREQ), params)
  assert gains.shape == (NFREQ, n_corrprods)
  assert gains.dtype == np.complex64


def test_calc_correction_per_antenna(httpserver: HTTPServer, tmp_path):
  """The per-antenna prototype reproduces the per-corrprod autocorrelation gains.

  ``calc_correction_per_antenna`` only forms the four pol products of an
  antenna's own h/v feeds, so it corresponds exactly to the autocorrelation
  baselines of ``calc_correction_per_corrprod``; cross-antenna baselines are
  not representable from a single antenna's feeds.
  """
  _, tdp = _open_archive(httpserver, tmp_path, applycal="all", with_cal=True)
  params = tdp.calibration_params

  channels = slice(0, NFREQ)
  per_antenna = calc_correction_per_antenna(0, channels, params)
  per_corrprod = calc_correction_per_corrprod(0, channels, params)

  assert per_antenna.shape == (NANTS, NFREQ, 4)
  assert per_antenna.dtype == np.complex64

  # Group inputs by antenna into their (h, v) feed indices. Sorted antenna
  # order matches the antenna axis of per_antenna (which iterates the sorted
  # params.inputs).
  antenna_receptors: defaultdict[str, dict[str, int]] = defaultdict(dict)
  for idx, feed in enumerate(params.inputs):
    if (m := ANTENNA_RECEPTOR_REGEX.match(feed)) is None:
      raise ValueError(f"Invalid antenna receptor string {antenna_receptors}")

    antenna = f"{m.group('prefix')}{m.group('number')}"
    receptor = m.group("receptor")
    antenna_receptors[antenna][receptor] = idx
  antennas = sorted(antenna_receptors)
  assert len(antennas) == NANTS

  # per_antenna[a, :, p] = g_in[ant_r1] * conj(g_in[ant_r2]) for the antenna's
  # own feeds, with pols ordered (hh, hv, vh, vv). Locate the matching
  # autocorrelation corrprod (same input pair) and compare.
  input1 = np.asarray(params.input1_index)
  input2 = np.asarray(params.input2_index)
  for a, ant in enumerate(antennas):
    hi, vi = antenna_receptors[ant]["h"], antenna_receptors[ant]["v"]
    pol_inputs = [(hi, hi), (hi, vi), (vi, hi), (vi, vi)]  # hh, hv, vh, vv
    for p, (in1, in2) in enumerate(pol_inputs):
      (matches,) = np.where((input1 == in1) & (input2 == in2))
      assert len(matches) == 1, f"expected one corrprod for {ant} pol {p}"
      np.testing.assert_allclose(
        per_antenna[a, :, p], per_corrprod[:, matches[0]], rtol=1e-6, atol=1e-6
      )


@pytest.mark.parametrize("applycal", ["all"])
@pytest.mark.parametrize("stream_name", [""])
def test_calibration_array_backend(httpserver, tmp_path, applycal, stream_name):
  _, rdb_url = _build_archive(httpserver, tmp_path, applycal=applycal, with_cal=True)

  datasource = Multiton(
    TelstateDataSource.from_url,
    rdb_url,
    chunk_store=None,
    capture_block_id=CBID,
  )

  data_products = Multiton(TelstateDataProducts, datasource, applycal=applycal)
  array = LazilyIndexedArray(CalibrationBackendArray(data_products))

  dataset = xarray.Dataset(
    {"DATA": (("time", "antenna_name", "frequency", "polarization"), array)}
  )

  subds = dataset.isel({"time": slice(0, 10), "frequency": [1, 4, 8]})
  subds.load()

  assert {"time": 10, "frequency": 3}.items() <= dict(subds.sizes).items()
