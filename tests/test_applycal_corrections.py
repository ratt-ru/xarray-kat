"""Tests that numpy and numba applycal correction variants produce identical results."""

import numpy as np
import pytest

from xarray_kat.third_party.vendored.katdal.applycal_minimal import (
  nb_apply_flags_correction,
  nb_apply_vis_correction,
  nb_apply_weights_correction,
  np_apply_flags_correction,
  np_apply_vis_correction,
  np_apply_weights_correction,
)
from xarray_kat.third_party.vendored.katdal.flags import (
  NAMES as FLAG_NAMES,
)
from xarray_kat.third_party.vendored.katdal.flags import (
  POSTPROC,
)

numba = pytest.importorskip("numba")

SHAPE = (4, 8, 10)  # (ntime, nfreq, ncorrprod)


def _make_correction(nan_fraction):
  rng = np.random.default_rng(0)
  re = rng.standard_normal(SHAPE).astype(np.float32)
  im = rng.standard_normal(SHAPE).astype(np.float32)
  correction = (re + 1j * im).astype(np.complex64)
  correction[rng.random(SHAPE) < nan_fraction] = np.nan + 1j * np.nan
  return correction


@pytest.fixture
def vis():
  rng = np.random.default_rng(1)
  re = rng.standard_normal(SHAPE).astype(np.float32)
  im = rng.standard_normal(SHAPE).astype(np.float32)
  return (re + 1j * im).astype(np.complex64)


@pytest.fixture
def weights():
  return np.random.default_rng(2).uniform(0, 1, SHAPE).astype(np.float32)


@pytest.fixture
def flags():
  return np.random.default_rng(3).integers(0, len(FLAG_NAMES), SHAPE, dtype=np.uint8)


@pytest.fixture
def correction_no_nans():
  return _make_correction(nan_fraction=0.0)


@pytest.fixture
def correction_some_nans():
  return _make_correction(nan_fraction=0.2)


@pytest.fixture
def correction_sparse_nans():
  return _make_correction(nan_fraction=0.3)


@pytest.fixture
def correction_all_nans():
  return np.full(SHAPE, np.nan + 1j * np.nan, dtype=np.complex64)


class TestApplyVisCorrection:
  def test_no_nans(self, vis, correction_no_nans):
    np.testing.assert_allclose(
      np_apply_vis_correction(vis, correction_no_nans),
      nb_apply_vis_correction(vis, correction_no_nans),
      rtol=1e-6,
    )

  def test_some_nans(self, vis, correction_some_nans):
    np.testing.assert_allclose(
      np_apply_vis_correction(vis, correction_some_nans),
      nb_apply_vis_correction(vis, correction_some_nans),
      rtol=1e-6,
    )

  def test_all_nans(self, vis, correction_all_nans):
    np.testing.assert_array_equal(
      np_apply_vis_correction(vis, correction_all_nans),
      nb_apply_vis_correction(vis, correction_all_nans),
    )

  def test_nan_entries_unchanged(self, vis, correction_sparse_nans):
    """Where correction is NaN the original data value must be preserved."""
    result = np_apply_vis_correction(vis, correction_sparse_nans)
    nan_mask = np.isnan(correction_sparse_nans)
    np.testing.assert_array_equal(result[nan_mask], vis[nan_mask])


class TestApplyWeightsCorrection:
  def test_no_nans(self, weights, correction_no_nans):
    np.testing.assert_array_equal(
      np_apply_weights_correction(weights, correction_no_nans),
      nb_apply_weights_correction(weights, correction_no_nans),
    )

  def test_some_nans(self, weights, correction_some_nans):
    np.testing.assert_array_equal(
      np_apply_weights_correction(weights, correction_some_nans),
      nb_apply_weights_correction(weights, correction_some_nans),
    )

  def test_all_nans(self, weights, correction_all_nans):
    np.testing.assert_array_equal(
      np_apply_weights_correction(weights, correction_all_nans),
      nb_apply_weights_correction(weights, correction_all_nans),
    )

  def test_nan_entries_zeroed(self, weights, correction_sparse_nans):
    """Where correction is NaN the weight must be zeroed."""
    result = np_apply_weights_correction(weights, correction_sparse_nans)
    nan_mask = np.isnan(correction_sparse_nans.real**2 + correction_sparse_nans.imag**2)
    assert np.all(result[nan_mask] == 0)


class TestApplyFlagsCorrection:
  def test_no_nans(self, flags, correction_no_nans):
    np.testing.assert_array_equal(
      np_apply_flags_correction(flags, correction_no_nans),
      nb_apply_flags_correction(flags, correction_no_nans),
    )

  def test_some_nans(self, flags, correction_some_nans):
    np.testing.assert_array_equal(
      np_apply_flags_correction(flags, correction_some_nans),
      nb_apply_flags_correction(flags, correction_some_nans),
    )

  def test_all_nans(self, flags, correction_all_nans):
    np.testing.assert_array_equal(
      np_apply_flags_correction(flags, correction_all_nans),
      nb_apply_flags_correction(flags, correction_all_nans),
    )

  def test_nan_entries_set_postproc(self, correction_sparse_nans):
    """Where correction is NaN the POSTPROC flag bit must be set."""
    data = np.zeros(SHAPE, dtype=np.uint8)
    result = np_apply_flags_correction(data, correction_sparse_nans)
    nan_mask = np.isnan(correction_sparse_nans)
    assert np.all(result[nan_mask] & POSTPROC)
    assert np.all(result[~nan_mask] == 0)
