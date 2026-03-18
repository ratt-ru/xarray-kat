"""Tests that numpy and numba applycal correction variants produce identical results."""

import numpy as np
import pytest

from xarray_kat.third_party.vendored.katdal.applycal_minimal import (
  POSTPROC,
  nb_apply_flags_correction,
  nb_apply_vis_correction,
  nb_apply_weights_correction,
  np_apply_flags_correction,
  np_apply_vis_correction,
  np_apply_weights_correction,
)

numba = pytest.importorskip("numba")

RNG = np.random.default_rng(42)
SHAPE = (4, 8, 10)  # (ntime, nfreq, ncorrprod)


def make_vis(shape=SHAPE):
  re = RNG.standard_normal(shape).astype(np.float32)
  im = RNG.standard_normal(shape).astype(np.float32)
  return (re + 1j * im).astype(np.complex64)


def make_weights(shape=SHAPE):
  return RNG.uniform(0, 1, shape).astype(np.float32)


def make_flags(shape=SHAPE):
  return RNG.integers(0, 2, shape, dtype=np.uint8)


def make_correction(shape=SHAPE, nan_fraction=0.1):
  """Complex correction array with some NaN entries to simulate flagged solutions."""
  re = RNG.standard_normal(shape).astype(np.float32)
  im = RNG.standard_normal(shape).astype(np.float32)
  correction = (re + 1j * im).astype(np.complex64)
  nan_mask = RNG.random(shape) < nan_fraction
  correction[nan_mask] = np.nan + 1j * np.nan
  return correction


class TestApplyVisCorrection:
  def test_no_nans(self):
    data = make_vis()
    correction = make_correction(nan_fraction=0.0)
    np.testing.assert_allclose(
      np_apply_vis_correction(data, correction),
      nb_apply_vis_correction(data, correction),
      rtol=1e-6,
    )

  def test_some_nans(self):
    data = make_vis()
    correction = make_correction(nan_fraction=0.2)
    np.testing.assert_allclose(
      np_apply_vis_correction(data, correction),
      nb_apply_vis_correction(data, correction),
      rtol=1e-6,
    )

  def test_all_nans(self):
    data = make_vis()
    correction = np.full(SHAPE, np.nan + 1j * np.nan, dtype=np.complex64)
    np.testing.assert_array_equal(
      np_apply_vis_correction(data, correction),
      nb_apply_vis_correction(data, correction),
    )

  def test_nan_entries_unchanged(self):
    """Where correction is NaN the original data value must be preserved."""
    data = make_vis()
    correction = make_correction(nan_fraction=0.3)
    result = np_apply_vis_correction(data, correction)
    nan_mask = np.isnan(correction)
    np.testing.assert_array_equal(result[nan_mask], data[nan_mask])


class TestApplyWeightsCorrection:
  def test_no_nans(self):
    data = make_weights()
    correction = make_correction(nan_fraction=0.0)
    np.testing.assert_array_equal(
      np_apply_weights_correction(data, correction),
      nb_apply_weights_correction(data, correction),
    )

  def test_some_nans(self):
    data = make_weights()
    correction = make_correction(nan_fraction=0.2)
    np.testing.assert_array_equal(
      np_apply_weights_correction(data, correction),
      nb_apply_weights_correction(data, correction),
    )

  def test_all_nans(self):
    data = make_weights()
    correction = np.full(SHAPE, np.nan + 1j * np.nan, dtype=np.complex64)
    np.testing.assert_array_equal(
      np_apply_weights_correction(data, correction),
      nb_apply_weights_correction(data, correction),
    )

  def test_nan_entries_zeroed(self):
    """Where correction is NaN the weight must be zeroed."""
    data = make_weights()
    correction = make_correction(nan_fraction=0.3)
    result = np_apply_weights_correction(data, correction)
    nan_mask = np.isnan(correction.real**2 + correction.imag**2)
    assert np.all(result[nan_mask] == 0)


class TestApplyFlagsCorrection:
  def test_no_nans(self):
    data = make_flags()
    correction = make_correction(nan_fraction=0.0)
    np.testing.assert_array_equal(
      np_apply_flags_correction(data, correction),
      nb_apply_flags_correction(data, correction),
    )

  def test_some_nans(self):
    data = make_flags()
    correction = make_correction(nan_fraction=0.2)
    np.testing.assert_array_equal(
      np_apply_flags_correction(data, correction),
      nb_apply_flags_correction(data, correction),
    )

  def test_all_nans(self):
    data = make_flags()
    correction = np.full(SHAPE, np.nan + 1j * np.nan, dtype=np.complex64)
    np.testing.assert_array_equal(
      np_apply_flags_correction(data, correction),
      nb_apply_flags_correction(data, correction),
    )

  def test_nan_entries_set_postproc(self):
    """Where correction is NaN the POSTPROC flag bit must be set."""
    data = np.zeros(SHAPE, dtype=np.uint8)
    correction = make_correction(nan_fraction=0.3)
    result = np_apply_flags_correction(data, correction)
    nan_mask = np.isnan(correction)
    assert np.all(result[nan_mask] & POSTPROC)
    assert np.all(result[~nan_mask] == 0)
