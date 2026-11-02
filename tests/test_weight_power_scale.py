from itertools import combinations_with_replacement

import numpy as np

from xarray_kat.third_party.vendored.katdal.vis_flags_weights_minimal import (
  corrprod_to_autocorr,
  weight_power_scale,
)


def test_weights_scale_the_same_from_only_the_autocorrelations():
  antennas = [f"m00{a}" for a in range(3)]
  corrprods = [
    (f"{a1}{p1}", f"{a2}{p2}")
    for a1, a2 in combinations_with_replacement(antennas, 2)
    for p1 in "hv"
    for p2 in "hv"
  ]
  autocorrs = corrprod_to_autocorr(corrprods)
  rng = np.random.default_rng(42)
  shape = (2, 5, len(corrprods))
  vis = (rng.random(shape) + 1j * rng.random(shape)).astype(np.complex64)
  weights = rng.random(shape).astype(np.float32)

  expected = weight_power_scale(
    vis, weights, autocorrs.auto_indices, autocorrs.index1, autocorrs.index2
  )
  result = weight_power_scale(
    vis[..., autocorrs.auto_indices],
    weights,
    np.arange(len(autocorrs.auto_indices)),
    autocorrs.index1,
    autocorrs.index2,
  )

  np.testing.assert_array_equal(result, expected)
