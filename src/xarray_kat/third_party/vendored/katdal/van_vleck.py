################################################################################
# Copyright (c) 2012,2020,2021, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Routines for performing quantisation (Van Vleck) correction."""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

try:
  import numba
except ImportError:
  numba = None


@dataclass
class LookupTables:
  quantised: npt.NDArray
  true: npt.NDArray


if numba:

  @numba.vectorize(["f8(f8, f8)"], nopython=True, cache=True)
  def norm0_cdf(x, scale):
    """Fast zero-mean (loc=0) implementation of :meth:`scipy.stats.norm.cdf`."""
    return 0.5 * (math.erf(np.sqrt(0.5) * x / scale) + 1.0)
else:
  # ERF approximation constants
  ERF_P = 0.3275911
  ERF_A1 = 0.254829592
  ERF_A2 = -0.284496736
  ERF_A3 = 1.421413741
  ERF_A4 = -1.453152027
  ERF_A5 = 1.061405429

  def approx_erf(x):
    """Approxation of erf from Abramowitz and Stegan formula 7.1.26

    Error is less than 1.5 * 10^-7 for all inputs

    - https://www.amazon.com/dp/0486612724
    - https://stackoverflow.com/a/457805
    """
    x_abs = np.abs(x)
    t = 1.0 / (1.0 + ERF_P * x_abs)
    y = 1.0 - (
      ((((ERF_A5 * t + ERF_A4) * t) + ERF_A3) * t + ERF_A2) * t + ERF_A1
    ) * t * np.exp(-x_abs * x_abs)
    return np.sign(x) * y

  def norm0_cdf(x, scale):
    """Fast zero-mean (loc=0) implementation of :meth:`scipy.stats.norm.cdf`."""
    return 0.5 * (approx_erf(np.sqrt(0.5) * x / scale) + 1.0)


def _quant_norm0_pmf(levels, var=1.0):
  """Probability mass function of quantised zero-mean normal variable."""
  edges = np.r_[-np.inf, levels[:-1] + np.diff(levels) / 2.0, np.inf]
  return np.diff(norm0_cdf(edges, np.sqrt(var)))


def _squared_quant_norm0_mean(levels, var=1.0):
  """Mean of squared quantised zero-mean normal variable (same shape as `var`)."""
  levels = np.asarray(levels)
  # Allow var and levels to be broadcast against each other, with levels as last dimension
  var = np.asarray(var)[..., np.newaxis]
  pmf = _quant_norm0_pmf(levels, var)
  return pmf.dot(levels * levels)


def autocorr_lookup_table(levels, size=4000):
  """Lookup table that corrects complex autocorrelation quantisation effects.

  This maps the variance of a quantised complex voltage signal to the variance
  of the unquantised signal under the assumption that the signal is proper
  (circularly-symmetric) complex normally distributed.

  Parameters
  ----------
  levels : sequence of float
      Quantisation levels for real and imaginary components of voltage signal
  size : int, optional
      Size of lookup table

  Returns
  -------
  quantised_autocorr_table, true_autocorr_table : array of float, shape (`size`,)
      Lookup table associating quantised autocorrelations and unquantised
      autocorrelations (i.e. power/variance of complex signals)
  """
  # Terminology:
  # x = Proper complex normal voltage signal (zero-mean)
  # rxx = Power (variance) *per* real/imag component of unquantised / true x
  # sxx = Power (variance) *per* real/imag component of quantised x
  abs_levels = np.abs(levels)
  sxx_min_nonzero = abs_levels[abs_levels > 0].min() ** 2
  sxx_max = abs_levels.max() ** 2
  # Sweep across range of true power values, placing more table entries at tricky lower end
  rxx_grid = np.r_[
    np.logspace(-2.4, 0, size // 2, endpoint=False),
    np.logspace(0, np.log10(sxx_max / sxx_min_nonzero) + 8, size - 2 - size // 2),
  ]
  # Shift the table to place inflection point at minimum non-zero sxx
  rxx_grid *= sxx_min_nonzero
  # Map true power to expected quantised power
  sxx_mean = _squared_quant_norm0_mean(levels, rxx_grid)
  # Extend quantised power values to its maximum range
  sxx_table = np.r_[0.0, sxx_mean, sxx_max]
  # Replace asymptotic with linear decay at bottom end, and clip unbounded growth at top end
  rxx_table = np.r_[0.0, rxx_grid, rxx_grid[-1]]
  # The factor 2 converts power per real/imag component to power/variance of complex signal
  return LookupTables(2.0 * sxx_table, 2.0 * rxx_table)
