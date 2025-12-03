################################################################################
# Copyright (c) 2017,2020-2022,2024, National Research Foundation (SARAO)
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

"""A (lazy) container for the triplet of visibilities, flags and weights."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

try:
  import numba
except ImportError:
  numba = None


@dataclass
class AutoCorrelationIndices:
  auto_indices: npt.NDArray
  index1: npt.NDArray
  index2: npt.NDArray


def _narrow(array):
  """Reduce an integer array to the narrowest type that can hold it.

  It is specialised for unsigned types. It will not alter the dtype
  if the array contains negative values.

  If the type is not changed, a view is returned rather than a copy.
  """
  if array.dtype.kind not in ["u", "i"]:
    raise ValueError("Array is not integral")
  if not array.size:
    dtype = np.uint8
  else:
    low = np.min(array)
    high = np.max(array)
    if low < 0:
      dtype = array.dtype
    elif high <= 0xFF:
      dtype = np.uint8
    elif high <= 0xFFFF:
      dtype = np.uint16
    elif high <= 0xFFFFFFFF:
      dtype = np.uint32
    else:
      dtype = array.dtype
  return array.astype(dtype, copy=False)


def corrprod_to_autocorr(corrprods):
  """Find the autocorrelation indices of correlation products.

  Parameters
  ----------
  corrprods : sequence of 2-tuples or ndarray
      Input labels of the correlation products

  Returns
  -------
  auto_indices : np.ndarray
      The indices in corrprods that correspond to auto-correlations
  index1, index2 : np.ndarray
      Lists of the same length as corrprods, containing the indices within
      `auto_indices` referring to the first and second corresponding
      autocorrelations.

  Raises
  ------
  KeyError
      If any of the autocorrelations are missing
  """
  auto_indices = []
  auto_lookup = {}
  for i, baseline in enumerate(corrprods):
    if baseline[0] == baseline[1]:
      auto_lookup[baseline[0]] = len(auto_indices)
      auto_indices.append(i)
  index1 = [auto_lookup[a] for (a, b) in corrprods]
  index2 = [auto_lookup[b] for (a, b) in corrprods]

  return AutoCorrelationIndices(
    auto_indices=_narrow(np.array(auto_indices)),
    index1=_narrow(np.array(index1)),
    index2=_narrow(np.array(index2)),
  )


if numba:

  @numba.jit(nopython=True, nogil=True)
  def weight_power_scale(
    vis, weights, auto_indices, index1, index2, out=None, divide=True
  ):
    """Divide (or multiply) weights by autocorrelations (ndarray version).

    The weight associated with visibility (i,j) is divided (or multiplied) by
    the corresponding real visibilities (i,i) and (j,j).

    This function is designed to be usable with :func:`dask.array.blockwise`.

    Parameters
    ----------
    vis : np.ndarray
        Chunk of visibility data, with dimensions time, frequency, baseline
        (or any two dimensions then baseline). It must contain all the
        baselines of a stream, even though only the autocorrelations are used.
    weights : np.ndarray
        Chunk of weight data, with the same shape as `vis`
    auto_indices, index1, index2 : np.ndarray
        Arrays returned by :func:`corrprod_to_autocorr`
    out : np.ndarray, optional
        If specified, the output array, with same shape as `vis` and
        dtype ``np.float32``
    divide : bool, optional
        True if weights will be divided by autocorrelations, otherwise
        they will be multiplied
    """
    auto_scale = np.empty(len(auto_indices), np.float32)
    out = np.empty(vis.shape, np.float32) if out is None else out
    bad_weight = np.float32(2.0**-32)
    for i in range(vis.shape[0]):
      for j in range(vis.shape[1]):
        for k in range(len(auto_indices)):
          autocorr = vis[i, j, auto_indices[k]].real
          auto_scale[k] = np.reciprocal(autocorr) if divide else autocorr
        for k in range(vis.shape[2]):
          p = auto_scale[index1[k]] * auto_scale[index2[k]]
          # If either or both of the autocorrelations has zero power then
          # there is likely something wrong with the system. Set the
          # weight to very close to zero (not actually zero, since that
          # can cause divide-by-zero problems downstream).
          if not np.isfinite(p):
            p = bad_weight
          out[i, j, k] = p * weights[i, j, k]
    return out
else:

  def weight_power_scale(
    vis, weights, auto_indices, index1, index2, out=None, divide=True
  ):
    raise NotImplementedError("pure numpy weight_power_scale")
