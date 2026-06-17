from __future__ import annotations

from bisect import insort
from collections import defaultdict
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

from xarray_kat.utils import ANTENNA_RECEPTOR_REGEX

if TYPE_CHECKING:
  from xarray_kat.katdal_types import CorrectionParams


def calc_correction_per_antenna(dump: int, channels: slice, params: CorrectionParams):
  """Prototype: gain correction per channel per *input* for a given dump.

  This is the per-antenna-feed counterpart to
  :func:`calc_correction_per_corrprod`. It performs the same accumulation of
  all requested calibration products into a gain per input (where an input is
  an antenna-pol feed such as ``m000h``), but stops short of folding pairs of
  inputs into correlation products. The returned gains can be combined into
  per-corrprod gains as ``g[:, i] * conj(g[:, j])`` for a baseline of inputs
  ``i`` and ``j``.

  Parameters
  ----------
  dump : int
      Dump index (applicable to full data set, i.e. absolute)
  channels : slice
      Channel indices (applicable to full data set, i.e. absolute)
  params : :class:`CorrectionParams`
      Corrections per input

  Returns
  -------
  gains : array of complex64, shape (n_antenna, n_chans, n_pol)
      Gain corrections per channel per input, columns ordered as
      ``params.inputs``
  """

  # Build a map of the form {"mxyz: [("h", i), ("v", j)]"}
  # where mxyz is the antenna number and i and j
  # are the indices in params.inputs of m000xh and m000xv respectively
  antenna_receptor_map: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)

  for i, antenna_receptor in enumerate(params.inputs):
    if (
      (m := ANTENNA_RECEPTOR_REGEX.match(antenna_receptor)) is None
      or (prefix := m.group("prefix")) is None
      or (number := m.group("number")) is None
      or (receptor := m.group("receptor")) is None
    ):
      raise ValueError(f"Invalid antenna receptor string '{antenna_receptor}'")

    insort(antenna_receptor_map[f"{prefix}{number}"], (receptor, i))

  # This should always hold, but perform a sanity check
  for (
    antenna,
    receptor_map,
  ) in antenna_receptor_map.items():
    if (ant_receptors := tuple(r for r, _ in receptor_map)) != ("h", "v"):
      raise ValueError(
        f"{antenna} does not have corrections for both "
        f"h and v receptors: {ant_receptors}"
      )

  nant = len(antenna_receptor_map)
  npol = 4  # Follows from the above invariant
  nchan = channels.stop - channels.start
  gains = np.ones((nant, nchan, npol), dtype="complex64")

  # Iterate over each gain type (cal_product)
  for cal_product, product_corrections in params.corrections.items():
    channel_map = params.channel_maps[cal_product]

    # Construct each antenna's gains from it's feed receptors
    for a, (antenna, ((_, hi), (_, vi))) in enumerate(antenna_receptor_map.items()):
      h_corrections = product_corrections[hi][dump]
      v_corrections = product_corrections[vi][dump]
      h_gains = channel_map(h_corrections, channels)
      v_gains = channel_map(v_corrections, channels)
      args = ((h_gains,), (v_gains,))

      # Multiply feed receptor values into the net gain for each polarisation
      for p, ((r1_gains,), (r2_gains,)) in enumerate(product(*(args, args))):
        gains[a, :, p] *= r1_gains
        gains[a, :, p] *= np.conj(r2_gains)

  return gains
