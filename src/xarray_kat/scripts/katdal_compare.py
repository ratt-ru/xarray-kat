import argparse

import numpy as np
import xarray

from xarray_kat.katdal_types import TelstateDataProducts
from xarray_kat.multiton import Multiton
from xarray_kat.utils import corrprods_to_baseline_pols

try:
  import katdal
except ImportError as e:
  raise ValueError("pip install katdal") from e


def create_parser():
  p = argparse.ArgumentParser()
  p.add_argument("url")
  p.add_argument("--applycal", default="all")
  return p.parse_args()


def report_mismatch(actual, desired, decimal=6):
  """Report differences in non-matching visibility data, similar to
  np.testing.assert_array_almost_equal output."""
  abs_diff = np.abs(actual - desired)
  threshold = 1.5 * 10 ** (-decimal)
  mismatch_mask = abs_diff >= threshold

  n_mismatch = np.count_nonzero(mismatch_mask)
  n_total = mismatch_mask.size

  print(
    f"  Mismatched elements: {n_mismatch} / {n_total} "
    f"({100.0 * n_mismatch / n_total:.1f}%)"
  )

  if n_mismatch == 0:
    return

  mismatch_diffs = abs_diff[mismatch_mask]
  max_abs_diff = mismatch_diffs.max()
  print(f"  Max absolute difference: {max_abs_diff:.6g}")

  # Relative difference, guarded against division by zero
  with np.errstate(invalid="ignore", divide="ignore"):
    rel_diff = abs_diff / (np.abs(desired) + 1e-300)
  max_rel_diff = rel_diff[mismatch_mask].max()
  print(f"  Max relative difference: {max_rel_diff:.6g}")

  # Worst-offending index (flat → multi-dimensional)
  worst_flat = np.argmax(abs_diff)
  worst_idx = np.unravel_index(worst_flat, abs_diff.shape)
  print(f"  Worst mismatch at index {worst_idx}:")
  print(f"    actual:  {actual[worst_idx]}")
  print(f"    desired: {desired[worst_idx]}")
  print(f"    abs diff: {abs_diff[worst_idx]:.6g}")

  # Distribution of mismatches across axes (time, baseline, freq, pol)
  axis_names = ("time", "baseline", "frequency", "polarization")
  for axis, name in enumerate(axis_names[: mismatch_mask.ndim]):
    n_slices_with_mismatches = np.count_nonzero(
      mismatch_mask.any(axis=tuple(i for i in range(mismatch_mask.ndim) if i != axis))
    )
    print(f"  Mismatches span {n_slices_with_mismatches} unique {name} indices")


def compare_vs_katdal(args):
  dt = xarray.open_datatree(
    args.url,
    chunked_array_type="xarray-kat",
    chunks={},
    applycal=args.applycal,
    uvw_sign_convention="fourier",
  )
  ds = katdal.open(args.url, applycal=args.applycal)

  # Compute the argsort into MSv4 order for the correlation products
  data_products = next(
    iter(
      v[0]
      for v in Multiton._INSTANCE_CACHE.values()
      if isinstance(v[0], TelstateDataProducts)
    )
  )
  telstate = data_products.telstate
  corrprods = telstate["bls_ordering"]
  baseline_pols = corrprods_to_baseline_pols(corrprods)
  cp_argsort = np.array(
    sorted(range(len(baseline_pols)), key=lambda i: baseline_pols[i])
  )

  if (cal_params := data_products.calibration_params) is not None:
    print(f"Calibration parameters: {list(cal_params.corrections.keys())}")

  for path, node in dt.children.items():
    scan = node.ds
    print(
      f"Loading {path} of size {scan.nbytes / 1024.0**3:,.2f}GB "
      f"and shape {dict(scan.sizes)} "
      f"{scan.attrs.get('description', '')}"
    )
    scans = list(set(map(int, scan.scan_name.load().values)))
    scan.load()
    ntime, nbl, nfreq, npol = (
      scan.sizes[d] for d in ("time", "baseline_id", "frequency", "polarization")
    )
    ds.select(scans=scans)

    # Compare uvw
    uvw = np.stack([ds.u, ds.v, ds.w], axis=2)[:, cp_argsort]

    if not np.allclose(uvw[:, ::npol], scan.UVW.values):
      print("UVW values don't match")

    # Compare timestamps
    if not np.allclose(ds.timestamps, scan.time.values):
      print(f"Timestamps don't match\n{ds.timestamps}\n{scan.time.values}")

    # Compare frequencies
    if not np.allclose(ds.freqs, scan.frequency.values):
      print("Frequencies don't match\n", f"{ds.freqs}\n{scan.frequency.values}")

    # Compare visibilities
    vis = (
      ds.vis[:, :, cp_argsort].reshape(ntime, nfreq, nbl, npol).transpose(0, 2, 1, 3)
    )
    xr_vis = scan.VISIBILITY.values
    mask = np.isclose(vis, xr_vis)
    matching = np.count_nonzero(mask)
    print(f"{100.0 * matching / mask.size:.1f}% visibilities matching")

    if matching < mask.size:
      report_mismatch(xr_vis, vis)

    # Compare weights
    weights = (
      ds.weights[:, :, cp_argsort]
      .reshape(ntime, nfreq, nbl, npol)
      .transpose(0, 2, 1, 3)
    )
    xr_weights = scan.WEIGHT.values
    mask = np.isclose(weights, xr_weights)
    matching = np.count_nonzero(mask)
    print(f"{100.0 * matching / mask.size:.1f}% weights matching")

    if matching < mask.size:
      report_mismatch(xr_weights, weights)
