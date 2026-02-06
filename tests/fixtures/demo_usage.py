#!/usr/bin/env python3
"""Demonstration of synthetic observation generation for testing.

This script shows how to use the SyntheticObservation class to create
mock MeerKAT observation data for testing xarray-kat.
"""

from pathlib import Path
import tempfile

from katsdptelstate import TelescopeState

from synthetic_observation import SyntheticObservation


def main():
  """Create and demonstrate a synthetic observation."""

  # Create a temporary directory for output
  with tempfile.TemporaryDirectory() as tmpdir:
    output_path = Path(tmpdir)

    # Create a synthetic observation
    print("Creating synthetic observation...")
    obs = SyntheticObservation(
      capture_block_id="1234567890",
      ntime=20,  # 20 time samples
      nfreq=32,  # 32 frequency channels
      nants=4,  # 4 antennas
      int_time=8.0,  # 8 second integration
    )

    # Add multiple scans
    obs.add_scan(indices=range(0, 10), state="track", target="PKS1934")
    obs.add_scan(indices=range(10, 15), state="slew", target=None)
    obs.add_scan(indices=range(15, 20), state="scan", target="3C286")

    print(f"  Observation: {obs.capture_block_id}")
    print(f"  Dimensions: {obs.ntime} time × {obs.nfreq} freq × {obs.ncorrprod} corrprod")
    print(f"  Antennas: {obs.nants} ({', '.join(obs.ant_names)})")
    print(f"  Baselines: {obs.nbl} (including autocorr)")
    print(f"  Scans: {len(obs.scan_configs)}")

    # Save to disk
    print("\nSaving observation to disk...")
    stats = obs.save_to_directory(output_path)

    print(f"  RDB keys: {stats['rdb_keys']}")
    print(f"  Visibility chunks: {stats['correlator_data_chunks']}")
    print(f"  Flag chunks: {stats['flags_chunks']}")
    print(f"  Weight chunks: {stats['weights_chunks']}")
    print(f"  Channel weight chunks: {stats['weights_channel_chunks']}")

    # Verify by loading the telstate
    print("\nVerifying telstate...")
    rdb_path = output_path / "1234567890_sdp_l0.full.rdb"
    ts = TelescopeState()
    ts.load_from_file(rdb_path)

    print(f"  Integration time: {ts.get('int_time')} s")
    print(f"  Center frequency: {ts.get('center_freq') / 1e9:.3f} GHz")
    print(f"  Bandwidth: {ts.get('bandwidth') / 1e6:.1f} MHz")
    print(f"  Number of channels: {ts.get('n_chans')}")
    print(f"  Correlation products: {len(ts.get('bls_ordering'))}")

    # Check chunk_info
    chunk_info = ts.get("chunk_info")
    print("\nArray metadata:")
    for array_name in ["correlator_data", "flags", "weights", "weights_channel"]:
      meta = chunk_info[array_name]
      print(f"  {array_name}:")
      print(f"    shape: {meta['shape']}")
      print(f"    dtype: {meta['dtype']}")
      print(f"    chunks: {len(meta['chunks'][0])} × {len(meta['chunks'][1])}")

    # List generated files
    print("\nGenerated files:")
    print(f"  RDB file: {rdb_path.name}")
    prefix = "1234567890-sdp-l0"
    for array_name in ["correlator_data", "flags", "weights", "weights_channel"]:
      array_dir = output_path / prefix / array_name
      npy_files = list(array_dir.glob("*.npy"))
      print(f"  {array_name}/: {len(npy_files)} .npy chunks")

    print("\n✓ Synthetic observation created successfully!")
    print(f"\nTo serve this data via HTTP, copy the contents of {output_path}/")
    print("to your mock HTTP server's data directory.")


if __name__ == "__main__":
  main()
