"""Generate synthetic MeerKAT observations for testing.

This module provides the SyntheticObservation class which generates complete
mock observations including telstate metadata (RDB file) and chunked visibility
data (.npy files) that can be served via a mock HTTP endpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from .rdb_generator import create_sensor_data, dict_to_rdb

logger = logging.getLogger(__name__)


class SyntheticObservation:
  """Generate synthetic MeerKAT observation data for testing.

  This class creates a complete mock observation with:
  - Telstate metadata (stored in RDB format)
  - Chunked numpy arrays (correlator_data, flags, weights, weights_channel)
  - Realistic scan information and sensor data

  The generated data matches the structure expected by xarray-kat and can be
  served via a mock HTTP endpoint for integration testing.

  Example:
    >>> obs = SyntheticObservation(
    ...     capture_block_id="1234567890",
    ...     ntime=20,
    ...     nfreq=32,
    ...     nants=4
    ... )
    >>> obs.add_scan(indices=range(0, 10), state="track", target="PKS1934")
    >>> obs.add_scan(indices=range(10, 20), state="scan", target="3C286")
    >>> obs.save_to_directory(Path("/tmp/mock_archive"))
  """

  def __init__(
    self,
    capture_block_id: str,
    ntime: int = 10,
    nfreq: int = 16,
    nants: int = 4,
    npol: int = 4,
    int_time: float = 8.0,
    center_freq: float = 1284e6,
    bandwidth: float = 856e6,
  ):
    """Initialize a synthetic observation.

    Args:
      capture_block_id: Unique identifier for this observation (e.g. timestamp).
      ntime: Number of time samples.
      nfreq: Number of frequency channels.
      nants: Number of antennas (must be >= 2).
      npol: Number of polarizations (typically 4 for linear: XX, XY, YX, YY).
      int_time: Integration time in seconds.
      center_freq: Center frequency in Hz.
      bandwidth: Total bandwidth in Hz.
    """
    if nants < 2:
      raise ValueError(f"Need at least 2 antennas, got {nants}")

    self.capture_block_id = capture_block_id
    self.ntime = ntime
    self.nfreq = nfreq
    self.nants = nants
    self.npol = npol
    self.int_time = int_time
    self.center_freq = center_freq
    self.bandwidth = bandwidth

    # Calculate derived quantities
    self.nbl = nants * (nants + 1) // 2  # Include autocorrelations
    self.ncorrprod = self.nbl * npol
    self.channel_width = bandwidth / nfreq

    # Generate antenna names
    self.ant_names = [f"m{i:03d}" for i in range(nants)]

    # Generate correlation products (baseline-polarization pairs)
    self.bls_ordering = self._generate_bls_ordering()

    # Scan configurations (can be added via add_scan)
    self.scan_configs: List[Dict] = []

    # Default chunking (can be customized)
    self.time_chunk_size = 2
    self.freq_chunk_size = 8

  def _generate_bls_ordering(self) -> List[List[str]]:
    """Generate correlation products in MeerKAT format.

    Returns list of [antenna1_pol, antenna2_pol] pairs, e.g.:
    [['m000h', 'm000h'], ['m000h', 'm000v'], ['m000h', 'm001h'], ...]

    This creates all unique baselines (including autocorrelations) for
    each polarization combination.
    """
    bls_ordering = []
    for i in range(self.nants):
      for j in range(i, self.nants):  # i <= j (includes autocorrs)
        for pol1 in ["h", "v"]:
          for pol2 in ["h", "v"]:
            bls_ordering.append([f"{self.ant_names[i]}{pol1}", f"{self.ant_names[j]}{pol2}"])
    return bls_ordering

  def add_scan(
    self, indices: range | List[int], state: str, target: str | None
  ) -> None:
    """Add a scan configuration to the observation.

    Args:
      indices: Time indices for this scan (range or list).
      state: Scan state ('track', 'scan', 'slew', 'stop').
      target: Target name (None for slew states).
    """
    self.scan_configs.append({"indices": indices, "state": state, "target_name": target})

  def _get_time_chunks(self) -> Tuple[int, ...]:
    """Get time chunking tuple for dask-style chunks."""
    n_full_chunks = self.ntime // self.time_chunk_size
    remainder = self.ntime % self.time_chunk_size
    chunks = tuple([self.time_chunk_size] * n_full_chunks)
    if remainder > 0:
      chunks += (remainder,)
    return chunks

  def _get_freq_chunks(self) -> Tuple[int, ...]:
    """Get frequency chunking tuple for dask-style chunks."""
    n_full_chunks = self.nfreq // self.freq_chunk_size
    remainder = self.nfreq % self.freq_chunk_size
    chunks = tuple([self.freq_chunk_size] * n_full_chunks)
    if remainder > 0:
      chunks += (remainder,)
    return chunks

  def create_telstate_dict(self) -> Dict:
    """Create a complete telstate dictionary matching MeerKAT schema.

    Returns:
      Dictionary with all required telstate keys for xarray-kat.
    """
    prefix = f"{self.capture_block_id}-sdp-l0"
    time_chunks = self._get_time_chunks()
    freq_chunks = self._get_freq_chunks()

    telstate_dict = {
      # Timing information
      "sync_time": 1234567890.0,  # Mock epoch time
      "first_timestamp": 0.0,
      "int_time": self.int_time,
      # Frequency information
      "center_freq": self.center_freq,
      "bandwidth": self.bandwidth,
      "n_chans": self.nfreq,
      "sub_band": "l",  # L-band
      # Antenna information
      "sub_pool_resources": ",".join(self.ant_names),
      "bls_ordering": self.bls_ordering,
      # Observation metadata
      "obs_params": {
        "observer": "test_observer",
        "experiment_id": "TEST-001",
        "description": f"Synthetic observation {self.capture_block_id}",
      },
      # Array chunk information
      "chunk_info": {
        "correlator_data": {
          "prefix": prefix,
          "dtype": "<c8",  # complex64 little-endian
          "shape": (self.ntime, self.nfreq, self.ncorrprod),
          "chunks": (time_chunks, freq_chunks, (self.ncorrprod,)),
        },
        "flags": {
          "prefix": prefix,
          "dtype": "|u1",  # uint8
          "shape": (self.ntime, self.nfreq, self.ncorrprod),
          "chunks": (time_chunks, freq_chunks, (self.ncorrprod,)),
        },
        "weights": {
          "prefix": prefix,
          "dtype": "|u1",  # uint8
          "shape": (self.ntime, self.nfreq, self.ncorrprod),
          "chunks": (time_chunks, freq_chunks, (self.ncorrprod,)),
        },
        "weights_channel": {
          "prefix": prefix,
          "dtype": "<f4",  # float32 little-endian
          "shape": (self.ntime, self.nfreq),
          "chunks": (time_chunks, freq_chunks),
        },
      },
    }

    # Add per-antenna observer strings
    for ant in self.ant_names:
      # Format: "name, latitude, longitude, altitude, diameter, delay"
      telstate_dict[f"{ant}_observer"] = (
        f"{ant}, -30.721, 21.411, 1035.0, 13.5, 0.0"
      )

    return telstate_dict

  def create_sensor_cache_dict(self) -> Dict:
    """Create sensor cache data for scan information.

    Returns:
      Dictionary with sensor data arrays.
    """
    if not self.scan_configs:
      # Default: single scan covering all times
      self.add_scan(range(self.ntime), "track", "MockTarget")

    return create_sensor_data(self.ntime, self.scan_configs)

  def generate_array_data(
    self, array_name: str, dtype: npt.DTypeLike, shape: Tuple[int, ...]
  ) -> npt.NDArray:
    """Generate synthetic data for a specific array.

    Args:
      array_name: Name of the array ('correlator_data', 'flags', etc.).
      dtype: Numpy data type.
      shape: Array shape.

    Returns:
      Synthetic data array with realistic values.
    """
    if array_name == "correlator_data":
      # Generate complex visibility data
      # Use a simple pattern: ramp + some phase
      ramp = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
      phase = 2 * np.pi * ramp / np.prod(shape)
      return (ramp * np.exp(1j * phase)).astype(dtype)

    elif array_name == "flags":
      # Generate mostly unflagged data (0 = unflagged, 1+ = flagged)
      # Flag about 5% of data randomly
      data = np.zeros(shape, dtype=dtype)
      n_flagged = int(0.05 * np.prod(shape))
      flat_indices = np.random.choice(np.prod(shape), n_flagged, replace=False)
      data.flat[flat_indices] = 1
      return data

    elif array_name == "weights":
      # Generate weights (typically 0-255 for uint8)
      # Most should be high weight (~200)
      return np.full(shape, 200, dtype=dtype)

    elif array_name == "weights_channel":
      # Channel weights (float, typically 0.0 to 1.0)
      # Use slightly varying weights per channel
      data = np.ones(shape, dtype=dtype)
      # Add some variation
      data *= 0.8 + 0.2 * np.random.rand(*shape)
      return data

    else:
      raise ValueError(f"Unknown array name: {array_name}")

  def _save_array_chunks(
    self, base_path: Path, prefix: str, array_name: str, array_meta: Dict
  ) -> int:
    """Save chunked numpy arrays in MeerKAT archive format.

    Args:
      base_path: Base directory for saving chunks.
      prefix: Data prefix (e.g., '1234567890-sdp-l0').
      array_name: Name of the array.
      array_meta: Metadata dict with 'shape', 'chunks', 'dtype'.

    Returns:
      Number of chunk files written.
    """
    shape = array_meta["shape"]
    chunks = array_meta["chunks"]
    dtype = np.dtype(array_meta["dtype"])

    # Create directory for this array
    array_dir = base_path / prefix / array_name
    array_dir.mkdir(parents=True, exist_ok=True)

    # Generate full array data once
    full_data = self.generate_array_data(array_name, dtype, shape)

    # Calculate chunk boundaries
    chunk_indices = []
    for dim_chunks in chunks:
      boundaries = [0]
      for chunk_size in dim_chunks:
        boundaries.append(boundaries[-1] + chunk_size)
      chunk_indices.append(boundaries)

    # Write each chunk
    n_chunks_written = 0
    for t_idx, t_chunk in enumerate(chunks[0]):
      t_start = sum(chunks[0][:t_idx])
      t_end = t_start + t_chunk

      for f_idx, f_chunk in enumerate(chunks[1]):
        f_start = sum(chunks[1][:f_idx])
        f_end = f_start + f_chunk

        # Handle 2D vs 3D arrays
        if len(shape) == 3:
          # 3D array (time, freq, corrprod)
          # Corrprod is never chunked (always one chunk)
          chunk_data = full_data[t_start:t_end, f_start:f_end, :]
          chunk_filename = f"{t_start:05d}_{f_start:05d}_00000.npy"
        else:
          # 2D array (time, freq) - for weights_channel
          chunk_data = full_data[t_start:t_end, f_start:f_end]
          chunk_filename = f"{t_start:05d}_{f_start:05d}.npy"

        # Save chunk
        chunk_path = array_dir / chunk_filename
        np.save(chunk_path, chunk_data)
        n_chunks_written += 1

    return n_chunks_written

  def save_to_directory(self, base_path: Path) -> Dict[str, int]:
    """Save complete observation to a directory.

    This creates:
    - RDB file with telstate metadata
    - Chunked .npy files for all data arrays

    Args:
      base_path: Directory where files will be saved.

    Returns:
      Dictionary with statistics:
        - 'rdb_keys': Number of RDB keys written
        - 'vis_chunks': Number of visibility chunks
        - 'flag_chunks': Number of flag chunks
        - 'weight_chunks': Number of weight chunks
        - 'weight_channel_chunks': Number of channel weight chunks
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    # Create and save telstate RDB
    telstate_dict = self.create_telstate_dict()
    sensor_cache = self.create_sensor_cache_dict()
    telstate_dict.update(sensor_cache)

    rdb_filename = f"{self.capture_block_id}_sdp_l0.full.rdb"
    rdb_path = base_path / rdb_filename
    n_keys = dict_to_rdb(telstate_dict, rdb_path)

    logger.info(f"Wrote {n_keys} keys to {rdb_path}")

    # Save chunked arrays
    prefix = f"{self.capture_block_id}-sdp-l0"
    chunk_info = telstate_dict["chunk_info"]
    stats = {"rdb_keys": n_keys}

    for array_name in ["correlator_data", "flags", "weights", "weights_channel"]:
      n_chunks = self._save_array_chunks(
        base_path, prefix, array_name, chunk_info[array_name]
      )
      stats[f"{array_name.replace('_', '_')}_chunks"] = n_chunks
      logger.info(f"Wrote {n_chunks} chunks for {array_name}")

    return stats
