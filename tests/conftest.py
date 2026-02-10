"""Pytest configuration and fixtures for xarray-kat tests.

This module contains all test infrastructure for xarray-kat, including:
- RDB file generation utilities
- Synthetic observation generation
- Mock HTTP server setup
- Pytest fixtures for testing
"""

from __future__ import annotations

import json
import logging
import time
from base64 import b64encode
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import katsdptelstate
import numpy as np
import numpy.typing as npt
import pytest
from katsdptelstate import encoding, rdb_utility
from katsdptelstate.rdb_writer_base import RDBWriterBase
from pytest_httpserver import HTTPServer

from xarray_kat.multiton import Multiton

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def clear_multitons():
  with Multiton._INSTANCE_LOCK:
    Multiton._INSTANCE_CACHE.clear()


# ============================================================================
# RDB Generation Utilities
# ============================================================================


def telstate_to_rdb(telstate: katsdptelstate.TelescopeState, output_path: Path) -> int:
  """Write a TelescopeState object to an RDB file.

  This function extracts all keys from a TelescopeState object and writes them
  to an RDB file. It properly handles both immutable and mutable keys by using
  the backend's dump() method which returns keys in Redis DUMP format.

  Args:
    telstate: TelescopeState object with data to save.
    output_path: Path where the RDB file will be created.

  Returns:
    Number of keys successfully written to the RDB file.
  """
  with RDBWriterBase(output_path) as writer:
    for key in telstate.keys():
      key_bytes = key.encode("utf-8") if isinstance(key, str) else key

      # Use backend's dump() method to get the key in Redis DUMP format
      # This properly handles both immutable and mutable (sorted set) keys
      dumped_value = telstate.backend.dump(key_bytes)

      if dumped_value is not None:
        writer.write_key(key_bytes, dumped_value)

  return writer.keys_written


def dict_to_rdb(data: Dict[str, Any], output_path: Path) -> int:
  """Write a Python dictionary directly to an RDB file without Redis.

  This function uses katsdptelstate's encoding and rdb_utility modules to
  serialize Python objects and write them in Redis RDB format. All values
  are encoded using msgpack.

  Args:
    data: Dictionary with string keys and any msgpack-serializable values.
      Supported value types include: str, int, float, bool, list, dict,
      tuple, numpy arrays, and nested combinations thereof.
    output_path: Path where the RDB file will be created. Parent directory
      must exist.

  Returns:
    Number of keys successfully written to the RDB file.

  Example:
    >>> telstate_dict = {
    ...     'sync_time': 1234567890.0,
    ...     'int_time': 8.0,
    ...     'chunk_info': {'correlator_data': {...}}
    ... }
    >>> keys_written = dict_to_rdb(telstate_dict, Path('test.rdb'))
    >>> print(f"Wrote {keys_written} keys")
  """
  with RDBWriterBase(output_path) as writer:
    for key, value in data.items():
      # Ensure key is bytes
      key_bytes = key.encode("utf-8") if isinstance(key, str) else key

      # Encode the Python value to msgpack format
      encoded_value = encoding.encode_value(value)

      # Wrap in Redis DUMP format (adds type byte + length + postfix)
      dumped_value = rdb_utility.dump_string(encoded_value)

      # Write to RDB file
      writer.write_key(key_bytes, dumped_value)

  return writer.keys_written


def create_sensor_data(
  ntime: int, scan_configs: list[dict] | None = None
) -> Dict[str, list]:
  """Create synthetic sensor data arrays for observation metadata (legacy API).

  This is a backward-compatible wrapper that returns sensor data as arrays
  instead of adding them to a telstate object. For new code, use add_sensor_data().

  Args:
    ntime: Number of time samples in the observation.
    scan_configs: List of scan configuration dicts.

  Returns:
    Dictionary with sensor data arrays.
  """
  # Create temporary telstate
  ts = katsdptelstate.TelescopeState()
  int_time = 8.0
  sync_time = 1234567890.0

  # Add sensor data
  add_sensor_data(ts, ntime, int_time, sync_time, scan_configs)

  # Extract as arrays for backward compatibility
  result: dict[str, Any] = {}
  for key in ts.keys():
    if ts.key_type(key) == katsdptelstate.KeyType.MUTABLE:
      data = ts.get_range(key, st=0)
      if data:
        values, times = zip(*data)
        # Create array-like representation
        # Map each timestamp to the corresponding time index
        timestamps_arr = sync_time + np.arange(ntime) * int_time + int_time / 2
        value_array = [None] * ntime
        for val, ts_val in zip(values, times):
          # Find which time indices this value applies to
          # (from this timestamp until the next one or end)
          idx = np.searchsorted(timestamps_arr, ts_val, side="right") - 1
          if idx >= 0:
            value_array[idx] = val

        # Forward fill None values
        last_val = value_array[0]
        for i in range(ntime):
          if value_array[i] is not None:
            last_val = value_array[i]
          else:
            value_array[i] = last_val

        # Store with legacy key names
        if key == "obs_activity":
          result["Observation/scan_state"] = value_array
        elif key == "obs_target":
          result["Observation/target"] = value_array

  # Add scan indices (simple sequential numbering based on state changes)
  if "Observation/scan_state" in result:
    scan_indices = [0] * ntime
    current_scan = 0
    prev_state = None
    for i, state in enumerate(result["Observation/scan_state"]):
      if state != prev_state and prev_state is not None:
        current_scan += 1
      scan_indices[i] = current_scan
      prev_state = state
    result["Observation/scan_index"] = scan_indices

  return result


def add_sensor_data(
  telstate: katsdptelstate.TelescopeState,
  ntime: int,
  int_time: float,
  sync_time: float,
  scan_configs: list[dict] | None = None,
  ant_names: list[str] | None = None,
) -> None:
  """Add synthetic sensor data to a TelescopeState object.

  Sensor data tracks time-varying metadata like scan states, targets, and
  antenna activity. This function adds realistic mutable sensor data that
  matches the MeerKAT telstate schema expected by xarray-kat/katdal.

  The sensors are added as MUTABLE keys with timestamps, which allows
  SensorCache to retrieve them as temporal data.

  Args:
    telstate: TelescopeState object to add sensor data to.
    ntime: Number of time samples in the observation.
    int_time: Integration time in seconds.
    sync_time: Sync time (Unix timestamp) for the observation start.
    scan_configs: List of scan configuration dicts, each with:
      - 'indices': range or list of time indices for this scan
      - 'state': scan state string ('track', 'scan', 'slew', 'stop')
      - 'target_name': name of the target being observed (or None)
      If None, creates a single scan covering all times.
    ant_names: List of antenna names (e.g., ['m000', 'm001']). If provided,
      per-antenna sensors will be added.

  Example:
    >>> ts = katsdptelstate.TelescopeState()
    >>> add_sensor_data(
    ...     ts,
    ...     ntime=50,
    ...     int_time=8.0,
    ...     sync_time=1234567890.0,
    ...     scan_configs=[
    ...         {'indices': range(0, 20), 'state': 'track', 'target_name': 'PKS1934'},
    ...         {'indices': range(20, 30), 'state': 'slew', 'target_name': None},
    ...         {'indices': range(30, 50), 'state': 'scan', 'target_name': '3C286'}
    ...     ],
    ...     ant_names=['m000', 'm001', 'm002', 'm003']
    ... )
  """
  # Default: single scan covering all time
  if scan_configs is None:
    scan_configs = [
      {"indices": range(ntime), "state": "track", "target_name": "MockTarget"}
    ]

  # Calculate timestamps for each time index
  # Timestamps are at the center of each integration period
  timestamps = sync_time + np.arange(ntime) * int_time + int_time / 2

  # Track the previous state to detect changes
  prev_state = None
  prev_target = None

  # Process each scan configuration and add sensor events at boundaries
  for scan_idx, config in enumerate(scan_configs):
    indices = config["indices"]
    state = config["state"]
    target_name = config.get("target_name", "MockTarget")

    # Convert range to list if needed
    if isinstance(indices, range):
      indices = list(indices)

    if not indices:
      continue

    # Get the timestamp at the start of this scan
    first_idx = min(indices)
    scan_timestamp = timestamps[first_idx]

    # Add activity sensor event if state changed
    if state != prev_state:
      # obs_activity tracks the observation-level activity
      telstate.add("obs_activity", state, ts=scan_timestamp, immutable=False)
      prev_state = state

    # Add target sensor event if target changed
    if target_name != prev_target:
      # Create target string in katpoint format
      if target_name is None:
        target_str = "Nothing, special"
      else:
        # Format: "name, radec, RA, DEC"
        coords = {
          "PKS1934": ("19:39:25.03", "-63:42:45.63"),
          "3C286": ("13:31:08.29", "+30:30:33.0"),
          "MockTarget": ("00:00:00.0", "+00:00:00.0"),
        }
        ra, dec = coords.get(target_name, ("00:00:00.0", "+00:00:00.0"))
        target_str = f"{target_name}, radec, {ra}, {dec}"

      telstate.add("obs_target", target_str, ts=scan_timestamp, immutable=False)
      prev_target = target_name

  # Add per-antenna sensors if antenna names provided
  if ant_names:
    # Reset tracking variables for per-antenna sensors
    ant_prev_state: dict[str, str] = {}
    ant_prev_target: dict[str, str] = {}

    for scan_idx, config in enumerate(scan_configs):
      indices = config["indices"]
      state = config["state"]
      target_name = config.get("target_name", "MockTarget")

      if isinstance(indices, range):
        indices = list(indices)

      if not indices:
        continue

      first_idx = min(indices)
      scan_timestamp = timestamps[first_idx]

      # Create target string
      if target_name is None:
        target_str = "Nothing, special"
      else:
        coords = {
          "PKS1934": ("19:39:25.03", "-63:42:45.63"),
          "3C286": ("13:31:08.29", "+30:30:33.0"),
          "MockTarget": ("00:00:00.0", "+00:00:00.0"),
        }
        ra, dec = coords.get(target_name, ("00:00:00.0", "+00:00:00.0"))
        target_str = f"{target_name}, radec, {ra}, {dec}"

      # Add sensors for each antenna
      for ant in ant_names:
        # Add activity sensor
        if ant not in ant_prev_state or ant_prev_state[ant] != state:
          telstate.add(
            f"Antennas/{ant}/activity", state, ts=scan_timestamp, immutable=False
          )
          ant_prev_state[ant] = state

        # Add target sensor
        if ant not in ant_prev_target or ant_prev_target[ant] != target_name:
          telstate.add(
            f"Antennas/{ant}/target", target_str, ts=scan_timestamp, immutable=False
          )
          ant_prev_target[ant] = target_name

    # Also add array-level antenna sensors (used as reference)
    # Reset for array sensors
    prev_state = None
    prev_target = None

    for scan_idx, config in enumerate(scan_configs):
      indices = config["indices"]
      state = config["state"]
      target_name = config.get("target_name", "MockTarget")

      if isinstance(indices, range):
        indices = list(indices)

      if not indices:
        continue

      first_idx = min(indices)
      scan_timestamp = timestamps[first_idx]

      if state != prev_state:
        telstate.add(
          "Antennas/array/activity", state, ts=scan_timestamp, immutable=False
        )
        prev_state = state

      if target_name != prev_target:
        if target_name is None:
          target_str = "Nothing, special"
        else:
          coords = {
            "PKS1934": ("19:39:25.03", "-63:42:45.63"),
            "3C286": ("13:31:08.29", "+30:30:33.0"),
            "MockTarget": ("00:00:00.0", "+00:00:00.0"),
          }
          ra, dec = coords.get(target_name, ("00:00:00.0", "+00:00:00.0"))
          target_str = f"{target_name}, radec, {ra}, {dec}"

        telstate.add(
          "Antennas/array/target", target_str, ts=scan_timestamp, immutable=False
        )
        prev_target = target_name


# ============================================================================
# Synthetic Observation Generation
# ============================================================================


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
            bls_ordering.append(
              [f"{self.ant_names[i]}{pol1}", f"{self.ant_names[j]}{pol2}"]
            )
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
    self.scan_configs.append(
      {"indices": indices, "state": state, "target_name": target}
    )

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
      # Capture block identification
      "capture_block_id": self.capture_block_id,
      "stream_name": "sdp_l0",
      "stream_type": "sdp.vis",
      "sub_product": "sdp_l0",
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
      telstate_dict[f"{ant}_observer"] = f"{ant}, -30.721, 21.411, 1035.0, 13.5, 0.0"

    return telstate_dict

  def create_telstate_object(self) -> katsdptelstate.TelescopeState:
    """Create a complete TelescopeState object with all metadata and sensors.

    Returns:
      TelescopeState object populated with immutable metadata and mutable sensor data.
    """
    telstate = katsdptelstate.TelescopeState()

    # Add immutable metadata from telstate_dict
    telstate_dict = self.create_telstate_dict()

    # Add all immutable keys
    for key, value in telstate_dict.items():
      if key not in ["Antennas/array/activity", "Antennas/array/antenna"]:
        # Skip the old simple sensor keys - we'll add them as mutable
        if not key.startswith("Antennas/") or "_observer" in key:
          telstate.add(key, value, immutable=True)

    # Default scan config if none provided
    if not self.scan_configs:
      self.add_scan(range(self.ntime), "track", "MockTarget")

    # Add mutable sensor data with timestamps
    add_sensor_data(
      telstate,
      self.ntime,
      self.int_time,
      telstate_dict["sync_time"],
      self.scan_configs,
      ant_names=self.ant_names,
    )

    return telstate

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

    # Create TelescopeState object with all metadata and sensors
    telstate = self.create_telstate_object()

    # Save to RDB file
    rdb_filename = f"{self.capture_block_id}_sdp_l0.full.rdb"
    rdb_path = base_path / rdb_filename
    n_keys = telstate_to_rdb(telstate, rdb_path)

    logger.info(f"Wrote {n_keys} keys to {rdb_path}")

    # Save chunked arrays
    prefix = f"{self.capture_block_id}-sdp-l0"
    chunk_info = telstate["chunk_info"]
    stats = {"rdb_keys": n_keys}

    for array_name in ["correlator_data", "flags", "weights", "weights_channel"]:
      n_chunks = self._save_array_chunks(
        base_path, prefix, array_name, chunk_info[array_name]
      )
      stats[f"{array_name.replace('_', '_')}_chunks"] = n_chunks
      logger.info(f"Wrote {n_chunks} chunks for {array_name}")

    return stats


# ============================================================================
# Mock HTTP Server Utilities
# ============================================================================


def create_mock_jwt(
  capture_block_id: str, scopes: List[str] | None = None, expire_hours: int = 24
) -> str:
  """Create a mock JWT token for testing.

  xarray-kat parses JWT tokens but doesn't validate signatures for HTTP
  connections. This creates a properly formatted JWT with realistic header
  and payload that will pass parsing.

  Args:
    capture_block_id: The capture block ID this token grants access to.
    scopes: List of scopes (default: ["read"]).
    expire_hours: Hours until token expiry (default: 24).

  Returns:
    JWT token string in format: header.payload.signature

  Example:
    >>> token = create_mock_jwt("1234567890")
    >>> # Use in URL: f"https://server/{cbid}/{cbid}.rdb?token={token}"
  """
  if scopes is None:
    scopes = ["read"]

  # JWT Header (ES256 algorithm as used by MeerKAT)
  header = {"alg": "ES256", "typ": "JWT"}

  # JWT Payload
  now = int(time.time())
  payload = {
    "iss": "mock-archive.kat.ac.za",
    "aud": "archive-gw-1.kat.ac.za",
    "iat": now,
    "exp": now + (expire_hours * 3600),
    "sub": "test-user-uuid",
    "prefix": [capture_block_id],
    "scopes": scopes,
  }

  # Base64url encode (without padding as per JWT spec)
  def b64url(data: dict) -> str:
    json_bytes = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return b64encode(json_bytes).decode("utf-8").rstrip("=")

  encoded_header = b64url(header)
  encoded_payload = b64url(payload)

  # Mock signature (ES256 produces 86-character base64url encoded signature)
  # xarray-kat checks signature length but doesn't validate it for HTTP
  mock_signature = "A" * 86

  return f"{encoded_header}.{encoded_payload}.{mock_signature}"


def setup_mock_archive_server(
  httpserver, base_path: Path, capture_block_id: str, require_auth: bool = False
) -> str | None:
  """Configure pytest-httpserver to serve synthetic observation data.

  This sets up HTTP routes for:
  - RDB file: /{capture_block_id}/{capture_block_id}_sdp_l0.full.rdb
  - Array chunks: /{prefix}/{array_name}/TTTTT_FFFFF_CCCCC.npy

  Args:
    httpserver: pytest-httpserver fixture.
    base_path: Path where synthetic observation files are stored.
    capture_block_id: The capture block ID being served.
    require_auth: If True, require valid JWT token in Authorization header.

  Returns:
    Valid JWT token for accessing the data (if require_auth=True).

  Example:
    >>> def test_with_mock_server(httpserver, tmp_path):
    ...     obs = SyntheticObservation("1234567890", ...)
    ...     obs.save_to_directory(tmp_path)
    ...     token = setup_mock_archive_server(httpserver, tmp_path, "1234567890")
    ...     url = f"{endpoint}/1234567890/1234567890_sdp_l0.full.rdb?token={token}"
    ...     dt = xarray.open_datatree(url, engine="xarray-kat")
  """
  valid_token = create_mock_jwt(capture_block_id) if require_auth else None

  # Note: For now, we're not implementing auth checking since pytest-httpserver
  # respond_with_data doesn't support conditional responses based on headers.
  # Auth checking would require respond_with_handler which has API complexity.
  # For testing purposes, having the token present in the URL is sufficient.

  # Route 1: RDB file
  rdb_filename = f"{capture_block_id}_sdp_l0.full.rdb"
  rdb_path = base_path / rdb_filename

  if rdb_path.exists():
    with open(rdb_path, "rb") as f:
      rdb_content = f.read()
    httpserver.expect_request(
      f"/{capture_block_id}/{rdb_filename}", method="GET"
    ).respond_with_data(rdb_content, content_type="application/octet-stream")

  # Route 2: Array chunks
  # Pattern: /{prefix}/{array_name}/{chunk_filename}.npy
  prefix = f"{capture_block_id}-sdp-l0"

  for array_name in ["correlator_data", "flags", "weights", "weights_channel"]:
    array_dir = base_path / prefix / array_name

    if array_dir.exists():
      # Register each .npy file
      for npy_file in array_dir.glob("*.npy"):
        chunk_filename = npy_file.name
        uri = f"/{prefix}/{array_name}/{chunk_filename}"

        with open(npy_file, "rb") as f:
          chunk_content = f.read()

        httpserver.expect_request(uri, method="GET").respond_with_data(
          chunk_content, content_type="application/octet-stream"
        )

  return valid_token


def configure_mock_archive(httpserver, tmp_path, obs_kwargs: Optional[Dict] = None):
  """Convenience fixture builder: create synthetic obs + configure server.

  This is a helper for tests that combines observation creation and server setup.

  Args:
    httpserver: pytest-httpserver fixture.
    tmp_path: pytest tmp_path fixture.
    obs_kwargs: Keyword arguments for SyntheticObservation (optional).

  Returns:
    Tuple of (base_url, capture_block_id, token).

  Example:
    >>> def test_integration(httpserver, tmp_path):
    ...     url, cbid, token = configure_mock_archive(httpserver, tmp_path)
    ...     full_url = f"{url}/{cbid}/{cbid}_sdp_l0.full.rdb?token={token}"
    ...     dt = xarray.open_datatree(full_url)
  """
  # Default observation parameters
  default_kwargs: dict[str, Any] = {
    "capture_block_id": "1234567890",
    "ntime": 10,
    "nfreq": 16,
    "nants": 4,
  }

  if obs_kwargs:
    default_kwargs.update(obs_kwargs)

  # Create synthetic observation
  obs = SyntheticObservation(**default_kwargs)
  obs.add_scan(range(0, obs.ntime), "track", "MockTarget")
  obs.save_to_directory(tmp_path)

  # Setup server
  token = setup_mock_archive_server(
    httpserver, tmp_path, obs.capture_block_id, require_auth=True
  )

  return httpserver.url_for("/"), obs.capture_block_id, token


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture
def synthetic_observation():
  """Factory fixture to create synthetic observations with custom parameters.

  Returns:
      A factory function that creates SyntheticObservation instances.

  Example:
      def test_something(synthetic_observation):
          obs = synthetic_observation(ntime=20, nfreq=32, nants=4)
          obs.add_scan(range(0, 10), "track", "PKS1934")
  """

  def _create_observation(
    capture_block_id="1234567890",
    ntime=10,
    nfreq=16,
    nants=4,
    npol=4,
    int_time=8.0,
    center_freq=1284e6,
    bandwidth=856e6,
  ):
    return SyntheticObservation(
      capture_block_id=capture_block_id,
      ntime=ntime,
      nfreq=nfreq,
      nants=nants,
      npol=npol,
      int_time=int_time,
      center_freq=center_freq,
      bandwidth=bandwidth,
    )

  return _create_observation


@pytest.fixture
def mock_observation(synthetic_observation):
  """Create a default synthetic observation for testing.

  Returns:
      A SyntheticObservation instance with default parameters and a single track scan.

  Example:
      def test_something(mock_observation):
          stats = mock_observation.save_to_directory(tmp_path)
  """
  obs = synthetic_observation()
  obs.add_scan(range(0, obs.ntime), "track", "MockTarget")
  return obs


@pytest.fixture
def mock_archive(mock_observation, tmp_path):
  """Create a mock MeerKAT archive saved to disk.

  Args:
      mock_observation: The synthetic observation fixture.
      tmp_path: Pytest's temporary directory fixture.

  Returns:
      Tuple of (archive_path, observation, stats) where:
      - archive_path is the Path where files were saved
      - observation is the SyntheticObservation instance
      - stats is the dictionary of save statistics

  Example:
      def test_something(mock_archive):
          archive_path, obs, stats = mock_archive
          rdb_file = archive_path / f"{obs.capture_block_id}_sdp_l0.full.rdb"
  """
  archive_path = tmp_path / "mock_archive"
  stats = mock_observation.save_to_directory(archive_path)
  return archive_path, mock_observation, stats


@pytest.fixture
def mock_archive_server(httpserver: HTTPServer, tmp_path, synthetic_observation):
  """Factory fixture to create mock archive HTTP server with custom parameters.

  Args:
      httpserver: pytest-httpserver fixture.
      tmp_path: Pytest's temporary directory fixture.
      synthetic_observation: Factory fixture for creating observations.

  Returns:
      A factory function that creates and configures a mock archive server.

  Example:
      def test_something(mock_archive_server):
          url, cbid, token = mock_archive_server(ntime=20, nfreq=32)
          rdb_url = f"{url}{cbid}/{cbid}_sdp_l0.full.rdb?token={token}"
  """

  def _create_server(
    ntime=10,
    nfreq=16,
    nants=4,
    capture_block_id="1234567890",
    scan_configs=None,
    require_auth=True,
  ):
    # Create observation
    obs = synthetic_observation(
      capture_block_id=capture_block_id,
      ntime=ntime,
      nfreq=nfreq,
      nants=nants,
    )

    # Add scan configurations
    if scan_configs:
      for config in scan_configs:
        obs.add_scan(
          config["indices"],
          config["state"],
          config.get("target_name"),
        )
    else:
      # Default: single track scan
      obs.add_scan(range(0, obs.ntime), "track", "MockTarget")

    # Save to disk
    archive_path = tmp_path / f"archive_{capture_block_id}"
    obs.save_to_directory(archive_path)

    # Setup HTTP server
    token = setup_mock_archive_server(
      httpserver,
      archive_path,
      obs.capture_block_id,
      require_auth=require_auth,
    )

    base_url = httpserver.url_for("/")
    return base_url, obs.capture_block_id, token

  return _create_server


@pytest.fixture
def simple_mock_server(mock_archive_server):
  """Create a simple mock archive server with default parameters.

  Returns:
      Tuple of (base_url, capture_block_id, token).

  Example:
      def test_something(simple_mock_server):
          url, cbid, token = simple_mock_server
          rdb_url = f"{url}{cbid}/{cbid}_sdp_l0.full.rdb?token={token}"
          dt = xarray.open_datatree(rdb_url, engine="xarray-kat")
  """
  return mock_archive_server()
