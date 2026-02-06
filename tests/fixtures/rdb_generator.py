"""Utilities for generating RDB files from Python dictionaries without Redis.

This module provides functions to create mock MeerKAT telstate RDB files
directly from Python data structures, eliminating the need for a Redis server
during testing.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
from katsdptelstate import encoding, rdb_utility
from katsdptelstate.rdb_writer_base import RDBWriterBase


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
) -> Dict[str, np.ndarray]:
  """Create synthetic sensor data arrays for observation metadata.

  Sensor data tracks time-varying metadata like scan states, targets, and
  antenna pointing. This function generates realistic sensor data arrays
  that match the MeerKAT telstate schema.

  Args:
    ntime: Number of time samples in the observation.
    scan_configs: List of scan configuration dicts, each with:
      - 'indices': range or list of time indices for this scan
      - 'state': scan state string ('track', 'scan', 'slew', 'stop')
      - 'target_name': name of the target being observed (or None)
      If None, creates a single scan covering all times.

  Returns:
    Dictionary with sensor data arrays:
      - 'Observation/target': array of Target objects (or string names)
      - 'Observation/scan_index': array of scan indices (int)
      - 'Observation/scan_state': array of scan states (str)

  Example:
    >>> sensor_data = create_sensor_data(
    ...     ntime=50,
    ...     scan_configs=[
    ...         {'indices': range(0, 20), 'state': 'track', 'target_name': 'PKS1934'},
    ...         {'indices': range(20, 30), 'state': 'slew', 'target_name': None},
    ...         {'indices': range(30, 50), 'state': 'scan', 'target_name': '3C286'}
    ...     ]
    ... )
  """
  # Import Target class for creating proper target objects
  try:
    from katpoint import Target
  except ImportError:
    # Fallback to simple string representation if katpoint not available
    Target = None

  # Default: single scan covering all time
  if scan_configs is None:
    scan_configs = [
      {"indices": range(ntime), "state": "track", "target_name": "MockTarget"}
    ]

  # Initialize arrays
  scan_indices = np.zeros(ntime, dtype=int)
  scan_states = np.empty(ntime, dtype=object)
  targets = np.empty(ntime, dtype=object)

  for scan_idx, config in enumerate(scan_configs):
    indices = config["indices"]
    state = config["state"]
    target_name = config.get("target_name", "MockTarget")

    # Convert range to list if needed
    if isinstance(indices, range):
      indices = list(indices)

    # Fill in arrays
    scan_indices[indices] = scan_idx
    scan_states[indices] = state

    # Create target objects
    if target_name is None:
      # For slew states, use a special "Nothing" target
      # katpoint provides Target.NULL for this purpose
      if Target is not None:
        target_obj = Target("Nothing, special")
      else:
        target_obj = None
    else:
      if Target is not None:
        # Create a proper katpoint Target with reasonable coordinates
        # Format: "name, radec, RA, DEC" (RA and DEC as separate fields)
        # Using some common calibrator coordinates
        coords = {
          "PKS1934": ("19:39:25.03", "-63:42:45.63"),
          "3C286": ("13:31:08.29", "+30:30:33.0"),
          "MockTarget": ("00:00:00.0", "+00:00:00.0"),
        }
        ra, dec = coords.get(target_name, ("00:00:00.0", "+00:00:00.0"))
        target_obj = Target(f"{target_name}, radec, {ra}, {dec}")
      else:
        target_obj = target_name

    targets[indices] = target_obj

  # Convert Target objects to string representations for serialization
  # katsdptelstate can't serialize Target objects directly
  target_strings = np.empty(ntime, dtype=object)
  for i, target in enumerate(targets):
    if target is None:
      target_strings[i] = "Nothing, special"
    elif hasattr(target, "description"):
      # katpoint Target has a description attribute
      target_strings[i] = target.description
    else:
      # Fallback to string representation
      target_strings[i] = str(target)

  return {
    "Observation/target": target_strings.tolist(),  # Convert to list for serialization
    "Observation/scan_index": scan_indices.tolist(),
    "Observation/scan_state": scan_states.tolist(),
  }
