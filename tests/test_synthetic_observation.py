"""Tests for synthetic observation generation.

These tests verify that the mock data generation infrastructure works correctly
and produces valid telstate and array data.
"""

from pathlib import Path

import numpy as np
import pytest
from katsdptelstate import TelescopeState

from fixtures import SyntheticObservation, create_sensor_data, dict_to_rdb


class TestRDBGeneration:
  """Test RDB file generation without Redis."""

  def test_dict_to_rdb_simple(self, tmp_path):
    """Test writing a simple dictionary to RDB."""
    data = {
      "string_value": "hello world",
      "int_value": 42,
      "float_value": 3.14159,
      "list_value": [1, 2, 3],
      "dict_value": {"nested": "value"},
    }

    rdb_path = tmp_path / "test.rdb"
    n_keys = dict_to_rdb(data, rdb_path)

    assert n_keys == 5
    assert rdb_path.exists()

    # Verify we can read it back
    ts = TelescopeState()
    ts.load_from_file(rdb_path)
    assert ts.get("string_value") == "hello world"
    assert ts.get("int_value") == 42
    assert abs(ts.get("float_value") - 3.14159) < 1e-6
    assert ts.get("list_value") == [1, 2, 3]
    assert ts.get("dict_value") == {"nested": "value"}

  def test_dict_to_rdb_with_numpy(self, tmp_path):
    """Test writing numpy arrays to RDB."""
    data = {
      "numpy_array": np.array([1, 2, 3, 4, 5]),
      "numpy_2d": np.arange(12).reshape(3, 4),
    }

    rdb_path = tmp_path / "test_numpy.rdb"
    n_keys = dict_to_rdb(data, rdb_path)

    assert n_keys == 2

    ts = TelescopeState()
    ts.load_from_file(rdb_path)
    np.testing.assert_array_equal(ts.get("numpy_array"), data["numpy_array"])
    np.testing.assert_array_equal(ts.get("numpy_2d"), data["numpy_2d"])


class TestSensorDataGeneration:
  """Test sensor data generation."""

  def test_create_sensor_data_default(self):
    """Test sensor data creation with default single scan."""
    sensor_data = create_sensor_data(ntime=10)

    assert "Observation/target" in sensor_data
    assert "Observation/scan_index" in sensor_data
    assert "Observation/scan_state" in sensor_data

    # Sensor data is now returned as lists
    assert len(sensor_data["Observation/target"]) == 10
    assert len(sensor_data["Observation/scan_index"]) == 10
    assert len(sensor_data["Observation/scan_state"]) == 10

    # All should be in same scan
    assert all(idx == 0 for idx in sensor_data["Observation/scan_index"])
    assert all(state == "track" for state in sensor_data["Observation/scan_state"])

  def test_create_sensor_data_multiple_scans(self):
    """Test sensor data with multiple scans."""
    scan_configs = [
      {"indices": range(0, 5), "state": "track", "target_name": "PKS1934"},
      {"indices": range(5, 8), "state": "slew", "target_name": None},
      {"indices": range(8, 15), "state": "scan", "target_name": "3C286"},
    ]

    sensor_data = create_sensor_data(ntime=15, scan_configs=scan_configs)

    # Check scan indices (data is now in list form)
    scan_idx = sensor_data["Observation/scan_index"]
    assert all(idx == 0 for idx in scan_idx[:5])
    assert all(idx == 1 for idx in scan_idx[5:8])
    assert all(idx == 2 for idx in scan_idx[8:])

    # Check scan states
    scan_state = sensor_data["Observation/scan_state"]
    assert all(state == "track" for state in scan_state[:5])
    assert all(state == "slew" for state in scan_state[5:8])
    assert all(state == "scan" for state in scan_state[8:])


class TestSyntheticObservation:
  """Test synthetic observation generation."""

  def test_initialization(self):
    """Test basic initialization."""
    obs = SyntheticObservation(
      capture_block_id="1234567890", ntime=10, nfreq=16, nants=4
    )

    assert obs.capture_block_id == "1234567890"
    assert obs.ntime == 10
    assert obs.nfreq == 16
    assert obs.nants == 4
    assert obs.npol == 4
    assert obs.nbl == 10  # 4 * 5 / 2
    assert obs.ncorrprod == 40  # 10 * 4

  def test_initialization_invalid_antennas(self):
    """Test that fewer than 2 antennas raises error."""
    with pytest.raises(ValueError, match="at least 2 antennas"):
      SyntheticObservation(capture_block_id="123", nants=1)

  def test_bls_ordering(self):
    """Test correlation products generation."""
    obs = SyntheticObservation(capture_block_id="123", nants=3)

    # 3 antennas: 3 * 4 / 2 = 6 baselines
    # 4 polarizations = 24 correlation products
    assert len(obs.bls_ordering) == 24

    # Check format
    assert obs.bls_ordering[0] == ["m000h", "m000h"]  # Autocorr
    assert obs.bls_ordering[1] == ["m000h", "m000v"]
    assert obs.bls_ordering[2] == ["m000v", "m000h"]
    assert obs.bls_ordering[3] == ["m000v", "m000v"]

  def test_telstate_dict_structure(self):
    """Test telstate dictionary has required keys."""
    obs = SyntheticObservation(
      capture_block_id="1234567890", ntime=10, nfreq=16, nants=4
    )

    telstate = obs.create_telstate_dict()

    # Check timing keys
    assert "sync_time" in telstate
    assert "first_timestamp" in telstate
    assert "int_time" in telstate

    # Check frequency keys
    assert "center_freq" in telstate
    assert "bandwidth" in telstate
    assert "n_chans" in telstate
    assert telstate["n_chans"] == 16

    # Check antenna keys
    assert "sub_pool_resources" in telstate
    assert "bls_ordering" in telstate
    assert "m000_observer" in telstate

    # Check chunk_info
    assert "chunk_info" in telstate
    assert "correlator_data" in telstate["chunk_info"]
    assert "flags" in telstate["chunk_info"]
    assert "weights" in telstate["chunk_info"]
    assert "weights_channel" in telstate["chunk_info"]

    # Verify shapes
    chunk_info = telstate["chunk_info"]
    assert chunk_info["correlator_data"]["shape"] == (10, 16, 40)
    assert chunk_info["weights_channel"]["shape"] == (10, 16)

  def test_add_scan(self):
    """Test adding scans to observation."""
    obs = SyntheticObservation(capture_block_id="123", ntime=20)

    obs.add_scan(range(0, 10), "track", "PKS1934")
    obs.add_scan(range(10, 20), "scan", "3C286")

    assert len(obs.scan_configs) == 2
    assert obs.scan_configs[0]["state"] == "track"
    assert obs.scan_configs[1]["target_name"] == "3C286"

  def test_generate_array_data(self):
    """Test synthetic array data generation."""
    obs = SyntheticObservation(capture_block_id="123", ntime=8, nfreq=16, nants=4)

    # Test correlator data
    vis_data = obs.generate_array_data(
      "correlator_data", np.complex64, (8, 16, 40)
    )
    assert vis_data.shape == (8, 16, 40)
    assert vis_data.dtype == np.complex64

    # Test flags
    flag_data = obs.generate_array_data("flags", np.uint8, (8, 16, 40))
    assert flag_data.shape == (8, 16, 40)
    assert flag_data.dtype == np.uint8
    # Should be mostly zeros (unflagged)
    assert np.sum(flag_data == 0) > 0.9 * flag_data.size

    # Test weights
    weight_data = obs.generate_array_data("weights", np.uint8, (8, 16, 40))
    assert weight_data.shape == (8, 16, 40)
    assert weight_data.dtype == np.uint8
    assert np.all(weight_data == 200)

    # Test channel weights
    chan_weight_data = obs.generate_array_data(
      "weights_channel", np.float32, (8, 16)
    )
    assert chan_weight_data.shape == (8, 16)
    assert chan_weight_data.dtype == np.float32
    assert np.all(chan_weight_data > 0.5)
    assert np.all(chan_weight_data <= 1.0)

  def test_save_to_directory(self, tmp_path):
    """Test saving complete observation to disk."""
    obs = SyntheticObservation(
      capture_block_id="1234567890", ntime=8, nfreq=16, nants=4
    )
    obs.add_scan(range(0, 4), "track", "PKS1934")
    obs.add_scan(range(4, 8), "scan", "3C286")

    stats = obs.save_to_directory(tmp_path)

    # Check statistics
    assert stats["rdb_keys"] > 0
    assert stats["correlator_data_chunks"] > 0
    assert stats["flags_chunks"] > 0
    assert stats["weights_chunks"] > 0
    assert stats["weights_channel_chunks"] > 0

    # Check RDB file exists and is valid
    rdb_path = tmp_path / "1234567890_sdp_l0.full.rdb"
    assert rdb_path.exists()

    ts = TelescopeState()
    ts.load_from_file(rdb_path)
    assert ts.get("chunk_info") is not None

    # Check array directories exist
    prefix = "1234567890-sdp-l0"
    for array_name in ["correlator_data", "flags", "weights", "weights_channel"]:
      array_dir = tmp_path / prefix / array_name
      assert array_dir.exists()
      assert array_dir.is_dir()

      # Check that .npy files exist
      npy_files = list(array_dir.glob("*.npy"))
      assert len(npy_files) > 0

      # Check file naming convention
      for npy_file in npy_files:
        if array_name == "weights_channel":
          # 2D array: TTTTT_FFFFF.npy
          assert len(npy_file.stem.split("_")) == 2
        else:
          # 3D array: TTTTT_FFFFF_CCCCC.npy
          assert len(npy_file.stem.split("_")) == 3

  def test_load_saved_chunks(self, tmp_path):
    """Test that saved chunks can be loaded and have correct data."""
    obs = SyntheticObservation(
      capture_block_id="1234567890", ntime=4, nfreq=8, nants=4
    )
    obs.save_to_directory(tmp_path)

    # Load a chunk and verify
    prefix = "1234567890-sdp-l0"
    vis_dir = tmp_path / prefix / "correlator_data"
    chunk_files = sorted(vis_dir.glob("*.npy"))

    assert len(chunk_files) > 0

    # Load first chunk
    chunk_data = np.load(chunk_files[0])

    # Verify shape (should be a chunk, not full array)
    assert chunk_data.ndim == 3
    assert chunk_data.shape[0] <= obs.ntime
    assert chunk_data.shape[1] <= obs.nfreq
    assert chunk_data.shape[2] == obs.ncorrprod
    assert chunk_data.dtype == np.complex64


class TestIntegration:
  """Integration tests using synthetic observations."""

  def test_end_to_end_small_observation(self, tmp_path):
    """Test creating and verifying a small observation end-to-end."""
    # Create a minimal observation
    obs = SyntheticObservation(
      capture_block_id="1750997776",  # Use realistic ID
      ntime=10,
      nfreq=16,
      nants=4,
      int_time=8.0,
    )
    obs.add_scan(range(0, 10), "track", "MockTarget")

    # Save to disk
    stats = obs.save_to_directory(tmp_path)

    # Verify telstate can be loaded
    rdb_path = tmp_path / "1750997776_sdp_l0.full.rdb"
    ts = TelescopeState()
    ts.load_from_file(rdb_path)

    # Verify key metadata
    assert ts.get("int_time") == 8.0
    assert ts.get("n_chans") == 16
    assert len(ts.get("bls_ordering")) == 40  # 4 ants = 10 bl * 4 pol

    # Verify chunk info matches array chunks
    chunk_info = ts.get("chunk_info")
    for array_name in ["correlator_data", "flags", "weights", "weights_channel"]:
      array_meta = chunk_info[array_name]
      shape = array_meta["shape"]
      chunks = array_meta["chunks"]

      # Verify chunks partition the shape
      for dim, (s, c) in enumerate(zip(shape, chunks)):
        assert sum(c) == s, f"{array_name} dim {dim}: sum(chunks)={sum(c)} != shape={s}"

    print(f"\n✓ Created observation with {stats['rdb_keys']} telstate keys")
    print(f"✓ Generated {stats['correlator_data_chunks']} visibility chunks")
    print(f"✓ Total size: ~{sum(stats.values()) * 1000} bytes (mock data)")
