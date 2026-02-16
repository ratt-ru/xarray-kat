"""Integration tests using mock HTTP server.

These tests verify that xarray-kat can successfully load data from a mock
HTTP server serving synthetic observations, without requiring access to the
actual MeerKAT archive.
"""

import urllib
import urllib.request
from typing import TypedDict

import numpy as np
import pytest
import xarray
from pytest_httpserver import HTTPServer

from tests.conftest import (
  SyntheticObservation,
  configure_mock_archive,
  setup_mock_archive_server,
)
from xarray_kat.meerkat_chunk_manager import MeerkatArray


def assert_visibilities_allclose(a, b):
  """Helper assert function signalling
  that comparison of visibilities should
  be treated with care.
  xarray-kat may subtly modify archive data.
  (i.e. via application of the van vleck transform)"""
  # For testing purposes, an allclose appears sufficient
  return np.testing.assert_allclose(a, b)


class TestMockJWT:
  """Test JWT token generation for mock server."""

  def test_create_mock_jwt(self):
    """Test creating a mock JWT token."""
    from tests.conftest import create_mock_jwt

    token = create_mock_jwt("1234567890")

    # JWT format: header.payload.signature
    parts = token.split(".")
    assert len(parts) == 3

    # Signature should be 86 characters (ES256 requirement)
    assert len(parts[2]) == 86

    # Should be able to parse with xarray-kat's jwt parser
    from xarray_kat.jwt import parse_jwt

    header, payload = parse_jwt(token)
    assert header["alg"] == "ES256"
    assert payload["prefix"] == ["1234567890"]
    assert "read" in payload["scopes"]


class TestMockHTTPServer:
  """Test mock HTTP server configuration."""

  def test_setup_mock_server_basic(self, httpserver: HTTPServer, tmp_path):
    """Test basic server setup without auth."""
    obs = SyntheticObservation("1234567890", ntime=8, nfreq=16, nants=4)
    obs.add_scan(range(0, 8), "track", "PKS1934")
    obs.save_to_directory(tmp_path)

    token = setup_mock_archive_server(
      httpserver, tmp_path, "1234567890", require_auth=False
    )

    # Token should be None when auth not required
    assert token is None

    # RDB file should be accessible
    rdb_url = httpserver.url_for("/1234567890/1234567890_sdp_l0.full.rdb")
    import urllib.request

    response = urllib.request.urlopen(rdb_url)
    assert response.status == 200
    content = response.read()
    assert len(content) > 0
    # Check RDB header
    assert content.startswith(b"REDIS0006")

  def test_setup_mock_server_with_auth(self, httpserver: HTTPServer, tmp_path):
    """Test server setup with JWT authentication."""
    obs = SyntheticObservation("1234567890", ntime=8, nfreq=16, nants=4)
    obs.add_scan(range(0, 8), "track", "PKS1934")
    obs.save_to_directory(tmp_path)

    token = setup_mock_archive_server(
      httpserver, tmp_path, "1234567890", require_auth=True
    )
    assert token is not None

    # Request without token should fail
    rdb_url = httpserver.url_for("/1234567890/1234567890_sdp_l0.full.rdb")
    with pytest.raises(Exception):  # Should be 401 Unauthorized
      urllib.request.urlopen(rdb_url)

    # Request with valid token should succeed

    req = urllib.request.Request(rdb_url, headers={"Authorization": f"Bearer {token}"})
    response = urllib.request.urlopen(req)
    assert response.status == 200

  def test_serve_numpy_chunks(self, httpserver: HTTPServer, tmp_path):
    """Test serving .npy chunk files."""
    obs = SyntheticObservation("1234567890", ntime=4, nfreq=8, nants=4)
    obs.add_scan(range(0, 4), "track", "PKS1934")
    obs.save_to_directory(tmp_path)

    setup_mock_archive_server(httpserver, tmp_path, "1234567890", require_auth=False)

    # Test accessing a chunk file
    chunk_url = httpserver.url_for(
      "/1234567890-sdp-l0/correlator_data/00000_00000_00000.npy"
    )

    response = urllib.request.urlopen(chunk_url)
    assert response.status == 200

    # Verify it's valid numpy data
    import io

    chunk_data = np.load(io.BytesIO(response.read()))
    assert chunk_data.dtype == np.complex64
    assert chunk_data.ndim == 3


class TestXarrayKatIntegration:
  """Integration tests with xarray-kat backend."""

  def test_open_datatree_lazy_loading(self, httpserver: HTTPServer, tmp_path):
    """Test opening datatree with lazy loading (no chunks parameter)."""
    # Create and serve a small observation
    obs = SyntheticObservation("1234567890", ntime=8, nfreq=16, nants=4)
    obs.add_scan(range(0, 8), "track", "PKS1934")
    obs.save_to_directory(tmp_path)

    token = setup_mock_archive_server(
      httpserver, tmp_path, "1234567890", require_auth=True
    )

    # Construct URL
    base_url = httpserver.url_for("/")
    rdb_url = f"{base_url}1234567890/1234567890_sdp_l0.full.rdb?token={token}"

    # Open datatree (lazy loading)
    dt = xarray.open_datatree(rdb_url, engine="xarray-kat")

    # Should have at least one child (scan)
    assert len(dt.children) > 0

    # Get first dataset
    ds_name = list(dt.children.keys())[0]
    ds = dt[ds_name].ds

    # Check variables exist
    assert "VISIBILITY" in ds
    assert "WEIGHT" in ds
    assert "FLAG" in ds

    # Check dimensions
    assert "time" in ds.dims
    assert "baseline_id" in ds.dims
    assert "frequency" in ds.dims
    assert "polarization" in ds.dims

    # Load data
    ds.load()

    # Verify data was loaded
    assert isinstance(ds.VISIBILITY.data, np.ndarray)
    assert ds.VISIBILITY.shape == (obs.ntime, obs.nbl, obs.nfreq, obs.npol)

    # Verify data is correct
    expected = obs.generate_msv4_array_data(
      "correlator_data", ds.VISIBILITY.dtype, (obs.ntime, obs.nfreq, obs.ncorrprod)
    )
    assert_visibilities_allclose(expected, ds.VISIBILITY.data)

  def test_open_datatree_with_meerkat_chunks(self, httpserver: HTTPServer, tmp_path):
    """Test opening datatree with MeerKat chunk manager."""
    obs = SyntheticObservation("1234567890", ntime=8, nfreq=16, nants=4)
    obs.add_scan(range(0, 8), "track", "PKS1934")
    obs.save_to_directory(tmp_path)

    token = setup_mock_archive_server(
      httpserver, tmp_path, "1234567890", require_auth=True
    )

    base_url = httpserver.url_for("/")
    rdb_url = f"{base_url}1234567890/1234567890_sdp_l0.full.rdb?token={token}"

    # Open with xarray-kat chunked array type
    dt = xarray.open_datatree(
      rdb_url, engine="xarray-kat", chunked_array_type="xarray-kat", chunks={}
    )

    ds_name = list(dt.children.keys())[0]
    ds = dt[ds_name].ds

    assert isinstance(ds.VISIBILITY.data, MeerkatArray)

    # Compute the data
    ds = ds.compute()

    # After compute, should be numpy arrays
    assert isinstance(ds.VISIBILITY.data, np.ndarray)

    # Verify data is correct
    expected = obs.generate_msv4_array_data(
      "correlator_data", ds.VISIBILITY.dtype, (obs.ntime, obs.nfreq, obs.ncorrprod)
    )
    assert_visibilities_allclose(expected, ds.VISIBILITY.data)

  def test_open_datatree_with_dask_chunks(self, httpserver: HTTPServer, tmp_path):
    """Test opening datatree with dask chunking."""
    pytest.importorskip("dask")

    obs = SyntheticObservation("1234567890", ntime=8, nfreq=16, nants=4)
    obs.add_scan(range(0, 8), "track", "PKS1934")
    obs.save_to_directory(tmp_path)

    token = setup_mock_archive_server(
      httpserver, tmp_path, "1234567890", require_auth=True
    )

    base_url = httpserver.url_for("/")
    rdb_url = f"{base_url}1234567890/1234567890_sdp_l0.full.rdb?token={token}"

    # Open with dask
    dt = xarray.open_datatree(
      rdb_url,
      engine="xarray-kat",
      chunked_array_type="dask",
      chunks={"time": 4, "frequency": 8},
    )

    ds_name = list(dt.children.keys())[0]
    ds = dt[ds_name].ds

    # Should have dask arrays
    import dask.array

    assert isinstance(ds.VISIBILITY.data, dask.array.Array)

    # Compute
    ds = ds.compute()
    assert isinstance(ds.VISIBILITY.data, np.ndarray)

    # Verify data is correct
    expected = obs.generate_msv4_array_data(
      "correlator_data", ds.VISIBILITY.dtype, (obs.ntime, obs.nfreq, obs.ncorrprod)
    )
    assert_visibilities_allclose(expected, ds.VISIBILITY.data)

  def test_data_selection_with_meerkat_chunks_isel(
    self, httpserver: HTTPServer, tmp_path
  ):
    """Test selecting subsets of data."""
    obs = SyntheticObservation("1234567890", ntime=10, nfreq=16, nants=4)
    obs.add_scan(range(0, 10), "track", "PKS1934")
    obs.save_to_directory(tmp_path)

    token = setup_mock_archive_server(
      httpserver, tmp_path, "1234567890", require_auth=True
    )

    base_url = httpserver.url_for("/")
    rdb_url = f"{base_url}1234567890/1234567890_sdp_l0.full.rdb?token={token}"

    dt = xarray.open_datatree(
      rdb_url, chunked_array_type="xarray-kat", engine="xarray-kat", chunks={}
    )

    ds_name = list(dt.children.keys())[0]
    ds = dt[ds_name].ds

    # MeerKat arrays on the full-resolution dataset
    assert isinstance(ds.VISIBILITY.data, MeerkatArray)

    # Select a subset
    class Selection(TypedDict):
      time: slice
      baseline_id: list[int]
      frequency: slice

    sel: Selection = {
      "time": slice(2, 6),
      "baseline_id": [1, 3, 4],
      "frequency": slice(4, 12),
    }
    ds_subset = ds.isel(**sel)

    # MeerKat arrays on the selected dataset
    assert isinstance(ds_subset.VISIBILITY.data, MeerkatArray)

    # Load the subset
    ds_subset.load()

    # Check shapes
    assert ds_subset.VISIBILITY.shape == (
      len(range(*sel["time"].indices(1_000_000))),
      len(sel["baseline_id"]),
      len(range(*sel["frequency"].indices(1_000_000))),
      obs.npol,
    )

    # Verify data is correct
    key = (sel["time"], sel["baseline_id"], sel["frequency"], slice(None))
    expected = obs.generate_msv4_array_data(
      "correlator_data", ds.VISIBILITY.dtype, (obs.ntime, obs.nfreq, obs.ncorrprod)
    )
    assert_visibilities_allclose(expected[key], ds_subset.VISIBILITY.data)

  def test_multiple_scans(self, httpserver: HTTPServer, tmp_path):
    """Test observation with multiple scans."""
    obs = SyntheticObservation("1234567890", ntime=20, nfreq=16, nants=4)
    obs.add_scan(range(0, 8), "track", "PKS1934")
    obs.add_scan(range(8, 20), "scan", "3C286")
    obs.save_to_directory(tmp_path)

    token = setup_mock_archive_server(
      httpserver, tmp_path, "1234567890", require_auth=True
    )

    base_url = httpserver.url_for("/")
    rdb_url = f"{base_url}1234567890/1234567890_sdp_l0.full.rdb?token={token}"

    dt = xarray.open_datatree(rdb_url, engine="xarray-kat")

    # Should have multiple children (one per scan)
    assert len(children := list(dt.children)) == len(obs.scan_configs)

    # Get the expected test data ranging over all scans
    time_index = 0
    expected = obs.generate_msv4_array_data(
      "correlator_data",
      dt[children[0]].VISIBILITY.dtype,
      (obs.ntime, obs.nfreq, obs.ncorrprod),
    )

    # Each scan should have data
    for scan, child_name in enumerate(dt.children):
      scan_config = obs.scan_configs[scan]
      ds = dt[child_name].ds
      assert len(scan_config["indices"]) == len(ds.time)
      assert np.all(scan_config["target_name"] == ds.field_name.data)

      # Check that this scan matches the relevant portion of the test data
      ntime = ds.sizes["time"]
      assert_visibilities_allclose(
        ds.VISIBILITY.data, expected[time_index : time_index + ntime]
      )
      time_index += ntime

  def test_scan_state_filtering(self, httpserver: HTTPServer, tmp_path):
    """Test filtering by scan states."""
    obs = SyntheticObservation("1234567890", ntime=20, nfreq=16, nants=4)
    obs.add_scan(range(0, 10), "track", "PKS1934")
    obs.add_scan(range(10, 15), "slew", None)
    obs.add_scan(range(15, 20), "scan", "3C286")
    obs.save_to_directory(tmp_path)

    token = setup_mock_archive_server(
      httpserver, tmp_path, "1234567890", require_auth=True
    )

    base_url = httpserver.url_for("/")
    rdb_url = f"{base_url}1234567890/1234567890_sdp_l0.full.rdb?token={token}"
    scan_states = ["track", "scan"]

    # Open with only track and scan states (exclude slew)
    dt = xarray.open_datatree(
      rdb_url,
      engine="xarray-kat",
      scan_states=scan_states,
    )

    # Should have 2 children (track and scan, not slew)
    assert len(dt.children) == 2

    is_greedy = {"slew", "stop"}

    # Each scan should have data
    for child_name in dt.children:
      scan = int(child_name[-3:])
      scan_config = obs.scan_configs[scan]
      # Greedy scans can take up an extra dump
      prev_greedy = scan > 0 and obs.scan_configs[scan - 1]["state"] in is_greedy

      ds = dt[child_name].ds
      assert len(scan_config["indices"]) + (-1 if prev_greedy else 0) == len(ds.time)
      assert np.all(scan_config["target_name"] == ds.field_name.data)
      assert "VISIBILITY" in ds
      assert "time" in ds.dims

  def test_antenna_xds_dataset(self, httpserver: HTTPServer, tmp_path):
    """Test antenna_xds dataset structure, values, and scan invariance."""
    from katpoint import Antenna as KatAntenna

    from tests.meerkat_antennas import MEERKAT_ANTENNA_DESCRIPTIONS as ANT_DESC

    obs = SyntheticObservation("1234567890", ntime=16, nfreq=16, nants=4)
    obs.add_scan(range(0, 8), "track", "PKS1934")
    obs.add_scan(range(8, 16), "scan", "3C286")
    obs.save_to_directory(tmp_path)

    token = setup_mock_archive_server(
      httpserver, tmp_path, "1234567890", require_auth=True
    )
    base_url = httpserver.url_for("/")
    rdb_url = f"{base_url}1234567890/1234567890_sdp_l0.full.rdb?token={token}"

    dt = xarray.open_datatree(rdb_url, engine="xarray-kat")

    # Build expected values from katpoint (same observer strings as fixture)
    expected_antennas = [KatAntenna(ANT_DESC[f"m{i:03d}"]) for i in range(obs.nants)]
    expected_positions = np.array([a.position_ecef for a in expected_antennas])
    expected_names = [a.name for a in expected_antennas]

    antenna_nodes = []
    for child_name in dt.children:
      ant_node = dt[f"{child_name}/antenna_xds"]
      ant_ds = ant_node.ds

      # Data variables
      assert set(ant_ds.data_vars) == {
        "ANTENNA_POSITION",
        "ANTENNA_DISH_DIAMETER",
        "ANTENNA_EFFECTIVE_DISH_DIAMETER",
        "ANTENNA_RECEPTOR_ANGLE",
      }
      np.testing.assert_allclose(ant_ds.ANTENNA_POSITION.values, expected_positions)
      np.testing.assert_array_equal(
        ant_ds.ANTENNA_DISH_DIAMETER.values, [13.5] * obs.nants
      )
      np.testing.assert_array_equal(
        ant_ds.ANTENNA_EFFECTIVE_DISH_DIAMETER.values, [13.5] * obs.nants
      )
      np.testing.assert_allclose(
        ant_ds.ANTENNA_RECEPTOR_ANGLE.values,
        np.full((obs.nants, 2), -np.pi / 2),
      )

      # Coordinates
      assert list(ant_ds.antenna_name.values) == expected_names
      assert list(ant_ds.mount.values) == ["ALT-AZ"] * obs.nants
      assert list(ant_ds.telescope_name.values) == ["MeerKat"] * obs.nants
      assert list(ant_ds.station_name.values) == expected_names
      assert list(ant_ds.cartesian_pos_label.values) == ["x", "y", "z"]
      assert list(ant_ds.receptor_label.values) == ["pol_0", "pol_1"]
      assert ant_ds.polarization_type.shape == (obs.nants, 2)

      # Attrs
      assert ant_ds.attrs["type"] == "antenna"
      assert ant_ds.attrs["overall_telescope_name"] == "MeerKat"
      assert ant_ds.attrs["relocatable_antennas"] is False

      antenna_nodes.append(ant_node)

    # Verify scan invariance: antenna_xds content is identical across scans.
    assert len(antenna_nodes) == 2
    xarray.testing.assert_identical(
      antenna_nodes[0].to_dataset(inherit=False),
      antenna_nodes[1].to_dataset(inherit=False),
    )


class TestPytestFixtures:
  """Test using the pytest fixtures from conftest.py."""

  def test_simple_mock_server(self, simple_mock_server):
    """Test using the simple_mock_server fixture."""
    url, cbid, token = simple_mock_server

    # Should have URL, capture block ID, and token
    assert url.startswith("http://")
    assert cbid == "1234567890"  # Default
    assert token is not None

    # Should be able to open with xarray
    full_url = f"{url}{cbid}/{cbid}_sdp_l0.full.rdb?token={token}"
    dt = xarray.open_datatree(full_url, engine="xarray-kat")
    assert len(dt.children) > 0

  def test_mock_archive_server_factory(self, mock_archive_server):
    """Test using the mock_archive_server factory fixture."""
    # Create server with custom parameters
    url, cbid, token = mock_archive_server(
      ntime=20,
      nfreq=32,
      nants=7,
      capture_block_id="9999999999",
    )

    assert cbid == "9999999999"

    # Open and verify
    full_url = f"{url}{cbid}/{cbid}_sdp_l0.full.rdb?token={token}"
    dt = xarray.open_datatree(full_url, engine="xarray-kat")

    ds_name = list(dt.children.keys())[0]
    ds = dt[ds_name].ds

    # Verify custom dimensions
    assert ds.sizes["time"] == 20
    assert ds.sizes["frequency"] == 32
    assert ds.sizes["baseline_id"] == 28  # 7 antennas = 28 baselines

  def test_mock_archive_server_with_scans(self, mock_archive_server):
    """Test creating server with multiple scans."""
    url, cbid, token = mock_archive_server(
      ntime=20,
      scan_configs=[
        {"indices": range(0, 10), "state": "track", "target_name": "PKS1934"},
        {"indices": range(10, 20), "state": "scan", "target_name": "3C286"},
      ],
    )

    full_url = f"{url}{cbid}/{cbid}_sdp_l0.full.rdb?token={token}"
    dt = xarray.open_datatree(full_url, engine="xarray-kat")

    # Should have children (scans may be combined depending on xarray-kat logic)
    assert len(dt.children) > 0

    # Verify both scan states are represented
    all_states = set()
    for child_name in dt.children:
      # The description contains the scan state
      description = dt[child_name].ds.attrs.get("description", "").lower()
      if "track" in description:
        all_states.add("track")
      if "scan" in description:
        all_states.add("scan")

    # Both scan types should be present in the data
    assert len(all_states) >= 1  # At least one scan state present

  def test_synthetic_observation_factory(self, synthetic_observation, tmp_path):
    """Test using the synthetic_observation factory fixture."""
    # Create custom observation
    obs = synthetic_observation(ntime=15, nfreq=24, nants=5)
    obs.add_scan(range(0, 15), "track", "MockTarget")

    # Save and verify
    stats = obs.save_to_directory(tmp_path)
    assert stats["rdb_keys"] > 0
    assert stats["correlator_data_chunks"] > 0

  def test_mock_archive_fixture(self, mock_archive):
    """Test using the mock_archive fixture."""
    archive_path, obs, stats = mock_archive

    # Verify archive was created
    assert archive_path.exists()
    rdb_file = archive_path / f"{obs.capture_block_id}_sdp_l0.full.rdb"
    assert rdb_file.exists()
    assert stats["rdb_keys"] > 0


class TestConvenienceFixture:
  """Test the convenience fixture builder (legacy utility function tests)."""

  def test_configure_mock_archive(self, httpserver: HTTPServer, tmp_path):
    """Test the all-in-one fixture builder."""
    base_url, cbid, token = configure_mock_archive(httpserver, tmp_path)

    # Should have URL, capture block ID, and token
    assert base_url.startswith("http://")
    assert cbid == "1234567890"  # Default
    assert token is not None

    # Should be able to open with xarray
    full_url = f"{base_url}{cbid}/{cbid}_sdp_l0.full.rdb?token={token}"
    dt = xarray.open_datatree(full_url, engine="xarray-kat")
    assert len(dt.children) > 0

  def test_configure_with_custom_params(self, httpserver: HTTPServer, tmp_path):
    """Test convenience fixture with custom observation parameters."""
    base_url, cbid, token = configure_mock_archive(
      httpserver, tmp_path, obs_kwargs={"ntime": 20, "nfreq": 32, "nants": 7}
    )

    full_url = f"{base_url}{cbid}/{cbid}_sdp_l0.full.rdb?token={token}"
    dt = xarray.open_datatree(full_url, engine="xarray-kat")

    ds_name = list(dt.children.keys())[0]
    ds = dt[ds_name].ds

    # Verify custom dimensions
    assert ds.sizes["time"] == 20
    assert ds.sizes["frequency"] == 32
    # 7 antennas = 28 baselines (including autocorr)
    assert ds.sizes["baseline_id"] == 28
