from pytest_httpserver import HTTPServer

import xarray
import xradio  # noqa
import xradio.measurement_set  # noqa. Required to register types for check_datatree
from xradio.schema.check import check_datatree

from tests.conftest import (
  SyntheticObservation,
  setup_mock_archive_server,
)


class TestXRadioValidation:
  def test_validation(self, httpserver: HTTPServer, tmp_path):
    """Tests that datatrees produced by xarray-kat are validated by the
    xradio schema checker"""
    obs = SyntheticObservation("1234567890", ntime=8, nfreq=16, nants=4)
    obs.add_scan(range(0, 8), "track", "PKS1934")
    obs.save_to_directory(tmp_path)

    _ = setup_mock_archive_server(
      httpserver, tmp_path, "1234567890", require_auth=False
    )

    base_url = httpserver.url_for("/")
    rdb_url = f"{base_url}1234567890/1234567890_sdp_l0.full.rdb"

    dt = xarray.open_datatree(rdb_url, engine="xarray-kat")
    issues = check_datatree(dt)
    assert not issues

