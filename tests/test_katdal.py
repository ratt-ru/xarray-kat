import katdal
import numpy as np
import xarray
from pytest_httpserver import HTTPServer

from tests.conftest import (
  SyntheticObservation,
  setup_mock_archive_server,
)


class TestKatdal:
  def test_katdal_mock_server_basic(self, httpserver: HTTPServer, tmp_path):
    """Tests that xarray-kat and katdal return the same data from the same datasource"""
    obs = SyntheticObservation("1234567890", ntime=8, nfreq=16, nants=4)
    obs.add_scan(range(0, 8), "track", "PKS1934")
    obs.save_to_directory(tmp_path)

    _ = setup_mock_archive_server(
      httpserver, tmp_path, "1234567890", require_auth=False
    )

    base_url = httpserver.url_for("/")
    rdb_url = f"{base_url}1234567890/1234567890_sdp_l0.full.rdb"

    ds = katdal.open(rdb_url)
    dt = xarray.open_datatree(rdb_url, engine="xarray-kat")

    def reorder_katdal_data(data):
      return (
        data[..., obs.corrprod_argsort]
        .reshape(obs.ntime, obs.nfreq, obs.nbl, obs.npol)
        .transpose(0, 2, 1, 3)
        .reshape(obs.ntime, obs.nbl, obs.nfreq, obs.npol)
      )

    assert len(children := list(dt.children)) == 1
    xarray_kat_vis = dt[children[0]].VISIBILITY.data
    katdal_vis = reorder_katdal_data(ds.vis[:])
    np.testing.assert_allclose(xarray_kat_vis, katdal_vis)

    xarray_kat_weight = dt[children[0]].WEIGHT.data
    katdal_weights = reorder_katdal_data(ds.weights[:])
    np.testing.assert_allclose(xarray_kat_weight, katdal_weights)

    xarray_kat_flags = dt[children[0]].FLAG.data
    katdal_flags = reorder_katdal_data(ds.flags[:])
    np.testing.assert_allclose(xarray_kat_flags, katdal_flags)

    xarray_kat_uvw = dt[children[0]].UVW.data
    katdal_uvw = np.stack([ds.u, ds.v, ds.w], axis=2)[:, obs.corrprod_argsort]
    np.testing.assert_allclose(xarray_kat_uvw, katdal_uvw[:, :: obs.npol])
