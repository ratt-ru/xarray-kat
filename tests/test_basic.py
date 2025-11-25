import pytest
import xarray

CAPTURE_BLOCK = 1234567890
TOKEN = ""


URL = f"https://archive-gw-1.kat.ac.za/{CAPTURE_BLOCK}/{CAPTURE_BLOCK}_sdp_l0.full.rdb?token={TOKEN}"
HAVE_TOKEN = bool(TOKEN)


@pytest.mark.skipif(not HAVE_TOKEN, reason="Provide a JWT token")
def test_xarray_lazy_loading():
  dt = xarray.open_datatree(URL)
  dt.load()


@pytest.mark.skipif(not HAVE_TOKEN, reason="Provide a JWT token")
def test_xarray_chunked_loading():
  pytest.importorskip("dask")
  dt = xarray.open_datatree(URL, chunks={})
  dt = dt.compute()
