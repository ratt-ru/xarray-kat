import pytest
import xarray

CAPTURE_BLOCK = 1722588648
TOKEN = "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJrYXQtYXJjaGl2ZS5rYXQuYWMuemEiLCJhdWQiOiJhcmNoaXZlLWd3LTEua2F0LmFjLnphIiwiaWF0IjoxNzYyNzcyMDYwLCJwcmVmaXgiOlsiMTcyMjU4ODY0OCJdLCJleHAiOjE3Njc5NTYwNjAsInN1YiI6ImYzYWM4ODNmLWNjOTYtNDJjZC1hM2NjLTEzN2EyMTIwM2Y2MCIsInNjb3BlcyI6WyJyZWFkIl19.8TtPYJqoDPNow_0IzsGglZltHFhDssBMLuhUh7GZmEfItaVBR9M7-SVtbym4lQqHvwlEj1aROUfgoi4Tl1stqA"

# CAPTURE_BLOCK = "1759119102"
# TOKEN = "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJrYXQtYXJjaGl2ZS5rYXQuYWMuemEiLCJhdWQiOiJhcmNoaXZlLWd3LTEua2F0LmFjLnphIiwiaWF0IjoxNzU5OTQ2MzQ3LCJwcmVmaXgiOlsiMTc1OTExOTEwMiJdLCJleHAiOjE3NjAwMzI3NDcsInN1YiI6ImYzYWM4ODNmLWNjOTYtNDJjZC1hM2NjLTEzN2EyMTIwM2Y2MCIsInNjb3BlcyI6WyJyZWFkIl19.vKnZlQku30IjHcpYLw5vUsxIiTWKXVaW02a4fRMO1w48clzfYSCHkHz3Ei_T3V56dYeJuse5ZBKRbBkHoT3aRg"

CAPTURE_BLOCK = 1750997776
TOKEN = "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJrYXQtYXJjaGl2ZS5rYXQuYWMuemEiLCJhdWQiOiJhcmNoaXZlLWd3LTEua2F0LmFjLnphIiwiaWF0IjoxNzYzMzk5MjkyLCJwcmVmaXgiOlsiMTc1MDk5Nzc3NiJdLCJleHAiOjE3Njg1ODMyOTIsInN1YiI6ImYzYWM4ODNmLWNjOTYtNDJjZC1hM2NjLTEzN2EyMTIwM2Y2MCIsInNjb3BlcyI6WyJyZWFkIl19.J4tUAqPje-_qZxgTRgkdvKDZfoToxEiiegv_t_WSjS8Tqr6cUD3_LUJCoLvz05BHsYDFdvsU1LcJ0xwd7ImBZw"

URL = f"https://archive-gw-1.kat.ac.za/{CAPTURE_BLOCK}/{CAPTURE_BLOCK}_sdp_l0.full.rdb?token={TOKEN}"
HAVE_TOKEN = bool(TOKEN)


@pytest.mark.skipif(not HAVE_TOKEN, reason="Provide a JWT token")
def test_xarray_lazy_loading():
  dt = xarray.open_datatree(URL)
  import ipdb

  ipdb.set_trace()
  # ds = dt["1753867906_sdp_l0"].ds.isel(time=slice(0, 50), baseline_id=[1,4,5,6], frequency=slice(256, 768))
  # ds.compute()
  # breakpoint()
  dt.load()


@pytest.mark.skipif(not HAVE_TOKEN, reason="Provide a JWT token")
def test_xarray_chunked_loading():
  pytest.importorskip("dask")
  dt = xarray.open_datatree(URL, chunks={})
  breakpoint()
  dt = dt.compute()
