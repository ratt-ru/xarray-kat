from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Iterable
from urllib.parse import SplitResult, parse_qs, urlsplit

from xarray import DataTree
from xarray.backends import BackendEntrypoint
from xarray.backends.common import AbstractDataStore

if TYPE_CHECKING:
  from io import BufferedIOBase


from xarray_kat.datatree_factory import DataTreeFactory
from xarray_kat.katdal_types import TelstateDataProducts, TelstateDataSource
from xarray_kat.multiton import Multiton
from xarray_kat.types import VanVleckLiteralType


class KatStore(AbstractDataStore):
  """Store for reading from a MeerKAT data source"""


class KatEntryPoint(BackendEntrypoint):
  open_dataset_parameters = ["filename_or_obj, capture_block_id", "stream_name"]
  description = "Opens a MeerKAT data source"
  supports_groups = True
  url = "https://github.com/ratt-ru/xarray-kat"

  def guess_can_open(
    self,
    filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
  ) -> bool:
    if not isinstance(filename_or_obj, (str, os.PathLike)):
      return False

    urlbits = urlsplit(str(filename_or_obj))

    return (
      "kat.ac.za" in urlbits.netloc
      and urlbits.path.endswith("rdb")
      and (
        urlbits.scheme == "http"
        or (urlbits.scheme == "https" and "token" in urlbits.query)
      )
    )

  def open_datatree(
    self,
    filename_or_obj,
    *,
    drop_variables=None,
    applycal: str | Iterable[str] = "",
    scan_states: Iterable[str] = ("scan", "track"),
    capture_block_id: str | None = None,
    stream_name: str | None = None,
    van_vleck: VanVleckLiteralType = "off",
  ):
    group_dicts = self.open_groups_as_dict(
      filename_or_obj,
      drop_variables=drop_variables,
      applycal=applycal,
      scan_states=scan_states,
      capture_block_id=capture_block_id,
      stream_name=stream_name,
      van_vleck=van_vleck,
    )
    return DataTree.from_dict(group_dicts)

  def open_groups_as_dict(
    self,
    filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
    *,
    drop_variables: str | Iterable[str] | None = None,
    applycal: str | Iterable[str] = "",
    scan_states: Iterable[str] = ("scan", "track"),
    capture_block_id: str | None = None,
    stream_name: str | None = None,
    van_vleck: VanVleckLiteralType = "off",
  ) -> Dict[str, Any]:
    url = str(filename_or_obj)
    urlbits = urlsplit(url)
    assert urlbits.scheme in {"http", "https"}

    token = parse_qs(urlbits.query).get("token", [None])[0]
    datasource = Multiton(
      TelstateDataSource.from_url,
      url,
      chunk_store=None,
      capture_block_id=capture_block_id,
      stream_name=stream_name,
      van_vleck=van_vleck,
    )

    telstate_data_products = Multiton(
      TelstateDataProducts, datasource, applycal=applycal
    )
    endpoint = SplitResult(urlbits.scheme, urlbits.netloc, "", "", "").geturl()
    group_factory = DataTreeFactory(
      telstate_data_products, applycal, scan_states, van_vleck, endpoint, token
    )
    return group_factory.create()
