from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Iterable
from urllib.parse import SplitResult, parse_qs, urlsplit
from urllib.request import urlopen

from xarray import DataTree
from xarray.backends import BackendEntrypoint
from xarray.backends.common import AbstractDataStore

if TYPE_CHECKING:
  from io import BufferedIOBase


from xarray_kat.datatree_factory import GroupFactory
from xarray_kat.telstate import telstate_factory


class KatStore(AbstractDataStore):
  """Store for reading from a MeerKAT data source"""


class KatEntryPoint(BackendEntrypoint):
  open_dataset_parameters = ["filename_or_obj, capture_block_id", "stream_name"]
  description = "Opens a MeerKAT data source"
  supports_groups = True

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
    capture_block_id: str | None = None,
    stream_name: str | None = None,
  ):
    group_dicts = self.open_groups_as_dict(
      filename_or_obj, drop_variables=drop_variables
    )
    return DataTree.from_dict(group_dicts)

  def open_groups_as_dict(
    self,
    filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
    *,
    drop_variables: str | Iterable[str] | None = None,
    capture_block_id: str | None = None,
    stream_name: str | None = None,
  ) -> Dict[str, Any]:
    url = str(filename_or_obj)
    urlbits = urlsplit(url)
    assert urlbits.scheme in {"http", "https"}

    qs = parse_qs(urlbits.query)
    token = qs.get("token", [None])[0]

    with urlopen(url) as response:
      telstate = telstate_factory(response, capture_block_id, stream_name)

    endpoint = SplitResult(urlbits.scheme, urlbits.netloc, "", "", "").geturl()
    groups = GroupFactory.make(telstate, endpoint, token)
    return groups
