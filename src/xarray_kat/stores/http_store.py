import logging
from typing import Collection, Dict

import tensorstore as ts

log = logging.getLogger(__name__)


def http_spec(
  endpoint: str, path: str, token: str | None
) -> Dict[str, Collection[str]]:
  """Creates a spec defining an http specification for accessing
  the MeerKAT HTTP archive.

  Args:
    endpoint: the http(s) endpoint
    path: Relative path from the endpoint
    token: The JWT token, if available

  Returns:
    Tensorstore kvstore specification
  """
  spec: Dict[str, Collection[str]] = {
    "driver": "http",
    "base_url": endpoint,
    "path": path,
  }

  if token:
    spec["headers"] = [f"Authorization: Bearer {token}"]

  return spec


def http_store_factory(
  endpoint: str,
  path: str,
  token: str | None = None,
  context: ts.Context | None = None,
) -> ts.TensorStore:
  """Creates an http(s) tensorstore referencing the specified
  endpoint and path.

  A jwt token is required if the endpoint is an https endpoint."""
  spec = http_spec(endpoint, path, token)
  return ts.KvStore.open(spec, context=context).result()
