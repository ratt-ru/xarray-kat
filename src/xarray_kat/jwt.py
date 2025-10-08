import json
from base64 import b64decode
from typing import Dict, Tuple

from xarray_kat.errors import InvalidJwtToken


def parse_jwt(token) -> Tuple[Dict, Dict]:
  """Poor man's JWT token parser made from json and base64"""
  try:
    encoded_header, encoded_payload, encoded_signature = token.split(".")
  except ValueError:
    raise InvalidJwtToken(
      f"Malformed JWT token {token[:8]}...{token[-8:]}. It did not contain 2 dots."
    )

  header = json.loads(b64decode(encoded_header))
  payload = json.loads(b64decode(encoded_payload))

  if header.get("alg") == "ES256" and (sig_len := len(encoded_signature)) != 86:
    raise InvalidJwtToken(f"Encoded JWT signature length {sig_len} != 86")

  return header, payload
