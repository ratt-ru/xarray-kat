class InvalidJwtToken(ValueError):
  """Raised if a JWT Token is invalid"""


class TelstateKeyError(ValueError):
  """Raised if some required key is not present in the telescope state"""
