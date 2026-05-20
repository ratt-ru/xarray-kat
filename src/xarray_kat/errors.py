class InvalidJwtToken(ValueError):
  """Raised if a JWT Token is invalid"""


class TelstateKeyError(ValueError):
  """Raised if some required key is not present in the telescope state"""


class IgnoredArgument(UserWarning):
  """Issued when keyword arguments are passed that this backend does not support."""
