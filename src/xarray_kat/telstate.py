from io import BytesIO, IOBase

from katsdptelstate import TelescopeState

from xarray_kat.errors import TelstateKeyError


def telstate_factory(
  telstate_bytes: IOBase | bytes,
  capture_block_id: None | str = None,
  stream_name: None | str = None,
) -> TelescopeState:
  """Creates a TelescopeState from ``telescope_bytes``
  with a view over

  1. capture_block_id / stream_name
  2. capture_block_id
  3. stream_name
  """
  fp: IOBase
  if isinstance(telstate_bytes, bytes):
    fp = BytesIO(telstate_bytes)
  else:
    fp = telstate_bytes

  telstate = TelescopeState()
  telstate.load_from_file(fp)

  if not capture_block_id:
    try:
      capture_block_id = telstate["capture_block_id"]
    except KeyError:
      raise TelstateKeyError(
        "'capture_block_id' not found in telstate. Pass it as a manual argument."
      )

  if not stream_name:
    try:
      stream_name = telstate["stream_name"]
    except KeyError:
      raise TelstateKeyError(
        "'stream_name' not found in telstate. Pass it as a manual argument."
      )

  telstate = telstate.view(stream_name)
  telstate = telstate.view(capture_block_id)
  capture_stream = telstate.join(capture_block_id, stream_name)
  telstate = telstate.view(capture_stream)
  return telstate
