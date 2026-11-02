from io import BytesIO

import numpy as np

from xarray_kat.stores.base_store import read_array


def npy_bytes(array: np.ndarray) -> bytes:
  buffer = BytesIO()
  np.save(buffer, array)
  return buffer.getvalue()


def test_read_array_decodes_without_copying_the_payload():
  expected = (np.arange(24) + 1j * np.arange(24)).astype(np.complex64).reshape(2, 3, 4)
  raw = npy_bytes(expected)

  result = read_array(raw)

  np.testing.assert_array_equal(result, expected)
  assert np.shares_memory(result, np.frombuffer(raw, np.uint8))


def test_read_array_rejects_a_truncated_payload():
  raw = npy_bytes(np.ones((2, 3), np.float32))

  assert read_array(raw[:-1]) is None
