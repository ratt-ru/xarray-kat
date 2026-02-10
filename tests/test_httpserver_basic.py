"""Basic test to debug pytest-httpserver usage."""

from pytest_httpserver import HTTPServer


def test_basic_response(httpserver: HTTPServer):
  """Test basic HTTP server response."""
  httpserver.expect_request("/test").respond_with_data(b"Hello World")

  import urllib.request

  url = httpserver.url_for("/test")
  response = urllib.request.urlopen(url)
  content = response.read()

  assert content == b"Hello World"


def test_handler_with_file(httpserver: HTTPServer, tmp_path):
  """Test handler that serves a file."""
  from werkzeug.wrappers import Response

  # Create a test file
  test_file = tmp_path / "test.txt"
  test_file.write_bytes(b"Test file content")

  def file_handler(request):
    """Handler that serves a file."""
    with open(test_file, "rb") as f:
      content = f.read()
    return Response(content, status=200, content_type="text/plain")

  httpserver.expect_request("/file").respond_with_handler(file_handler)

  import urllib.request

  url = httpserver.url_for("/file")
  response = urllib.request.urlopen(url)
  content = response.read()

  assert content == b"Test file content"


def test_handler_with_headers(httpserver: HTTPServer):
  """Test handler that accesses request headers."""
  from werkzeug.wrappers import Response

  def header_handler(request):
    """Handler that reads headers."""
    # Access headers from the werkzeug Request object
    auth = request.headers.get("Authorization", "none")
    return Response(f"Auth: {auth}".encode(), status=200)

  httpserver.expect_request("/headers").respond_with_handler(header_handler)

  import urllib.request

  req = urllib.request.Request(
    httpserver.url_for("/headers"), headers={"Authorization": "Bearer token123"}
  )
  response = urllib.request.urlopen(req)
  content = response.read()

  print(f"Response: {content}")
  assert b"token123" in content
