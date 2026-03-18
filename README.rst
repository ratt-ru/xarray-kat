Direct xarray views over the MeerKAT archive
============================================

This package present an xarray view over observations in the MeerKAT archive.

Required Reading
----------------

You'll need some familiarity with xarray_. In particular:

- `Indexing and selecting data <xarray-indexing_>`_
- `Lazy Loading behaviour <xarray-lazy-indexing_>`_


Opening a dataset
-----------------

Use ``xarray.open_datatree`` with ``engine="xarray-kat"`` (or let xarray infer
the engine automatically from the URL).  The call returns a
:class:`xarray.DataTree` whose children are the individual scans of the
observation.

.. code-block:: python

  import xarray_kat  # registers the "xarray-kat" engine
  import xarray

  token = "eyFILLMEIN"
  capture_block_id = 1234567890
  url = (
      f"https://archive-gw-1.kat.ac.za/{capture_block_id}"
      f"/{capture_block_id}_sdp_l0.full.rdb?token={token}"
  )

  dt = xarray.open_datatree(url, engine="xarray-kat", chunked_array_type="xarray-kat", chunks={})

Each child node is keyed ``"<capture_block_id>_<stream_name>/<scan_index>"``
and exposes the MSv4-style data variables:

- ``VISIBILITY`` — complex correlator data, shape ``(time, baseline_id, frequency, polarization)``
- ``WEIGHT`` — per-sample weights, same shape as ``VISIBILITY``
- ``FLAG`` — per-sample flags, same shape as ``VISIBILITY``
- ``UVW`` — baseline UVW coordinates, shape ``(time, baseline_id, uvw)``


Parameters
----------

All keyword arguments below are passed through ``xarray.open_datatree``.

``applycal`` : ``str`` or list of ``str``, default ``""``
    Calibration products to apply.  Use ``"all"`` to apply every available
    product, an explicit list such as ``["l1.G", "l1.B"]`` to apply specific
    products, or ``""`` to skip calibration entirely.

``scan_states`` : iterable of ``str``, default ``("scan", "track")``
    Only scans whose activity-sensor state appears in this collection are
    included in the tree.  Pass e.g. ``("track",)`` to keep only tracking
    scans.

``uvw_sign_convention`` : ``"casa"`` or ``"fourier"``, default ``"casa"``
    Sign convention for UVW coordinates.  ``"casa"`` follows the
    ``antenna2 - antenna1`` convention used by CASA and most radio-astronomy
    software.  ``"fourier"`` uses the opposite sign.

``van_vleck`` : ``"off"`` or ``"autocorr"``, default ``"off"``
    Van Vleck correction for the MeerKAT F-engine quantisation distortion.
    ``"autocorr"`` corrects autocorrelation amplitudes using the built-in
    lookup table; ``"off"`` leaves the data unchanged.

``preferred_chunks`` : dict, optional
    Preferred chunk sizes along named dimensions, e.g.
    ``{"time": 4, "frequency": 512}``.  These are hints; the engine uses the
    natural archive chunking where possible.

``capture_block_id`` : ``str``, optional
    Override the capture-block ID inferred from the RDB file.  Rarely needed
    in normal use; useful when the ID embedded in the file differs from the
    one in the URL.

``stream_name`` : ``str``, optional
    Override the data-stream name inferred from the RDB file (e.g.
    ``"sdp_l0"``).  Useful when an observation contains multiple streams and
    you want to open a specific one.


Example Usage
-------------

Load a small observation entirely into memory:

.. code-block:: python

  import xarray_kat
  import xarray

  token = "eyFILLMEIN"
  capture_block_id = 1234567890
  url = (
      f"https://archive-gw-1.kat.ac.za/{capture_block_id}"
      f"/{capture_block_id}_sdp_l0.full.rdb?token={token}"
  )

  dt = xarray.open_datatree(url, chunked_array_type="xarray-kat", chunks={})
  dt.load()

Select a subset of the data before loading:

.. code-block:: python

  ds = dt[f"{capture_block_id}_sdp_l0/0"].ds
  ds = ds.isel(
      time=slice(10, 20),
      baseline_id=[1, 20, 30, 31, 32, 50],
      frequency=slice(256, 768),
  )
  ds.load()

Apply calibration solutions and Van Vleck correction:

.. code-block:: python

  dt = xarray.open_datatree(
      url,
      chunked_array_type="xarray-kat",
      chunks={},
      applycal="all",
      van_vleck="autocorr",
  )

If dask is installed, request dask-backed chunks along specific dimensions:

.. code-block:: python

  # Natural (archive) chunking — most efficient
  dt = xarray.open_datatree(url, chunks={})
  dt = dt.compute()

  # Custom chunking — may cause repeated archive requests for overlapping chunks;
  # prefer rechunking on top of natural chunks, or use a cache pool
  dt = xarray.open_datatree(
      url, chunks={"time": 20, "baseline_id": 155, "frequency": 256}
  )
  dt = dt.compute()

.. _xarray: https://xarray.dev/
.. _xarray-indexing: https://docs.xarray.dev/en/latest/user-guide/indexing.html
.. _xarray-lazy-indexing: https://docs.xarray.dev/en/latest/internals/internal-design.html#lazy-indexing-classes
