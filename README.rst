Direct xarray views over the MeerKAT archive
============================================

This archive present an array view over observations in the MeerKAT archive.

Required Reading
----------------

You'll need some familiarity with xarray_. In particular:

- `Indexing and selecting data <xarray-indexing_>`_
- `Lazy Loading behaviour <xarray-lazy-indexing_>`_


Example Usage
-------------

At a basic level, one can use xarray's selection and lazy loading mechanisms to interact with
the data:

.. code-block:: python

  import xarray_kat
  import xarray

  token = "eyFILLMEIN"
  capture_block_id = 123456789
  url = f"https://archive-gw-1.kat.ac.za/{capture_block_id}/{capture_block_id}_sdp_l0.full.rdb?token={token}"

  # If the dataset is small you may be able to load it all in at once
  dt = xarray.open_datatree(url)
  dt.load()

  # Otherwise one can select a small partition of the data
  # that can fit in memory and interact with that
  ds = dt["123456789_sdp_l0"].ds
  ds = ds.isel(time=slice(10, 20), baseline_id=[1, 20, 30, 31, 32, 50], frequency=slice(256, 768))
  ds.load()

If dask is installed, one can request chunking along dimensions:

.. code-block:: python

  import xarray_kat
  import xarray

  token = "eyFILLMEIN"
  capture_block_id = 123456789
  url = f"https://archive-gw-1.kat.ac.za/{capture_block_id}/{capture_block_id}_sdp_l0.full.rdb?token={token}"

  # This specifies the natural chunking of the
  # underlying store
  dt = xarray.open_datatree(url, chunks={})
  dt = dt.compute()

  # More exotic chunking can be selected, but
  # as this pattern does not match the natural
  # chunking, it results in repeated requests for
  # the same data. It may be better to use a
  # dask.rechunk operation ontop of the natural
  # chunking, or use cache pools to ameliorate this
  dt = xarray.open_datatree(url, chunks={"time": 20, "baseline_id": 155, "frequency": 256})
  dt = dt.compute()

.. _xarray: https://xarray.dev/
.. _xarray-indexing: https://docs.xarray.dev/en/latest/user-guide/indexing.html
.. _xarray-lazy-indexing: https://docs.xarray.dev/en/latest/internals/internal-design.html#lazy-indexing-classes