Changelog
=========

X.Y.Z (DD-MM-YYY)
-----------------

* Move reshape and transpose functionality out of ``tensorstore.virtual_chunked`` into ``xarray.BackedArray.__getitem__`` (:pr:`1`).
* Delegate creation of unpickleable ``tensorstore.virtual_chunked`` stores into a pickleable ``MeerkatStoreProvider`` (:pr:`1`)
