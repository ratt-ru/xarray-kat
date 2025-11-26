Changelog
=========

X.Y.Z (DD-MM-YYY)
-----------------
* Add missing SensorGetter import (:pr:`17`)
* Correct typo in README (:pr:`14`)
* Add missing katpoint dependency (:pr:`13`)
* Split MeerKAT observations by scan (:pr:`11`)
* Convert slices into an immutable key value (:pr:`10`)
* Vendor katdal telstate and SensorCache code (:pr:`9`)
* Change Multiton instance lock to be re-entrant (:pr:`8`)
* Zero missing correlator data (visibility) data (:pr:`7`)
* Refactor code base to use multitons for tensorstore http stores (:pr:`6`)
* Move reshape and transpose functionality out of ``tensorstore.virtual_chunked`` into ``xarray.BackedArray.__getitem__`` (:pr:`1`).
* Delegate creation of unpickleable ``tensorstore.virtual_chunked`` stores into a pickleable ``MeerkatStoreProvider`` (:pr:`1`)
