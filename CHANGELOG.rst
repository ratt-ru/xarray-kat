Changelog
=========

X.Y.Z (DD-MM-YYY)
-----------------
* Add ``KatEntryPoint.url`` class variable (:pr:`24`)
* Support selection on scan states (:pr:`23`)
* Engage Multiton instance if it has already been created (:pr:`20`)
* Vendor van Fleck transformation code from katdal (:pr:`19`, :pr:`22`)
* Pass ``capture_block_id`` and ``stream_name`` to ``TelstateDataSource.from_url`` (:pr:`18`, :pr:`21`)
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
