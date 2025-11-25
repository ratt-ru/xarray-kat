from xarray_kat.multiton import Multiton
from xarray_kat.third_party.vendored.katdal.datasources_minimal import (
  TelstateDataSource,
)
from xarray_kat.third_party.vendored.katdal.sensordata import SensorCache
from xarray_kat.third_party.vendored.katdal.visdatav4_minimal import VisibilityDataV4


def sensor_cache_factory(datasource: Multiton[TelstateDataSource]) -> SensorCache:
  """Generate a katdal SensorCache.

  We call the katdal Dataset and appropriate the sensor member
  """
  return VisibilityDataV4(datasource.instance).sensor
