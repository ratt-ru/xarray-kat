from katsdptelstate import TelescopeState

from xarray_kat.multiton import Multiton
from xarray_kat.third_party.vendored.katdal.applycal_minimal import CorrectionParams
from xarray_kat.third_party.vendored.katdal.datasources_minimal import (
  TelstateDataSource,
)
from xarray_kat.third_party.vendored.katdal.sensordata import SensorCache
from xarray_kat.third_party.vendored.katdal.vis_flags_weights_minimal import (
  AutoCorrelationIndices,  # noqa: F401
  corrprod_to_autocorr,  # noqa: F401
)
from xarray_kat.third_party.vendored.katdal.visdatav4_minimal import VisibilityDataV4


class TelstateDataProducts:
  """ "A proxy over the Telstate Data Products encapsulated in a katdal Dataset"""

  def __init__(self, datasource: Multiton[TelstateDataSource], **kw):
    self._dataset = VisibilityDataV4(datasource.instance, **kw)

  @property
  def name(self):
    """Return the underlying dataset name"""
    return self._dataset.name

  @property
  def datasource(self) -> TelstateDataSource:
    """Return the Telstate DataSource"""
    return self._dataset.source

  @property
  def telstate(self) -> TelescopeState:
    """Return the TelescopeState"""
    return self.datasource.telstate

  @property
  def sensor_cache(self) -> SensorCache:
    """Return the SensorCache"""
    return self._dataset.sensor

  @property
  def calibration_params(self) -> CorrectionParams | None:
    """Return the Calibration Parameters"""
    return self._dataset.calibration_params
