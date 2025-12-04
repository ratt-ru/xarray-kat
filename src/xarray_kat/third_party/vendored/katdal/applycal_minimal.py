import logging
import warnings

import numpy as np

from xarray_kat.third_party.vendored.katdal.categorical import (
  CategoricalData,
  ComparableArrayWrapper,
)
from xarray_kat.third_party.vendored.katdal.sensordata import (
  SensorGetter,
  SimpleSensorGetter,
)
from xarray_kat.third_party.vendored.katdal.spectral_window import SpectralWindow

# A constant indicating invalid / absent gain (typically due to flagged data)
INVALID_GAIN = np.complex64(complex(np.nan, np.nan))
# All the calibration product types katdal knows about
CAL_PRODUCT_TYPES = ("K", "B", "G", "GPHASE", "GAMP_PHASE")

logger = logging.getLogger(__name__)


def complex_interp(x, xi, yi, left=None, right=None):
  """Piecewise linear interpolation of magnitude and phase of complex values.

  Given discrete data points (`xi`, `yi`), this returns a 1-D piecewise
  linear interpolation `y` evaluated at the `x` coordinates, similar to
  `numpy.interp(x, xi, yi)`. While :func:`numpy.interp` interpolates the real
  and imaginary parts of `yi` separately, this function interpolates
  magnitude and (unwrapped) phase separately instead. This is useful when the
  phase of `yi` changes more rapidly than its magnitude, as in electronic
  gains.

  Parameters
  ----------
  x : 1-D sequence of float, length *M*
      The x-coordinates at which to evaluate the interpolated values
  xi : 1-D sequence of float, length *N*
      The x-coordinates of the data points, must be sorted in ascending order
  yi : 1-D sequence of complex, length *N*
      The y-coordinates of the data points, same length as `xi`
  left : complex, optional
      Value to return for `x < xi[0]`, default is `yi[0]`
  right : complex, optional
      Value to return for `x > xi[-1]`, default is `yi[-1]`

  Returns
  -------
  y : array of complex, length *M*
      The evaluated y-coordinates, same length as `x` and same dtype as `yi`
  """
  # Extract magnitude and unwrapped phase
  mag_i = np.abs(yi)
  phase_i = np.unwrap(np.angle(yi))
  # Prepare left and right interpolation extensions
  mag_left = phase_left = mag_right = phase_right = None
  if left is not None:
    mag_left = np.abs(left)
    with np.errstate(invalid="ignore"):
      phase_left = np.unwrap([phase_i[0], np.angle(left)])[1]
  if right is not None:
    mag_right = np.abs(right)
    with np.errstate(invalid="ignore"):
      phase_right = np.unwrap([phase_i[-1], np.angle(right)])[1]
  # Interpolate magnitude and phase separately, and reassemble
  mag = np.interp(x, xi, mag_i, left=mag_left, right=mag_right)
  phase = np.interp(x, xi, phase_i, left=phase_left, right=phase_right)
  y = np.empty_like(phase, dtype=np.complex128)
  np.cos(phase, out=y.real)
  np.sin(phase, out=y.imag)
  y *= mag
  return y.astype(yi.dtype)


def quiet_reciprocal(x):
  """Invert `x` but don't complain about invalid values."""
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", "invalid value", RuntimeWarning)
    return np.reciprocal(x)


def _parse_cal_product(cal_product):
  """Split `cal_product` into `cal_stream` and `product_type` parts."""
  fields = cal_product.rsplit(".", 1)
  if len(fields) != 2:
    raise ValueError(
      f"Calibration product {cal_product} is not in the format "
      "<cal_stream>.<product_type>"
    )
  return fields[0], fields[1]


def get_cal_product(cache, cal_stream, product_type):
  """Extract calibration solution from cache as a sensor.

  Parameters
  ----------
  cache : :class:`~katdal.sensordata.SensorCache` object
      Sensor cache serving cal product sensors
  cal_stream : string
      Name of calibration stream (e.g. "l1")
  product_type : string
      Calibration product type (e.g. "G")
  """
  sensor_name = f"Calibration/Products/{cal_stream}/{product_type}"
  return cache.get(sensor_name)


def calc_delay_correction(sensor, index, data_freqs):
  """Calculate correction sensor from delay calibration solution sensor.

  Given the delay calibration solution `sensor`, this extracts the delay time
  series of the input specified by `index` (in the form (pol, ant)) and
  builds a categorical sensor for the corresponding complex correction terms
  (channelised by `data_freqs`).

  Invalid delays (NaNs) are replaced by zeros, since bandpass calibration
  still has a shot at fixing any residual delay.
  """
  delays = [np.nan_to_num(value[index]) for segm, value in sensor.segments()]
  # Delays produced by cal pipeline are raw phase slopes, i.e. exp(2 pi j d f)
  corrections = [
    np.exp(-2j * np.pi * d * data_freqs).astype("complex64") for d in delays
  ]
  corrections = [ComparableArrayWrapper(c) for c in corrections]
  return CategoricalData(corrections, sensor.events)


def calc_bandpass_correction(sensor, index, data_freqs, cal_freqs):
  """Calculate correction sensor from bandpass calibration solution sensor.

  Given the bandpass calibration solution `sensor`, this extracts the time
  series of bandpasses (channelised by `cal_freqs`) for the input specified
  by `index` (in the form (pol, ant)) and builds a categorical sensor for
  the corresponding complex correction terms (channelised by `data_freqs`).

  Invalid solutions (NaNs) are replaced by linear interpolations over
  frequency (separately for magnitude and phase), as long as some channels
  have valid solutions.
  """
  corrections = []
  for segment, value in sensor.segments():
    bp = value[(slice(None),) + index]
    valid = np.isfinite(bp)
    if valid.any():
      # Don't extrapolate to edges of band where gain typically drops off
      bp = complex_interp(
        data_freqs, cal_freqs[valid], bp[valid], left=INVALID_GAIN, right=INVALID_GAIN
      )
    else:
      bp = np.full(len(data_freqs), INVALID_GAIN)
    corrections.append(ComparableArrayWrapper(quiet_reciprocal(bp)))
  return CategoricalData(corrections, sensor.events)


def calc_gain_correction(sensor, index, targets=None):
  """Calculate correction sensor from gain calibration solution sensor.

  Given the gain calibration solution `sensor`, this extracts the time
  series of gains for the input specified by `index` (in the form (pol, ant))
  and interpolates them over time to get the corresponding complex correction
  terms. The optional `targets` parameter is a :class:`CategoricalData` i.e.
  a sensor indicating the target associated with each dump. The targets can
  be actual :class:`katpoint.Target` objects or indices, as long as they
  uniquely identify the target. If provided, interpolate solutions derived
  from one target only at dumps associated with that target, which is what
  you want for self-calibration solutions (but not for standard calibration
  based on gain calibrator sources).

  Invalid solutions (NaNs) are replaced by linear interpolations over time
  (separately for magnitude and phase), as long as some dumps have valid
  solutions on the appropriate target.
  """
  dumps = np.arange(sensor.events[-1])
  events = []
  gains = []
  for segment, value in sensor.segments():
    # Discard "invalid gain" placeholder (typically the initial value)
    if value is INVALID_GAIN:
      continue
    events.append(segment.start)
    gains.append(value[(Ellipsis,) + index])
  if not events:
    return np.full((len(dumps), 1), INVALID_GAIN)
  events = np.array(events)
  # Let the gains be shaped either (cal_n_chans, n_events) or (1, n_events)
  gains = np.atleast_2d(np.array(gains).T)
  # Assume all dumps have the same target by default, i.e. interpolate freely
  if targets is None:
    targets = CategoricalData([0], [0, len(dumps)])
  smooth_gains = np.full((len(dumps), gains.shape[0]), INVALID_GAIN)
  # We either have a single dummy target (L1) or iterate over actual targets (L2)
  for target in targets.unique_values:
    on_target = targets == target
    # Iterate over number of channels / "IFs" / subbands in gain product
    for chan, gains_per_chan in enumerate(gains):
      valid = np.isfinite(gains_per_chan) & on_target[events]
      if valid.any():
        # The current target has at least one valid gain solution in the channel
        smooth_gains[on_target, chan] = complex_interp(
          dumps[on_target], events[valid], gains_per_chan[valid]
        )
      elif not on_target[events].any():
        # We are on a target without any (L2) gain solutions, i.e. a calibrator.
        # Rather preserve the L1 gains by setting L2 gains to 1.0 in this case.
        smooth_gains[on_target, chan] = np.complex64(1.0)
  return quiet_reciprocal(smooth_gains)


def calibrate_flux(sensor, targets, gaincal_flux):
  """Apply flux scale to calibrator gains (aka flux calibration).

  Given the gain calibration solution `sensor`, this identifies the target
  associated with each set of solutions by looking up the gain events in the
  `targets` sensor, and then scales the gains by the inverse square root of
  the relevant flux if a valid match is found in the `gaincal_flux` dict. This
  is equivalent to the final step of the AIPS GETJY and CASA fluxscale tasks.
  """
  # If no calibration info is available, do nothing
  if not gaincal_flux:
    return sensor
  calibrated_gains = []
  for segment, gains in sensor.segments():
    # Ignore "invalid gain" placeholder (typically the initial value)
    if gains is INVALID_GAIN:
      calibrated_gains.append(ComparableArrayWrapper(gains))
      continue
    # Find the target at the time of the gain solution (i.e. gain calibrator)
    target = targets[segment.start]
    for name in [target.name] + target.aliases:
      flux = gaincal_flux.get(name, np.nan)
      # Scale the gains if a valid flux density was found for this target
      if flux > 0.0:
        calibrated_gains.append(ComparableArrayWrapper(gains / np.sqrt(flux)))
        break
    else:
      calibrated_gains.append(ComparableArrayWrapper(gains))
  return CategoricalData(calibrated_gains, sensor.events)


def add_applycal_sensors(
  cache, attrs, data_freqs, cal_stream, cal_substreams=None, gaincal_flux={}
):
  """Register virtual sensors for one calibration stream.

  This operates on a single calibration stream called `cal_stream` (possibly
  an alias), which derives from one or more underlying cal streams listed in
  `cal_substreams` and has stream attributes in `attrs`.

  The first set of virtual sensors maps all cal products into a unified
  namespace (template 'Calibration/Products/`cal_stream`/{product_type}').
  Map receptor inputs to the relevant indices in each calibration product
  based on the ants and pols found in `attrs`. Then register a virtual sensor
  per product type and per input in the SensorCache `cache`, with template
  'Calibration/Corrections/`cal_stream`/{product_type}/{inp}'. The virtual
  sensor function picks the appropriate correction calculator based on the
  cal product type, which also uses auxiliary info like the channel
  frequencies, `data_freqs`.

  Parameters
  ----------
  cache : :class:`~katdal.sensordata.SensorCache` object
      Sensor cache serving cal product sensors and receiving correction sensors
  attrs : dict-like
      Calibration stream attributes (e.g. a "cal" telstate view)
  data_freqs : array of float, shape (*F*,)
      Centre frequency of each frequency channel of visibilities, in Hz
  cal_stream : string
      Name of (possibly virtual) calibration stream (e.g. "l1")
  cal_substreams : sequence of string, optional
      Names of actual underlying calibration streams (e.g. ["cal"]),
      defaults to [`cal_stream`] itself
  gaincal_flux : dict mapping string to float, optional
      Flux density (in Jy) per gaincal target name, used to flux calibrate
      the "G" product, overriding the measured flux stored in `attrs`
      (if available). A value of None disables flux calibration.

  Returns
  -------
  cal_freqs : 1D array of float, or None
      Centre frequency of each frequency channel of calibration stream, in Hz
      (or None if no sensors were registered)
  """
  if cal_substreams is None:
    cal_substreams = [cal_stream]
  cal_ants = attrs.get("antlist", [])
  cal_pols = attrs.get("pol_ordering", [])
  cal_input_map = {
    ant + pol: (pol_idx, ant_idx)
    for (pol_idx, pol) in enumerate(cal_pols)
    for (ant_idx, ant) in enumerate(cal_ants)
  }
  if not cal_input_map:
    return
  try:
    cal_spw = SpectralWindow(
      attrs["center_freq"],
      None,
      attrs["n_chans"],
      sideband=1,
      bandwidth=attrs["bandwidth"],
    )
    cal_freqs = cal_spw.channel_freqs
  except KeyError:
    logger.warning(
      "Disabling cal stream '%s' due to missing spectral attributes", cal_stream
    )
    return
  targets = cache.get("Observation/target")
  # Override pipeline fluxes (or disable flux calibration)
  if gaincal_flux is None:
    gaincal_flux = {}
  else:
    measured_flux = attrs.get("measured_flux", {}).copy()
    measured_flux.update(gaincal_flux)
    gaincal_flux = measured_flux

  def indirect_cal_product_name(name, product_type):
    # XXX The first underscore below is actually a telstate separator...
    return name.split("/")[-2] + "_product_" + product_type

  def indirect_cal_product_raw(cache, name, product_type):
    # XXX The first underscore below is actually a telstate separator...
    product_str = "_product_" + product_type
    raw_products = []
    for stream in cal_substreams:
      sensor_name = stream + product_str
      raw_product = cache.get(sensor_name, extract=False)
      assert isinstance(raw_product, SensorGetter), (
        sensor_name + " is already extracted"
      )
      raw_products.append(raw_product)
    if len(raw_products) == 1:
      return raw_products[0]
    else:
      raw_products = [raw.get() for raw in raw_products]
      timestamps = np.concatenate(
        [raw_product.timestamp for raw_product in raw_products]
      )
      values = np.concatenate([raw_product.value for raw_product in raw_products])
      ordered = timestamps.argsort()
      timestamps = timestamps[ordered]
      values = values[ordered]
      return SimpleSensorGetter(
        indirect_cal_product_name(name, product_type), timestamps, values
      )

  def indirect_cal_product(cache, name, product_type):
    try:
      n_parts = int(attrs[f"product_{product_type}_parts"])
    except KeyError:
      return indirect_cal_product_raw(cache, name, product_type)
    # Handle multi-part cal product (as produced by "split cal")
    # First collect all the parts as sensors (and mark missing ones as None)
    parts = []
    for n in range(n_parts):
      try:
        part = indirect_cal_product_raw(cache, name + str(n), product_type + str(n))
      except KeyError:
        part = SimpleSensorGetter(name + str(n), np.array([]), np.array([]))
      parts.append(part)

    # Stitch together values with the same timestamp
    parts = [part.get() for part in parts]
    timestamps = []
    values = []
    part_indices = [0] * n_parts
    part_timestamps = [
      part.timestamp[0] if len(part.timestamp) else np.inf for part in parts
    ]
    while True:
      next_timestamp = min(part_timestamps)
      if next_timestamp == np.inf:
        break
      pieces = []
      for ts, ind, part in zip(part_timestamps, part_indices, parts):
        if ts == next_timestamp:
          piece = ComparableArrayWrapper.unwrap(part.value[ind])
          pieces.append(piece)
        else:
          pieces.append(None)
      if any(piece is None for piece in pieces):
        invalid = np.full_like(piece, INVALID_GAIN)
        pieces = [piece if piece is not None else invalid for piece in pieces]
      timestamps.append(next_timestamp)
      value = np.concatenate(pieces, axis=0)
      values.append(ComparableArrayWrapper(value))
      for i, part in enumerate(parts):
        if part_timestamps[i] == next_timestamp:
          ts = part.timestamp
          part_indices[i] += 1
          part_timestamps[i] = (
            ts[part_indices[i]] if part_indices[i] < len(ts) else np.inf
          )
    if not timestamps:
      raise KeyError(f"No cal product '{name}' parts found (expected {n_parts})")
    return SimpleSensorGetter(
      indirect_cal_product_name(name, product_type),
      np.array(timestamps),
      np.array(values),
    )

  def calc_correction_per_input(cache, name, inp, product_type):
    """Calculate correction sensor for input `inp` from cal solutions."""
    product_sensor = get_cal_product(cache, cal_stream, product_type)
    try:
      index = cal_input_map[inp]
    except KeyError:
      raise KeyError(
        f"No calibration solutions available for input '{inp}' - "
        f"available ones are {sorted(cal_input_map.keys())}"
      )
    if product_type == "K":
      correction_sensor = calc_delay_correction(product_sensor, index, data_freqs)
    elif product_type == "B":
      correction_sensor = calc_bandpass_correction(
        product_sensor, index, data_freqs, cal_freqs
      )
    elif product_type == "G":
      product_sensor = calibrate_flux(product_sensor, targets, gaincal_flux)
      correction_sensor = calc_gain_correction(product_sensor, index)
    elif product_type in ("GPHASE", "GAMP_PHASE"):
      correction_sensor = calc_gain_correction(product_sensor, index, targets)
    else:
      raise KeyError(
        f"Unknown calibration product type '{product_type}' - "
        f"available ones are {CAL_PRODUCT_TYPES}"
      )
    cache[name] = correction_sensor
    return correction_sensor

  template = f"Calibration/Products/{cal_stream}/{{product_type}}"
  cache.virtual[template] = indirect_cal_product
  template = f"Calibration/Corrections/{cal_stream}/{{product_type}}/{{inp}}"
  cache.virtual[template] = calc_correction_per_input
  return cal_freqs
