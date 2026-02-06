# Test Fixtures: Synthetic Observation Generation

This directory contains utilities for generating synthetic MeerKAT observations for testing xarray-kat without requiring access to the actual MeerKAT archive.

## Overview

The test infrastructure creates:
- **RDB files** containing telstate metadata (without Redis)
- **Chunked .npy files** with synthetic visibility, flag, and weight data
- **Realistic observation structure** matching MeerKAT archive format

## Components

### `rdb_generator.py`

Low-level utilities for creating RDB files from Python dictionaries:

```python
from fixtures import dict_to_rdb, create_sensor_data

# Create an RDB file directly from a Python dict
telstate_dict = {
    'sync_time': 1234567890.0,
    'int_time': 8.0,
    'chunk_info': {...}
}
dict_to_rdb(telstate_dict, Path('test.rdb'))

# Generate sensor data for scans
sensor_data = create_sensor_data(
    ntime=20,
    scan_configs=[
        {'indices': range(0, 10), 'state': 'track', 'target_name': 'PKS1934'},
        {'indices': range(10, 20), 'state': 'scan', 'target_name': '3C286'}
    ]
)
```

### `synthetic_observation.py`

High-level class for generating complete observations:

```python
from fixtures import SyntheticObservation

# Create an observation
obs = SyntheticObservation(
    capture_block_id="1234567890",
    ntime=20,
    nfreq=32,
    nants=4,
    int_time=8.0
)

# Add scans
obs.add_scan(range(0, 10), "track", "PKS1934")
obs.add_scan(range(10, 20), "scan", "3C286")

# Save everything
stats = obs.save_to_directory(Path("/tmp/mock_archive"))
```

This generates:
```
/tmp/mock_archive/
├── 1234567890_sdp_l0.full.rdb
└── 1234567890-sdp-l0/
    ├── correlator_data/
    │   ├── 00000_00000_00000.npy
    │   ├── 00000_00008_00000.npy
    │   ├── 00002_00000_00000.npy
    │   └── ...
    ├── flags/
    │   └── ...
    ├── weights/
    │   └── ...
    └── weights_channel/
        └── ...
```

## Usage in Tests

### Unit Testing Components

```python
def test_my_feature(tmp_path):
    """Test using synthetic observation."""
    obs = SyntheticObservation("123", ntime=10, nfreq=16, nants=4)
    obs.save_to_directory(tmp_path)

    # Now use the data
    rdb_path = tmp_path / "123_sdp_l0.full.rdb"
    ts = TelescopeState()
    ts.load_from_file(rdb_path)

    # Verify
    assert ts.get('int_time') == 8.0
```

### Integration Testing with Mock HTTP Server

The generated files can be served via a mock HTTP server (Phase 2):

```python
@pytest.fixture
def mock_archive(tmp_path):
    """Create mock archive with HTTP server."""
    obs = SyntheticObservation("1234567890", ntime=20, nfreq=32, nants=4)
    obs.save_to_directory(tmp_path)

    # Start HTTP server (to be implemented in Phase 2)
    server = MockArchiveServer(tmp_path)
    server.start()

    yield server

    server.stop()

def test_xarray_kat_integration(mock_archive):
    """Test xarray-kat with mock data."""
    url = f"{mock_archive.base_url}/1234567890/1234567890_sdp_l0.full.rdb"
    dt = xarray.open_datatree(url, engine="xarray-kat")
    dt.load()
```

## Features

### Customizable Dimensions

```python
obs = SyntheticObservation(
    capture_block_id="123",
    ntime=50,      # 50 time samples
    nfreq=128,     # 128 frequency channels
    nants=7,       # 7 antennas (21 baselines)
    npol=4,        # 4 polarizations
    int_time=4.0,  # 4 second integration
)
```

### Multiple Scans

```python
obs.add_scan(range(0, 20), "track", "PKS1934-638")
obs.add_scan(range(20, 30), "slew", None)
obs.add_scan(range(30, 50), "scan", "3C286")
```

### Custom Chunking

```python
obs.time_chunk_size = 4   # 4 samples per time chunk
obs.freq_chunk_size = 16  # 16 channels per freq chunk
```

### Realistic Data

- **Visibilities**: Complex ramp with phase variation
- **Flags**: ~5% randomly flagged
- **Weights**: Uniform high weights (200/255)
- **Channel weights**: Slight variation around 1.0

## Technical Details

### RDB Format Without Redis

Uses `katsdptelstate.rdb_utility` to encode Python objects directly to Redis DUMP format:

```python
# Encode value with msgpack
encoded = encoding.encode_value(python_value)

# Wrap in Redis DUMP format
dumped = rdb_utility.dump_string(encoded)

# Write to RDB file
writer.write_key(key_bytes, dumped)
```

This eliminates the need for a Redis server during testing.

### Array Chunking

Chunks match MeerKAT archive structure:
- Time: typically 2-8 samples per chunk
- Frequency: typically 8-256 channels per chunk
- Corrprod: never chunked (always full axis)

Chunk filenames: `TTTTT_FFFFF_CCCCC.npy` where:
- `TTTTT`: starting time index
- `FFFFF`: starting frequency index
- `CCCCC`: starting corrprod index (always 00000)

For 2D arrays (weights_channel): `TTTTT_FFFFF.npy`

### Telstate Schema

Generated telstate includes all keys required by xarray-kat:
- Timing: `sync_time`, `first_timestamp`, `int_time`
- Frequency: `center_freq`, `bandwidth`, `n_chans`, `sub_band`
- Antennas: `sub_pool_resources`, `bls_ordering`, `{ant}_observer`
- Arrays: `chunk_info` with metadata for each array
- Scans: `Observation/target`, `Observation/scan_index`, `Observation/scan_state`

## Testing

Run the test suite:

```bash
uv run pytest tests/test_synthetic_observation.py -v
```

Test coverage includes:
- RDB generation and serialization
- Sensor data generation
- Observation initialization
- Array data generation
- File I/O and chunking
- End-to-end integration

All tests pass without requiring external services or large data files.
