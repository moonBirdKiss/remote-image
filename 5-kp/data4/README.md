# Serverless Function Request Data Analysis

This project analyzes request logs from a serverless platform and produces:

1. Per-function request time-series exports in `output/single/<funcName>/`
2. Periodicity evidence for functions with strong recurring request patterns in `output/period/<funcName>/`

The code is designed for large CSV datasets and streams the input files in chunks instead of loading everything into memory at once.

## Project Files

- `analyze.py`: command-line entry point
- `serverless_analysis/io_utils.py`: schema, chunk normalization, and path helpers
- `serverless_analysis/pipeline.py`: streaming pipeline and output generation
- `serverless_analysis/periodicity.py`: periodogram/autocorrelation detection and plotting
- `requirements.txt`: Python dependencies

## Input Data

Place one or more CSV files under `data/`. The script processes every `*.csv` file under that directory recursively.

The expected columns are:

- `time_worker`
- `time_frontend`
- `requestID`
- `clusterName`
- `funcName`
- `podID`
- `userID`
- `totalCost_worker`
- `workerCost`
- `runtimeCost`
- `totalCost_frontend`
- `frontendCost`
- `busCost`
- `readBodyCost`
- `writeRspCost`
- `cpu_usage`
- `memory_usage`
- `requestBodySize`

The sample dataset in this workspace uses numeric timestamps in seconds. The loader also tries to parse datetime strings if needed.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Docker

Build the image:

```bash
docker build -t serverless-request-analysis .
```

Run it with local `data/` and `output/` directories mounted into the container:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  serverless-request-analysis
```

You can override the default command if needed, for example:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  serverless-request-analysis \
  python analyze.py --data-dir /app/data --output-dir /app/output --max-files 3
```

## Usage

Run the full analysis with default settings:

```bash
python analyze.py
```

Useful options:

```bash
python analyze.py \
  --data-dir data \
  --output-dir output \
  --timestamp-column time_worker \
  --chunk-size 200000 \
  --bin-size-seconds 10 \
  --min-bins 60 \
  --min-requests 15
```

For a quick smoke test on only a few files:

```bash
python analyze.py --max-files 3
```

## Output Layout

### `output/single/`

For every distinct function:

- `output/single/<funcName>/requests.csv`
  - All requests for that function, sorted by the chosen timestamp column
  - Includes timestamps, request ID, pod/user identifiers, latency metrics, CPU, memory, and request body size
- `output/single/<funcName>/summary.txt`
  - Request count and observed time range for that function
- `output/single/function_summary.csv`
  - One-row summary for every function

### `output/period/`

Only functions that pass the periodicity checks get a folder here:

- `output/period/<funcName>/binned_counts.csv`
  - Request counts per time bin
- `output/period/<funcName>/acf.csv`
  - Autocorrelation values by lag
- `output/period/<funcName>/periodogram.csv`
  - Spectral power for candidate periods
- `output/period/<funcName>/periodicity_report.txt`
  - Detected period and the main decision statistics
- `output/period/<funcName>/periodicity.png`
  - Plot containing the binned series, autocorrelation, and periodogram
- `output/period/periodic_functions.csv`
  - Summary table of all periodic functions that were detected

### Global Index

- `output/function_index.csv`
  - Maps the original `funcName` to the generated folder name
  - This is useful if a function name contains characters that are not safe in file paths

## Detection Method

The periodicity detector works on the request count series for each function:

1. Bin requests by time, defaulting to 10-second bins
   This keeps minute-level recurring traffic visible in the count series instead of flattening it into a constant one-request-per-minute pattern.
2. Compute a periodogram on the mean-centered request counts
3. Find the dominant candidate period
4. Validate that period using autocorrelation near one and two multiples of the detected lag

This combination reduces false positives from random spikes or one-off bursts. Thresholds are configurable from the command line.

## Notes on Performance

- Input CSV files are streamed in pandas chunks
- Per-function request files are written incrementally during ingestion
- Periodicity uses aggregated per-bin counts rather than raw timestamps, which is much more memory-efficient

## Expected Runtime Behavior

On a large dataset, the first phase spends most of its time streaming CSV chunks and writing `output/single/`. The periodicity phase runs after ingestion completes and only reads the in-memory aggregated count dictionaries, so it is usually much faster than the first phase.
