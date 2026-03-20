"""I/O helpers and shared constants for the analysis pipeline."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "time_worker",
    "time_frontend",
    "requestID",
    "clusterName",
    "funcName",
    "podID",
    "userID",
    "totalCost_worker",
    "workerCost",
    "runtimeCost",
    "totalCost_frontend",
    "frontendCost",
    "busCost",
    "readBodyCost",
    "writeRspCost",
    "cpu_usage",
    "memory_usage",
    "requestBodySize",
)

EXPORT_COLUMNS = (
    "time_worker",
    "time_frontend",
    "requestID",
    "clusterName",
    "funcName",
    "podID",
    "userID",
    "totalCost_worker",
    "workerCost",
    "runtimeCost",
    "totalCost_frontend",
    "frontendCost",
    "busCost",
    "readBodyCost",
    "writeRspCost",
    "cpu_usage",
    "memory_usage",
    "requestBodySize",
)

READ_DTYPES = {
    "time_worker": "string",
    "time_frontend": "string",
    "requestID": "string",
    "clusterName": "string",
    "funcName": "string",
    "podID": "string",
    "userID": "string",
    "totalCost_worker": "float64",
    "workerCost": "float64",
    "runtimeCost": "float64",
    "totalCost_frontend": "float64",
    "frontendCost": "float64",
    "busCost": "float64",
    "readBodyCost": "float64",
    "writeRspCost": "float64",
    "cpu_usage": "float64",
    "memory_usage": "float64",
    "requestBodySize": "Int64",
}

NUMERIC_COLUMNS = (
    "totalCost_worker",
    "workerCost",
    "runtimeCost",
    "totalCost_frontend",
    "frontendCost",
    "busCost",
    "readBodyCost",
    "writeRspCost",
    "cpu_usage",
    "memory_usage",
    "requestBodySize",
)

SAFE_NAME_PATTERN = re.compile(r"[^0-9A-Za-z._-]+")


def list_csv_files(data_dir: Path) -> list[Path]:
    """Return all CSV files under a data directory."""
    return sorted(path for path in data_dir.rglob("*.csv") if path.is_file())


def coerce_timestamp_column(series: pd.Series) -> pd.Series:
    """Convert timestamps to floating-point seconds.

    Numeric timestamps are used directly. If the column is not mostly numeric,
    the function tries to parse it as datetimes and converts to Unix seconds.
    """

    non_null = series.notna().sum()
    if non_null == 0:
        return pd.Series(np.nan, index=series.index, dtype="float64")

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= max(1, int(non_null * 0.9)):
        return numeric.astype("float64")

    datetimes = pd.to_datetime(series, errors="coerce", utc=True)
    seconds = pd.Series(np.nan, index=series.index, dtype="float64")
    valid = datetimes.notna()
    if valid.any():
        seconds.loc[valid] = datetimes.loc[valid].astype("int64") / 1_000_000_000
    return seconds


def normalize_chunk(chunk: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """Normalize dtypes and drop rows that cannot be analyzed."""

    normalized = chunk.copy()
    normalized["time_worker"] = coerce_timestamp_column(normalized["time_worker"])
    normalized["time_frontend"] = coerce_timestamp_column(normalized["time_frontend"])

    for column in NUMERIC_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized["funcName"] = normalized["funcName"].astype("string").str.strip()
    normalized = normalized[normalized["funcName"].notna()]
    normalized = normalized[normalized["funcName"] != ""]
    normalized = normalized.dropna(subset=[timestamp_column])
    normalized["funcName"] = normalized["funcName"].astype(str)
    return normalized


def sanitize_function_name(func_name: str, assigned_names: dict[str, str]) -> str:
    """Create a filesystem-safe function directory name with collision handling."""

    candidate = SAFE_NAME_PATTERN.sub("_", func_name).strip("._")
    if not candidate:
        candidate = "function"
    candidate = candidate[:100]

    if candidate in assigned_names and assigned_names[candidate] != func_name:
        digest = hashlib.sha1(func_name.encode("utf-8")).hexdigest()[:8]
        candidate = f"{candidate}_{digest}"

    assigned_names[candidate] = func_name
    return candidate


def validate_columns(csv_path: Path) -> None:
    """Fail early if an input file does not expose the expected schema."""

    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {', '.join(missing)}"
        )

