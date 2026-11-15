
from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "data" / "hour.csv"

REQUIRED_COLUMNS = {
    "instant",
    "dteday",
    "hr",
    "cnt",
    "temp",
    "hum",
    "windspeed",
    "workingday",
    "holiday",
    "weekday",
    "mnth",
}


def load_hourly_data(path=DATA_PATH):
    """Load the original UCI hourly dataset."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return df


def add_timestamp(df):
    """Create and sort hourly timestamps."""
    result = df.copy()

    dates = pd.to_datetime(result["dteday"])
    hours = pd.to_numeric(result["hr"], errors="raise")

    if hours.isna().any() or not hours.between(0, 23).all():
        raise ValueError("Invalid hour values")

    result["timestamp"] = dates + pd.to_timedelta(
        hours, unit="h"
    )

    return result.sort_values("timestamp").reset_index(drop=True)


def audit_timestamps(df):
    """Check duplicate and missing hourly timestamps."""
    timestamps = pd.DatetimeIndex(df["timestamp"])

    duplicate_count = int(timestamps.duplicated().sum())

    expected_hours = pd.date_range(
        start=timestamps.min(),
        end=timestamps.max(),
        freq="h",
    )

    missing_hours = expected_hours.difference(timestamps)

    return {
        "observed_rows": len(df),
        "expected_hours": len(expected_hours),
        "missing_hours": len(missing_hours),
        "duplicate_timestamps": duplicate_count,
    }

def audit_targets(df):
    """Check missing, negative, and non-integer counts."""
    counts = pd.to_numeric(df["cnt"], errors="coerce")

    invalid = (
        counts.isna()
        | (counts < 0)
        | (counts.notna() & (counts % 1 != 0))
    )

    return {
        "invalid_targets": int(invalid.sum()),
        "zero_rental_hours": int((counts == 0).sum()),
    }

def save_data_audit(df, output_path="outputs/data_audit.csv"):
    """Save an initial full-dataset audit."""
    timestamp_results = audit_timestamps(df)
    target_results = audit_targets(df)

    summary = {
        **timestamp_results,
        **target_results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([summary]).to_csv(output_path, index=False)

    return summary


if __name__ == "__main__":
    df = load_hourly_data()
    df = add_timestamp(df)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(df[["timestamp", "cnt"]].head())
