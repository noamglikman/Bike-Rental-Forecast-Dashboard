
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


if __name__ == "__main__":
    df = load_hourly_data()
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(df.head())
