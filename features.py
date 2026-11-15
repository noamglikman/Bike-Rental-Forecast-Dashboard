
import pandas as pd


def build_calendar_features(df):
    """Create calendar features for each observation."""

    required_columns = ["timestamp", "holiday", "workingday"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    result = pd.DataFrame(index=df.index)
    timestamps = pd.to_datetime(df["timestamp"])

    result["hour"] = timestamps.dt.hour

    # UCI dataset uses Sunday=0, Monday=1, ..., Saturday=6
    result["weekday"] = (timestamps.dt.weekday + 1) % 7

    result["month"] = timestamps.dt.month
    result["holiday"] = df["holiday"]
    result["workingday"] = df["workingday"]

    return result
