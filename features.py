
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


def build_count_lags(df):
    """Look up past rental counts by exact timestamps."""
    required = {"timestamp", "cnt"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    timestamps = pd.to_datetime(df["timestamp"])

    if timestamps.duplicated().any():
        raise ValueError("Duplicate timestamps are not allowed")

    history = pd.Series(
        df["cnt"].to_numpy(),
        index=pd.DatetimeIndex(timestamps),
    )

    result = pd.DataFrame(index=df.index)

    for hours in [1, 24, 168]:
        source_times = timestamps - pd.Timedelta(hours=hours)

        values = history.reindex(
            pd.DatetimeIndex(source_times)
        )

        result[f"lag_{hours}"] = values.to_numpy()
        result[f"lag_{hours}_missing"] = values.isna().to_numpy()

        result[f"lag_{hours}_source"] = (
            source_times.to_numpy()
        )

    return result


def build_rolling_mean_24(df):
    """Calculate the mean of the previous 24 complete hours."""
    required = {"timestamp", "cnt"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    timestamps = pd.to_datetime(df["timestamp"])

    if timestamps.duplicated().any():
        raise ValueError("Duplicate timestamps are not allowed")

    history = pd.Series(
        df["cnt"].to_numpy(),
        index=pd.DatetimeIndex(timestamps),
    ).sort_index()

    # Include missing clock hours explicitly.
    hourly_index = pd.date_range(
        history.index.min(),
        history.index.max(),
        freq="h",
    )
    hourly_history = history.reindex(hourly_index)

    # Use only the 24 hours BEFORE the current hour.
    rolling = (
        hourly_history.shift(1)
        .rolling(window=24, min_periods=24)
        .mean()
    )

    values = rolling.reindex(
        pd.DatetimeIndex(timestamps)
    )

    result = pd.DataFrame(index=df.index)
    result["rolling_mean_24"] = values.to_numpy()
    result["rolling_mean_24_missing"] = (
        values.isna().to_numpy()
    )
    result["rolling_mean_24_source"] = (
        timestamps - pd.Timedelta(hours=1)
    ).to_numpy()

    return result


def build_weather_lags(df):
    """Look up weather values from the exact previous hour."""
    required = {"timestamp", "temp", "hum", "windspeed"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    timestamps = pd.to_datetime(df["timestamp"])

    if timestamps.duplicated().any():
        raise ValueError("Duplicate timestamps are not allowed")

    source_times = timestamps - pd.Timedelta(hours=1)
    result = pd.DataFrame(index=df.index)

    for column in ["temp", "hum", "windspeed"]:
        history = pd.Series(
            df[column].to_numpy(),
            index=pd.DatetimeIndex(timestamps),
        )

        values = history.reindex(
            pd.DatetimeIndex(source_times)
        )

        result[f"{column}_lag_1"] = values.to_numpy()
        result[f"{column}_lag_1_missing"] = (
            values.isna().to_numpy()
        )
        result[f"{column}_lag_1_source"] = (
            source_times.to_numpy()
        )

    return result


def build_historical_features(df):
    """Combine all historical predictors."""
    count_lags = build_count_lags(df)
    rolling_mean = build_rolling_mean_24(df)
    weather_lags = build_weather_lags(df)

    return pd.concat(
        [count_lags, rolling_mean, weather_lags],
        axis=1,
    )


