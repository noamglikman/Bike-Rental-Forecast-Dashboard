
import pandas as pd

from features import build_count_lags


def test_exact_hour_lookup():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2011-01-02 04:00",
            "2011-01-02 06:00",
            "2011-01-02 07:00",
        ]),
        "cnt": [3, 10, 15],
    })

    result = build_count_lags(df)

    # 05:00 is absent, so lag_1 must be missing.
    assert pd.isna(result.loc[1, "lag_1"])
    assert result.loc[1, "lag_1_missing"]

    # 07:00 has an exact previous-hour observation.
    assert result.loc[2, "lag_1"] == 10
    assert not result.loc[2, "lag_1_missing"]


def test_rolling_mean_24():
    from features import build_rolling_mean_24

    df = pd.DataFrame({
        "timestamp": pd.date_range(
            "2011-01-01 00:00",
            periods=26,
            freq="h",
        ),
        "cnt": list(range(1, 27)),
    })

    result = build_rolling_mean_24(df)

    # Not enough history at the first hour.
    assert pd.isna(result.loc[0, "rolling_mean_24"])
    assert result.loc[0, "rolling_mean_24_missing"]

    # Hours 1 through 24 have counts 1 through 24.
    assert result.loc[24, "rolling_mean_24"] == 12.5
    assert not result.loc[24, "rolling_mean_24_missing"]


def test_rolling_mean_requires_all_hours():
    from features import build_rolling_mean_24

    timestamps = pd.date_range(
        "2011-01-01",
        periods=26,
        freq="h",
    )

    # Remove one hour from the history.
    df = pd.DataFrame({
        "timestamp": timestamps.delete(10),
        "cnt": 1,
    })

    result = build_rolling_mean_24(df)

    # The final 24-hour window contains a missing hour.
    assert pd.isna(
        result.iloc[-1]["rolling_mean_24"]
    )
    assert result.iloc[-1]["rolling_mean_24_missing"]


def test_weather_lags():
    from features import build_weather_lags

    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2011-01-01 08:00",
            "2011-01-01 09:00",
        ]),
        "temp": [0.2, 0.4],
        "hum": [0.7, 0.6],
        "windspeed": [0.1, 0.3],
    })

    result = build_weather_lags(df)

    assert result.loc[1, "temp_lag_1"] == 0.2
    assert result.loc[1, "hum_lag_1"] == 0.7
    assert result.loc[1, "windspeed_lag_1"] == 0.1

    assert not result.loc[1, "temp_lag_1_missing"]
    assert result.loc[0, "temp_lag_1_missing"]


def test_weather_missing_hour():
    from features import build_weather_lags

    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2011-01-01 08:00",
            "2011-01-01 10:00",
        ]),
        "temp": [0.2, 0.4],
        "hum": [0.7, 0.6],
        "windspeed": [0.1, 0.3],
    })

    result = build_weather_lags(df)

    assert pd.isna(result.loc[1, "temp_lag_1"])
    assert result.loc[1, "temp_lag_1_missing"]

    assert pd.isna(result.loc[1, "hum_lag_1"])
    assert pd.isna(result.loc[1, "windspeed_lag_1"])

    
def test_real_dataset_historical_features():
    from data import load_hourly_data, add_timestamp
    from features import build_historical_features

    df = add_timestamp(load_hourly_data())
    features = build_historical_features(df)

    # One feature row for every observation.
    assert len(features) == len(df)

    # Seven historical features.
    value_columns = [
        "lag_1", "lag_24", "lag_168",
        "rolling_mean_24",
        "temp_lag_1", "hum_lag_1",
        "windspeed_lag_1",
    ]

    for column in value_columns:
        assert column in features.columns
        assert f"{column}_missing" in features.columns
        assert f"{column}_source" in features.columns

    # Every source timestamp must precede its target.
    for column in value_columns:
        assert (
            features[f"{column}_source"].to_numpy()
            < df["timestamp"].to_numpy()
        ).all()

    # Missing values must have missing flags.
    for column in value_columns:
        assert (
            features[column].isna().to_numpy()
            == features[f"{column}_missing"].to_numpy()
        ).all()


