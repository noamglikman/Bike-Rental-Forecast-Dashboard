
import pandas as pd
import pytest

from data import load_hourly_data


def test_original_dataset_loads():
    df = load_hourly_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 17379
    assert "cnt" in df.columns
    assert "dteday" in df.columns
    assert "hr" in df.columns


def test_missing_file_raises_error(tmp_path):
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_hourly_data(missing_file)

def test_timestamp_creation():
    from data import add_timestamp

    df = pd.DataFrame({
        "dteday": ["2011-01-02", "2011-01-01"],
        "hr": [6, 23],
        "cnt": [2, 10],
    })

    result = add_timestamp(df)

    assert result["timestamp"].is_monotonic_increasing
    assert result.iloc[0]["timestamp"] == pd.Timestamp(
        "2011-01-01 23:00:00"
    )
    assert result.iloc[1]["timestamp"] == pd.Timestamp(
        "2011-01-02 06:00:00"
    )

def test_audit_detects_missing_and_duplicate_hours():
    from data import audit_timestamps

    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2011-01-01 00:00",
            "2011-01-01 02:00",
            "2011-01-01 02:00",
        ])
    })

    result = audit_timestamps(df)

    assert result["observed_rows"] == 3
    assert result["expected_hours"] == 3
    assert result["missing_hours"] == 1
    assert result["duplicate_timestamps"] == 1

def test_audit_targets():
    from data import audit_targets

    df = pd.DataFrame({
        "cnt": [0, 5, -2, None, 1.5]
    })

    result = audit_targets(df)

    assert result["invalid_targets"] == 3
    assert result["zero_rental_hours"] == 1

def test_period_audit_totals():
    from data import add_timestamp, audit_by_period, load_hourly_data

    df = add_timestamp(load_hourly_data())
    audit = audit_by_period(df)

    assert list(audit["period"]) == [
        "train", "validation", "test"
    ]
    assert audit["observed_rows"].sum() == 17379
    assert audit["missing_hours"].sum() == 165
    assert audit["duplicate_timestamps"].sum() == 0
    assert audit["invalid_targets"].sum() == 0

def test_chronological_split():
    from data import split_by_period

    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2011-01-01 00:00",
            "2012-04-30 23:00",
            "2012-05-01 00:00",
            "2012-07-31 23:00",
            "2012-08-01 00:00",
            "2012-12-31 23:00",
        ]),
        "cnt": [1, 2, 3, 4, 5, 6],
    })

    train, validation, test = split_by_period(df)

    assert len(train) == 2
    assert len(validation) == 2
    assert len(test) == 2

    assert train["timestamp"].max() < pd.Timestamp("2012-05-01")
    assert validation["timestamp"].min() >= pd.Timestamp("2012-05-01")
    assert validation["timestamp"].max() < pd.Timestamp("2012-08-01")
    assert test["timestamp"].min() >= pd.Timestamp("2012-08-01")


def test_real_dataset_split():
    from data import (
        load_hourly_data,
        add_timestamp,
        split_by_period,
    )

    df = add_timestamp(load_hourly_data())
    train, validation, test = split_by_period(df)

    assert len(train) + len(validation) + len(test) == len(df)

    train_times = set(train["timestamp"])
    validation_times = set(validation["timestamp"])
    test_times = set(test["timestamp"])

    assert train_times.isdisjoint(validation_times)
    assert train_times.isdisjoint(test_times)
    assert validation_times.isdisjoint(test_times)
