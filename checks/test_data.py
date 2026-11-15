
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
