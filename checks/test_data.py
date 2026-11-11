
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
