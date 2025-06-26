import pandas as pd
from energy_ml.Functions import count_units_by_period

def test_count_units():
    df = pd.DataFrame({
        "cycle_id": [1, 1, 2, 2, 3, 3],
        "timestamp": pd.date_range("2024-01-01", periods=6, freq="H")
    })
    result = count_units_by_period(df, time_col="timestamp", freq="2H")
    assert "units_produced" in result.columns
    assert result["units_produced"].sum() == 3

