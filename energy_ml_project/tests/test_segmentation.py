import pandas as pd
from energy_ml.Functions import segment_product_cycles

def test_segment_cycles():
    df = pd.DataFrame({"value": [0, 5, 15, 15, 5, 0, 20, 25, 0]})
    df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="min")
    result = segment_product_cycles(df, energy_col="value", threshold=10, min_duration=2)
    assert "cycle_id" in result.columns
    assert result["cycle_id"].max() >= 1
