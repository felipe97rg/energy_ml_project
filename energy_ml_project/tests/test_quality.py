import pandas as pd
from energy_ml.Functions import assess_cycle_quality

def test_quality_flags():
    timestamps = pd.date_range("2024-01-01", periods=20, freq="min")
    df = pd.DataFrame({
        "cycle_id": [1]*10 + [2]*10,
        "timestamp": list(timestamps)
    })
    result = assess_cycle_quality(df)
    assert "quality_flag" in result.columns
    assert set(result["quality_flag"]).issubset({"ok", "anomalous"})
