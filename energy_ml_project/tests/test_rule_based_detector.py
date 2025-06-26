import pandas as pd
from energy_ml.Functions import RuleBasedStateDetector

def test_rule_based_states():
    df = pd.DataFrame({"value": [0, 10, 25, 50, 200]})
    detector = RuleBasedStateDetector(df)
    result = detector.classify_states()
    assert result["state"].tolist() == [
        "Off", "On", "Standby", "Normal Production", "Abnormal Production"
    ]
