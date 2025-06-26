import pandas as pd
import numpy as np
from energy_ml.Functions import StateDetector

def test_state_detector_clustering():
    df = pd.DataFrame({"value": np.random.rand(100) * 100})
    detector = StateDetector(df, energy_col="value", n_states=3)
    result = detector.classify_states()
    assert "state_cluster" in result.columns
    assert len(result) == len(df)
