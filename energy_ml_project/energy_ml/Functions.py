import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


class StateDetector:
    def __init__(self, df: pd.DataFrame, energy_col: str = "value", n_states: int = 5):
        """
        Initialize the StateDetector with input data.

        Parameters:
        - df: DataFrame containing time-series energy data.
        - energy_col: Name of the column with energy measurements.
        - n_states: Number of operational states (clusters) to identify.
        """
        self.df = df.copy()
        self.energy_col = energy_col
        self.n_states = n_states

    def _extract_features(self, window: int = 5) -> pd.DataFrame:
        """
        Generate rolling statistical features from the energy signal.

        Parameters:
        - window: Size of the rolling window for smoothing and variability.

        Returns:
        - A DataFrame with four features:
            1. value: Original energy measurement
            2. delta: Absolute change between consecutive values
            3. rolling_mean: Rolling average (captures general trend)
            4. rolling_std: Rolling standard deviation (captures variability)
        """
        df = self.df.copy()

        # Compute absolute difference between consecutive energy values
        df["delta"] = df[self.energy_col].diff().fillna(0).abs()

        # Rolling average (centered)
        df["rolling_mean"] = df[self.energy_col].rolling(window, center=True).mean()
        df["rolling_mean"] = df["rolling_mean"].fillna(method='bfill').fillna(method='ffill')

        # Rolling standard deviation
        df["rolling_std"] = df[self.energy_col].rolling(window, center=True).std().fillna(0)

        # Return only the relevant feature columns
        return df[["value", "delta", "rolling_mean", "rolling_std"]]

    def classify_states(self) -> pd.DataFrame:
        """
        Classify operational states using KMeans clustering.

        Returns:
        - A new DataFrame with an additional column:
          'state_cluster' indicating the assigned cluster for each row.
        """
        # Extract engineered features
        df_features = self._extract_features()

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=self.n_states, random_state=42)
        state_labels = kmeans.fit_predict(df_features)

        # Add cluster labels to the original DataFrame
        df_result = self.df.copy()
        df_result["state_cluster"] = state_labels

        return df_result
    
class RuleBasedStateDetector:
    def __init__(self, df: pd.DataFrame, energy_col: str = "value"):
        self.df = df.copy()
        self.energy_col = energy_col

    def classify_states(self) -> pd.DataFrame:
        df = self.df.copy()

        def classify(value):
            if value == 0:
                return "Off"
            elif 0 < value <= 20:
                return "On"
            elif 20 < value <= 30:
                return "Standby"
            elif 30 < value <= 150:
                return "Normal Production"
            else:
                return "Abnormal Production"

        df["state"] = df[self.energy_col].apply(classify)
        return df
    

def segment_product_cycles(df: pd.DataFrame, energy_col: str = "value", threshold: float = 10.0, min_duration: int = 5) -> pd.DataFrame:
    """
    Automatically segment the energy time series into product cycles.

    Parameters:
    - df: Input DataFrame with a time-indexed energy signal.
    - energy_col: Name of the energy signal column.
    - threshold: Minimum energy value to consider the machine 'active'.
    - min_duration: Minimum number of consecutive active points to count as a cycle.

    Returns:
    - A copy of the DataFrame with a new column 'cycle_id' identifying each product cycle.
    """
    df = df.copy()
    df["active"] = df[energy_col] > threshold

    cycle_id = 0
    cycle_ids = []
    in_cycle = False
    duration = 0

    for is_active in df["active"]:
        if is_active:
            if not in_cycle:
                in_cycle = True
                duration = 1
                cycle_id += 1
            else:
                duration += 1
            cycle_ids.append(cycle_id)
        else:
            if in_cycle:
                if duration < min_duration:
                    cycle_id -= 1
                in_cycle = False
            duration = 0
            cycle_ids.append(0)

    df["cycle_id"] = cycle_ids
    df.drop(columns="active", inplace=True)
    return df

def count_units_by_period(df: pd.DataFrame, time_col: str = "timestamp", freq: str = "1H") -> pd.DataFrame:
    """
    Count number of production cycles per time period.

    Parameters:
    - df: DataFrame with a 'cycle_id' column and timestamp.
    - time_col: Column with timestamps.
    - freq: Resampling frequency (e.g., '1H' for hourly, '1D' for daily).

    Returns:
    - DataFrame with counts of cycles per period.
    """
    df_cycles = df[df["cycle_id"] > 0].drop_duplicates("cycle_id")
    df_cycles[time_col] = pd.to_datetime(df_cycles[time_col])
    df_cycles.set_index(time_col, inplace=True)

    count_by_period = df_cycles.resample(freq)["cycle_id"].count().rename("units_produced").reset_index()
    return count_by_period

def assess_cycle_quality(df: pd.DataFrame, time_col: str = "timestamp", expected_duration: float = None) -> pd.DataFrame:
    """
    Compare cycle durations against a standard and flag anomalies.

    Parameters:
    - df: DataFrame with 'cycle_id' and 'timestamp'.
    - expected_duration: Optional fixed expected duration in seconds.

    Returns:
    - DataFrame with one row per cycle and a 'quality_flag' column.
    """
    df_cycles = df[df["cycle_id"] > 0].groupby("cycle_id").agg(
        start_time=(time_col, "first"),
        end_time=(time_col, "last")
    ).reset_index()

    df_cycles["duration_sec"] = (df_cycles["end_time"] - df_cycles["start_time"]).dt.total_seconds()

    if expected_duration is None:
        expected_duration = df_cycles["duration_sec"].median()

    # Flag if duration is significantly different from expected
    tolerance = 0.5  # 40% deviation is considered anomalous
    df_cycles["quality_flag"] = df_cycles["duration_sec"].apply(
        lambda d: "anomalous" if abs(d - expected_duration) > tolerance * expected_duration else "ok"
    )

    return df_cycles
