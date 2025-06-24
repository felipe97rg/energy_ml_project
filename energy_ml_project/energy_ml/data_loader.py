import pandas as pd
from typing import Optional


class MachineDataLoader:
    def __init__(self, file_path: str, timestamp_col: str = "timestamp", energy_col: str = "value"):
        self.file_path = file_path
        self.timestamp_col = timestamp_col
        self.energy_col = energy_col
        self.df = None

    def load_data(self) -> pd.DataFrame:
        # Leer como JSON (lista de dicts)
        try:
            self.df = pd.read_json(self.file_path)
        except Exception as e:
            raise ValueError(f"Error reading JSON file: {e}")

        # Convertir timestamp
        self.df[self.timestamp_col] = pd.to_datetime(self.df[self.timestamp_col], errors='coerce')

        # Ordenar cronológicamente
        self.df.dropna(subset=[self.timestamp_col], inplace=True)
        self.df.sort_values(by=self.timestamp_col, inplace=True)

        return self.df

    def preprocess(self, method: str = "interpolate") -> pd.DataFrame:
        df = self.df.copy()

        # Eliminar valores negativos o atípicos si es necesario
        df = df[df[self.energy_col] > 0]

        # Rellenar valores faltantes
        if method == "interpolate":
            df[self.energy_col] = df[self.energy_col].interpolate(method="linear")
        elif method == "ffill":
            df[self.energy_col] = df[self.energy_col].fillna(method="ffill")

        return df
