# EnergyML

EnergyML is a Python toolkit for analyzing energy time-series data from industrial machines.  
It provides tools for state classification, production cycle segmentation, anomaly detection, and time-based aggregation.

---

## 📦 Features

- **State Detection** using KMeans or rule-based thresholds  
- **Cycle Segmentation** based on energy thresholds and durations  
- **Cycle Quality Assessment** to flag anomalies  
- **Time Aggregation** of production units (hourly, daily, etc.)  
- Designed for easy extension and integration in production pipelines

---

## 🔧 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/energy_ml_project.git
cd energy_ml_project
pip install -r requirements.txt

🚀 Quickstart
python
Copiar
Editar
import pandas as pd
from energy_ml.Functions import StateDetector, segment_product_cycles

# Load your energy time series
df = pd.read_csv("your_energy_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Detect machine states
detector = StateDetector(df, energy_col="value")
df_states = detector.classify_states()

# Segment product cycles
df_segmented = segment_product_cycles(df_states)

📁 Project Structure

energy_ml_project/
├── energy_ml/
│   └── Functions.py         # Core logic (state detection, segmentation, etc.)
├── tests/                   # Unit and integration tests (pytest)
├── docs/                    # Sphinx documentation
└── examples/                # Example notebooks and scripts


✅  Testing
To run unit tests:

pytest tests/



