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

---

## 📘 Demo Notebook

For a complete demonstration of how the project works — including data loading, state detection, cycle segmentation, anomaly detection, and conclusions — please refer to the [`demo_workflow.ipynb`](energy_ml_project/notebooks/demo_workflow.ipynb) notebook in the `notebooks/` folder.


## 🔧 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/felipe97rg/energy_ml_project.git
cd energy_ml_project
pip install -r requirements.txt

🚀 Quickstart

from energy_ml.Functions import StateDetector, segment_product_cycles



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



