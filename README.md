# Aviation Hydraulic Monitoring Project

This workspace contains a small hydraulic monitoring and flight simulation toolkit for aviation systems.

## Files and Purpose

- `train_hydraulic_model.py`
  - Builds a sensor-feature dataset from `configuration/*.txt`
  - Trains a machine learning model for hydraulic target labels
  - Saves the trained model

- `evaluate_hydraulic_model.py`
  - Evaluates a saved model on the current dataset
  - Supports full dataset or held-out test split evaluation

- `hydraulic_monitoring_system.py`
  - Implements a monitoring workflow
  - Performs data cleansing, model training, evaluation, and reporting
  - Writes a JSON monitoring report

- `analyze_hydraulic_topology.py`
  - Analyzes hydraulic line and junction safety limits from `model.json`
  - Prints recommended pressure configuration guidance

- `live_analyze_hydraulic_configuration.py`
  - Watches `model.json` and `configuration/*.txt`
  - Reruns topology safety analysis automatically when files change

- `realtime_hydraulic_alert.py`
  - Monitors hydraulic sensor files in real time
  - Raises alerts when pressure values exceed configured line thresholds

- `model.json`
  - Contains hydraulic topology, line limits, sensor mapping, and safety thresholds

- `configuration/`
  - Contains the sensor log files and `profile.txt` labels used for training and evaluation

## Setup

Use the local Python environment in `d:\aviationproject`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

### Train a model

```powershell
python .\train_hydraulic_model.py
```

### Evaluate a saved model

```powershell
python .\evaluate_hydraulic_model.py
```

### Run hydraulic topology safety analysis

```powershell
python .\analyze_hydraulic_topology.py
```

### Run live topology configuration analysis

```powershell
python .\live_analyze_hydraulic_configuration.py
```

### Run realtime hydraulic anomaly alerts

```powershell
python .\realtime_hydraulic_alert.py --watch
```

Monitor a specific line:

```powershell
python .\realtime_hydraulic_alert.py --line pump_line --watch
```

## Notes

- The alert system is currently file-based and uses values from the `configuration` directory.
- For real flight integration, adapt the data source to your live telemetry feed.
- The project is intended as a prototype for hydraulic monitoring, model training, and configuration safety analysis.
