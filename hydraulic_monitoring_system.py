import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("d:/aviationproject")
DATA_DIR = ROOT / "configuration"
MODEL_PATH = ROOT / "hydraulic_monitoring_model.joblib"
REPORT_PATH = ROOT / "hydraulic_monitoring_report.json"

SENSOR_FILES = [
    "PS1.txt",
    "PS2.txt",
    "PS3.txt",
    "PS4.txt",
    "PS5.txt",
    "PS6.txt",
    "EPS1.txt",
    "FS1.txt",
    "FS2.txt",
    "TS1.txt",
    "TS2.txt",
    "TS3.txt",
    "TS4.txt",
    "VS1.txt",
    "CE.txt",
    "CP.txt",
    "SE.txt",
]

TARGET_COLUMNS = {
    "cooler_condition": 0,
    "valve_condition": 1,
    "pump_leakage": 2,
    "accumulator_pressure": 3,
    "stable_flag": 4,
}


def load_time_series(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, engine="python")


def load_all_sensors(data_dir: Path, sensor_files: list[str] = SENSOR_FILES) -> dict[str, pd.DataFrame]:
    sensor_data: dict[str, pd.DataFrame] = {}
    for sensor_file in sensor_files:
        path = data_dir / sensor_file
        if not path.exists():
            raise FileNotFoundError(f"Missing sensor file: {path}")
        sensor_data[sensor_file] = load_time_series(path)
    return sensor_data


def load_targets(data_dir: Path, target_name: str) -> pd.Series:
    profile_path = data_dir / "profile.txt"
    if not profile_path.exists():
        raise FileNotFoundError(f"Missing profile file: {profile_path}")
    targets = pd.read_csv(profile_path, sep=r"\s+", header=None, engine="python")
    return targets.iloc[:, TARGET_COLUMNS[target_name]]


def build_feature_matrix(sensor_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    feature_frames: list[pd.DataFrame] = []
    for sensor_file, sensor_df in sensor_data.items():
        prefix = sensor_file.replace(".txt", "").lower()
        features = {
            f"{prefix}_mean": sensor_df.mean(axis=1),
            f"{prefix}_std": sensor_df.std(axis=1),
            f"{prefix}_min": sensor_df.min(axis=1),
            f"{prefix}_max": sensor_df.max(axis=1),
            f"{prefix}_median": sensor_df.median(axis=1),
            f"{prefix}_start_end_diff": sensor_df.iloc[:, -1] - sensor_df.iloc[:, 0],
            f"{prefix}_range": sensor_df.max(axis=1) - sensor_df.min(axis=1),
        }
        feature_frames.append(pd.DataFrame(features))
    return pd.concat(feature_frames, axis=1)


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    X_clean = X.copy()
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    X_clean = X_clean.dropna(axis=1, how="all")
    X_clean = X_clean.fillna(X_clean.mean())
    return X_clean


def normalize_features(X: pd.DataFrame) -> pd.DataFrame:
    return (X - X.mean()) / X.std(ddof=0).replace(0, 1)


def build_monitoring_report(X: pd.DataFrame, y: pd.Series, target_name: str) -> dict:
    report = {
        "target": target_name,
        "sample_count": int(len(X)),
        "feature_summary": {},
        "label_distribution": y.value_counts(dropna=False).to_dict(),
    }
    for column in X.columns:
        report["feature_summary"][column] = {
            "min": float(X[column].min()),
            "max": float(X[column].max()),
            "mean": float(X[column].mean()),
            "std": float(X[column].std()),
        }
    return report


def build_model(target_name: str):
    if target_name == "accumulator_pressure":
        estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def train_predictive_model(target_name: str, save_model: bool = True) -> dict:
    sensor_data = load_all_sensors(DATA_DIR)
    X = build_feature_matrix(sensor_data)
    X = clean_features(X)
    y = load_targets(DATA_DIR, target_name)
    report = build_monitoring_report(X, y, target_name)

    X_norm = normalize_features(X)
    model = build_model(target_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm,
        y,
        test_size=0.2,
        random_state=42,
        stratify=None if target_name == "accumulator_pressure" else y,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    if target_name == "accumulator_pressure":
        report["evaluation"] = {
            "mean_squared_error": float(mean_squared_error(y_test, predictions))
        }
    else:
        report["evaluation"] = {
            "classification_report": classification_report(y_test, predictions, digits=4, output_dict=True)
        }

    if save_model:
        joblib.dump(model, MODEL_PATH)
        report["model_path"] = str(MODEL_PATH)

    report["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return report


def evaluate_saved_model(target_name: str, use_full_data: bool = False) -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    sensor_data = load_all_sensors(DATA_DIR)
    X = build_feature_matrix(sensor_data)
    X = clean_features(X)
    y = load_targets(DATA_DIR, target_name)
    X_norm = normalize_features(X)

    if use_full_data:
        predictions = model.predict(X_norm)
        y_true = y
    else:
        _, X_test, _, y_test = train_test_split(
            X_norm,
            y,
            test_size=0.2,
            random_state=42,
            stratify=None if target_name == "accumulator_pressure" else y,
        )
        predictions = model.predict(X_test)
        y_true = y_test

    if target_name == "accumulator_pressure":
        evaluation = {"mean_squared_error": float(mean_squared_error(y_true, predictions))}
    else:
        evaluation = {"classification_report": classification_report(y_true, predictions, digits=4, output_dict=True)}

    return {
        "target": target_name,
        "evaluation": evaluation,
        "use_full_data": use_full_data,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def monitor_configuration(target_name: str, interval: float) -> None:
    watch_paths = [DATA_DIR / sensor_file for sensor_file in SENSOR_FILES] + [DATA_DIR / "profile.txt"]
    last_mtimes = {path: path.stat().st_mtime for path in watch_paths if path.exists()}
    print(f"Monitoring hydraulic configuration for target '{target_name}'...")
    print(f"Press Ctrl+C to stop. Poll interval: {interval} seconds")

    while True:
        time.sleep(interval)
        changed = False
        for path in watch_paths:
            if not path.exists():
                continue
            mtime = path.stat().st_mtime
            if last_mtimes.get(path) != mtime:
                print(f"Detected change in {path.name}")
                last_mtimes[path] = mtime
                changed = True
        if changed:
            report = evaluate_saved_model(target_name, use_full_data=True)
            print(json.dumps(report, indent=2))


def save_report(report: dict, output_path: Path = REPORT_PATH) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved monitoring report to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hydraulic system monitoring workflow for flight operations.")
    parser.add_argument("--action", choices=["train", "evaluate", "report", "monitor"], default="train")
    parser.add_argument("--target", choices=list(TARGET_COLUMNS.keys()), default="valve_condition")
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds for monitor mode.")
    parser.add_argument("--use-full-data", action="store_true", help="Evaluate on the full dataset instead of a held-out split.")
    parser.add_argument("--no-save-model", action="store_true", help="Train without saving the model file.")
    parser.add_argument("--output", type=Path, default=REPORT_PATH, help="Output path for the report JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "train":
        report = train_predictive_model(args.target, save_model=not args.no_save_model)
        save_report(report, args.output)
    elif args.action == "evaluate":
        report = evaluate_saved_model(args.target, use_full_data=args.use_full_data)
        save_report(report, args.output)
    elif args.action == "report":
        sensor_data = load_all_sensors(DATA_DIR)
        X = build_feature_matrix(sensor_data)
        X = clean_features(X)
        y = load_targets(DATA_DIR, args.target)
        report = build_monitoring_report(X, y, args.target)
        save_report(report, args.output)
    elif args.action == "monitor":
        monitor_configuration(args.target, args.interval)


if __name__ == "__main__":
    main()
