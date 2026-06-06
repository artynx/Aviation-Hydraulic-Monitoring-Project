import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


ROOT = Path("d:/aviationproject")
DATA_DIR = ROOT / "configuration"

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


def extract_features_for_sensor(sensor_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    features = {
        f"{prefix}_mean": sensor_df.mean(axis=1),
        f"{prefix}_std": sensor_df.std(axis=1),
        f"{prefix}_min": sensor_df.min(axis=1),
        f"{prefix}_max": sensor_df.max(axis=1),
        f"{prefix}_median": sensor_df.median(axis=1),
        f"{prefix}_start_end_diff": sensor_df.iloc[:, -1] - sensor_df.iloc[:, 0],
        f"{prefix}_range": sensor_df.max(axis=1) - sensor_df.min(axis=1),
    }
    return pd.DataFrame(features)


def build_feature_matrix(sensor_files: list[str], data_dir: Path) -> pd.DataFrame:
    feature_frames = []
    for sensor_file in sensor_files:
        path = data_dir / sensor_file
        print(f"Loading {sensor_file}...")
        df = load_time_series(path)
        prefix = sensor_file.replace(".txt", "").lower()
        feature_frames.append(extract_features_for_sensor(df, prefix))
    return pd.concat(feature_frames, axis=1)


def load_targets(data_dir: Path, target_name: str) -> pd.Series:
    profile_path = data_dir / "profile.txt"
    targets = pd.read_csv(profile_path, sep=r"\s+", header=None, engine="python")
    return targets.iloc[:, TARGET_COLUMNS[target_name]]


def build_model(target_name: str):
    if target_name == "accumulator_pressure":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)),
            ]
        )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)),
        ]
    )


def train_and_evaluate(target_name: str, test_size: float, random_seed: int, save_model: bool = True) -> None:
    X = build_feature_matrix(SENSOR_FILES, DATA_DIR)
    y = load_targets(DATA_DIR, target_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=None if target_name == "accumulator_pressure" else y
    )

    model = build_model(target_name)
    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    if target_name == "accumulator_pressure":
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean squared error: {mse:.4f}")
    else:
        print(classification_report(y_test, y_pred, digits=4))

    if save_model:
        output_path = ROOT / f"trained_{target_name}_model.joblib"
        joblib.dump(model, output_path)
        print(f"Saved trained model to {output_path}")


def evaluate_saved_model(
    target_name: str,
    model_path: Path,
    test_size: float,
    random_seed: int,
    use_full_data: bool = False,
) -> None:
    X = build_feature_matrix(SENSOR_FILES, DATA_DIR)
    y = load_targets(DATA_DIR, target_name)
    model = joblib.load(model_path)

    if use_full_data:
        print(f"Evaluating saved model on all available data ({model_path})...")
        y_pred = model.predict(X)
        y_true, y_eval = y, y_pred
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=None if target_name == "accumulator_pressure" else y
        )
        print(f"Evaluating saved model on held-out test data ({model_path})...")
        y_true, y_eval = y_test, model.predict(X_test)

    if target_name == "accumulator_pressure":
        mse = mean_squared_error(y_true, y_eval)
        print(f"Mean squared error: {mse:.4f}")
    else:
        print(classification_report(y_true, y_eval, digits=4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate a hydraulic system model from sensor data.")
    parser.add_argument(
        "--target",
        choices=list(TARGET_COLUMNS.keys()),
        default="valve_condition",
        help="Target label to train on or evaluate.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to reserve for testing.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for train/test splitting.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip training and evaluate an existing model instead.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a saved model file for evaluation.",
    )
    parser.add_argument(
        "--use-full-data",
        action="store_true",
        help="Evaluate the model on the full dataset instead of a test split.",
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Train the model but do not save the trained model file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Target: {args.target}")

    if args.evaluate_only:
        model_path = args.model_path or ROOT / f"trained_{args.target}_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Saved model not found: {model_path}")
        evaluate_saved_model(args.target, model_path, args.test_size, args.random_seed, args.use_full_data)
    else:
        train_and_evaluate(args.target, args.test_size, args.random_seed, save_model=not args.no_save_model)


if __name__ == "__main__":
    main()
