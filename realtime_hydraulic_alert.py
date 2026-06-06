import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path("d:/aviationproject")
DATA_DIR = ROOT / "configuration"
MODEL_PATH = ROOT / "model.json"

SENSOR_FILE_SUFFIX = ".txt"
PRESSURE_TOKEN_RE = re.compile(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)")


def load_model(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_line_thresholds(model: dict) -> dict[str, float]:
    topology = model.get("hydraulic_topology", {})
    lines = {line["id"]: float(line.get("max_pressure", 0)) for line in topology.get("lines", [])}
    junctions = topology.get("junctions", [])
    for junction in junctions:
        safetest = junction.get("safetest", {})
        junction_limit = safetest.get("max_pressure")
        if junction_limit is None:
            continue
        for line_id in junction.get("connected_lines", []):
            if line_id in lines:
                lines[line_id] = min(lines[line_id], float(junction_limit))
    return lines


def build_sensor_mapping(model: dict) -> dict[str, str]:
    mapping = {}
    for sensor in model.get("hydraulic_topology", {}).get("sensor_mapping", []):
        sensor_name = sensor.get("sensor")
        line_id = sensor.get("line")
        if sensor_name and line_id:
            mapping[sensor_name] = line_id
    return mapping


def get_latest_sensor_value(sensor_name: str) -> float | None:
    path = DATA_DIR / f"{sensor_name}{SENSOR_FILE_SUFFIX}"
    if not path.exists():
        return None
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        chunk_size = min(8192, size)
        f.seek(max(0, size - chunk_size))
        raw = f.read().decode("utf-8", errors="ignore")

    tokens = PRESSURE_TOKEN_RE.findall(raw)
    if not tokens:
        return None
    try:
        return float(tokens[-1])
    except ValueError:
        return None


def check_line_alerts(
    line_filter: str | None = None,
    override_threshold: float | None = None,
) -> list[dict]:
    model = load_model(MODEL_PATH)
    line_thresholds = build_line_thresholds(model)
    sensor_mapping = build_sensor_mapping(model)

    alerts: list[dict] = []
    for sensor_name, line_id in sensor_mapping.items():
        if line_filter and line_id != line_filter:
            continue

        value = get_latest_sensor_value(sensor_name)
        if value is None:
            continue

        threshold = override_threshold if override_threshold is not None else line_thresholds.get(line_id)
        if threshold is None:
            continue

        alert = {
            "sensor": sensor_name,
            "line": line_id,
            "value": value,
            "threshold": threshold,
            "alert": value > threshold,
        }
        alerts.append(alert)
    return alerts


def format_alerts(alerts: list[dict]) -> str:
    lines: list[str] = []
    if not alerts:
        return "No monitored sensors found for the configured lines."

    for entry in alerts:
        status = "ALERT" if entry["alert"] else "OK"
        lines.append(
            f"{status}: sensor={entry['sensor']} line={entry['line']} value={entry['value']:.2f} "
            f"threshold={entry['threshold']:.2f}"
        )
    return "\n".join(lines)


def watch_alerts(
    line_filter: str | None = None,
    override_threshold: float | None = None,
    interval: float = 2.0,
) -> None:
    watch_paths = [MODEL_PATH]
    sensor_mapping = build_sensor_mapping(load_model(MODEL_PATH))
    if line_filter:
        sensors = [name for name, line in sensor_mapping.items() if line == line_filter]
    else:
        sensors = list(sensor_mapping.keys())
    watch_paths.extend(DATA_DIR / f"{sensor}{SENSOR_FILE_SUFFIX}" for sensor in sensors)

    last_mtimes = {path: path.stat().st_mtime for path in watch_paths if path.exists()}
    print("Realtime hydraulic line alert monitor started.")
    print(f"Monitoring sensors: {', '.join(sensors)}")
    print(f"Polling interval: {interval} seconds")
    print("Press Ctrl+C to stop.")

    while True:
        time.sleep(interval)
        current_mtimes = {}
        changed = False
        for path in watch_paths:
            if not path.exists():
                continue
            current_mtimes[path] = path.stat().st_mtime
            if path not in last_mtimes or current_mtimes[path] != last_mtimes[path]:
                changed = True
        if changed:
            last_mtimes = current_mtimes
            alerts = check_line_alerts(line_filter=line_filter, override_threshold=override_threshold)
            print(time.strftime("\n[%Y-%m-%d %H:%M:%S]"))
            print(format_alerts(alerts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime hydraulic line anomaly alert system.")
    parser.add_argument(
        "--line",
        type=str,
        default=None,
        help="Specific hydraulic line ID to monitor (e.g. pump_line).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override pressure threshold in bar.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continue monitoring sensor files for changes in real time.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds for watch mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alerts = check_line_alerts(line_filter=args.line, override_threshold=args.threshold)
    print(format_alerts(alerts))

    if args.watch:
        watch_alerts(line_filter=args.line, override_threshold=args.threshold, interval=args.interval)


if __name__ == "__main__":
    main()
