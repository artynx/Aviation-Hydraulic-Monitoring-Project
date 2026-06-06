import argparse
import time
from pathlib import Path

import analyze_hydraulic_topology

ROOT = Path("d:/aviationproject")
MODEL_PATH = ROOT / "model.json"
CONFIG_DIR = ROOT / "configuration"


def get_watch_paths() -> list[Path]:
    paths = [MODEL_PATH]
    if CONFIG_DIR.exists() and CONFIG_DIR.is_dir():
        paths.extend(sorted(CONFIG_DIR.glob("*.txt")))
    return paths


def snapshot_paths(paths: list[Path]) -> dict[Path, float]:
    mtimes = {}
    for path in paths:
        if path.exists():
            mtimes[path] = path.stat().st_mtime
    return mtimes


def run_analysis() -> None:
    model = analyze_hydraulic_topology.load_model(MODEL_PATH)
    analysis = analyze_hydraulic_topology.analyze_topology(model)
    analyze_hydraulic_topology.print_analysis(analysis)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously watch hydraulic configuration files and rerun topology safety analysis.")
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds for file changes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Live hydraulic configuration analysis watcher")
    print(f"Watching: {MODEL_PATH}")
    print(f"Watching directory: {CONFIG_DIR}")
    print(f"Polling interval: {args.interval} seconds")
    print()

    watch_paths = get_watch_paths()
    last_mtimes = snapshot_paths(watch_paths)

    print("Initial analysis:")
    run_analysis()
    print("Watching for changes...\n")

    while True:
        time.sleep(args.interval)
        watch_paths = get_watch_paths()
        current_mtimes = snapshot_paths(watch_paths)

        if current_mtimes != last_mtimes:
            print(f"Change detected at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            run_analysis()
            last_mtimes = current_mtimes


if __name__ == "__main__":
    main()
