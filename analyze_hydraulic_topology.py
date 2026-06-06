import json
from pathlib import Path

MODEL_PATH = Path("d:/aviationproject/model.json")


def load_model(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_margin(line: dict, junction: dict) -> float:
    line_pressure = line.get("max_pressure", 0)
    junction_pressure = junction.get("safetest", {}).get("max_pressure", 0)
    return min(line_pressure, junction_pressure)


def analyze_topology(model: dict) -> dict:
    topology = model.get("hydraulic_topology", {})
    lines = {line["id"]: line for line in topology.get("lines", [])}
    junctions = topology.get("junctions", [])
    sensor_mapping = topology.get("sensor_mapping", [])

    analysis = {
        "junctions": [],
        "recommended_configuration": {
            "pump_max_pressure": None,
            "junction_limits": {},
            "sensor_assignments": []
        }
    }

    if not junctions:
        return analysis

    analysis["recommended_configuration"]["pump_max_pressure"] = model.get("components", {}).get("pump", {}).get("max_pressure")

    for sensor in sensor_mapping:
        analysis["recommended_configuration"]["sensor_assignments"].append(
            {
                "sensor": sensor["sensor"],
                "line": sensor["line"],
                "location": sensor.get("location"),
                "purpose": sensor.get("purpose")
            }
        )

    for junction in junctions:
        connected_lines = [lines[line_id] for line_id in junction.get("connected_lines", []) if line_id in lines]
        line_ids = [line["id"] for line in connected_lines]
        margins = [safe_margin(line, junction) for line in connected_lines]
        min_margin = min(margins) if margins else 0

        analysis["junctions"].append(
            {
                "id": junction["id"],
                "type": junction.get("type"),
                "connected_lines": line_ids,
                "safetest_max_pressure": junction.get("safetest", {}).get("max_pressure"),
                "min_line_rating": min((line.get("max_pressure", 0) for line in connected_lines), default=0),
                "safety_margin": min_margin,
                "recommended_pressure_limit": min(
                    junction.get("safetest", {}).get("max_pressure", 0),
                    min((line.get("max_pressure", 0) for line in connected_lines), default=0)
                )
            }
        )
        analysis["recommended_configuration"]["junction_limits"][junction["id"]] = analysis["junctions"][-1]["recommended_pressure_limit"]

    return analysis


def print_analysis(analysis: dict) -> None:
    print("Hydraulic Topology Safety Analysis")
    print("================================")
    print()
    print("Recommended pump maximum pressure:", analysis["recommended_configuration"]["pump_max_pressure"], "bar")
    print()
    print("Junction safety summary:")
    for junction in analysis["junctions"]:
        print(f"- {junction['id']} ({junction['type']}):")
        print(f"  connected lines: {', '.join(junction['connected_lines'])}")
        print(f"  junction pressure limit: {junction['safetest_max_pressure']} bar")
        print(f"  minimum connected line rating: {junction['min_line_rating']} bar")
        print(f"  effective safety margin: {junction['safety_margin']} bar")
        print(f"  recommended limit: {junction['recommended_pressure_limit']} bar")
        print()

    print("Sensor-to-line assignments:")
    for mapping in analysis["recommended_configuration"]["sensor_assignments"]:
        print(f"- {mapping['sensor']} -> {mapping['line']} ({mapping['purpose']})")
    print()
    print("Safest configuration guidance:")
    print("- Keep pump pressure at or below the lowest recommended junction limit.")
    for junction_id, limit in analysis["recommended_configuration"]["junction_limits"].items():
        print(f"  * {junction_id}: {limit} bar")
    print("- Use PS1 on the pump line to verify supply pressure before the tee junction.")
    print("- Use PS2 and PS3 to verify branch pressures in actuator lines.")


def main() -> None:
    model = load_model(MODEL_PATH)
    analysis = analyze_topology(model)
    print_analysis(analysis)


if __name__ == "__main__":
    main()
