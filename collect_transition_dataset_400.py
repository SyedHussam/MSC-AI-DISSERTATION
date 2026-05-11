import argparse
import importlib.util
import json
import os
import shutil
from datetime import datetime

DEFAULT_SPEEDS = [0.0, 20.0, 60.0]
DEFAULT_TRANSITIONS = [("Red", "Green"), ("Green", "Yellow"), ("Yellow", "Red")]
DEFAULT_WEATHER_MODES = ["day_rain", "harsh_rain", "harsh_fog", "sun_glare"]

DEFAULT_RAIN_BY_MODE = {
    "day_rain": 25.0,
    "harsh_rain": 100.0,
    "harsh_fog": 100.0,
    "sun_glare": 0.0,
    "night_glare": 20.0,
}


def load_generator_module(script_path: str):
    spec = importlib.util.spec_from_file_location("generator_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generator script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_rmtree(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def prune_case_folder(case_dir: str):
    keep = {"roi_rgb.png", "dvs_sequence.npy", "meta.json"}
    for name in os.listdir(case_dir):
        full = os.path.join(case_dir, name)
        if name not in keep:
            if os.path.isdir(full):
                shutil.rmtree(full, ignore_errors=True)
            else:
                try:
                    os.remove(full)
                except FileNotFoundError:
                    pass


def move_case_to_sample(case_dir: str, dest_root: str, cell_id: str, sample_idx: int):
    sample_dir = os.path.join(dest_root, cell_id, f"sample_{sample_idx:04d}")
    ensure_dir(sample_dir)
    prune_case_folder(case_dir)
    for name in os.listdir(case_dir):
        shutil.move(os.path.join(case_dir, name), os.path.join(sample_dir, name))
    safe_rmtree(case_dir)
    return sample_dir


def main():
    parser = argparse.ArgumentParser(description="Lean quota-based CARLA collector for 400 accepted samples.")
    parser.add_argument("--generator-script", type=str, default="./recommended_dvs_transition_roi_matrix_harsh_conditions.py")
    parser.add_argument("--output", type=str, default="./dataset_400")
    parser.add_argument("--map", type=str, default="Town01")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--per-cell", type=int, default=10, help="Accepted samples per condition cell.")
    parser.add_argument("--extra-baseline", type=int, default=40, help="Extra accepted baseline samples after the core cells.")
    parser.add_argument("--baseline-weather-mode", type=str, default="day_rain")
    parser.add_argument("--baseline-rain", type=float, default=25.0)
    parser.add_argument("--max-attempts-per-cell", type=int, default=80)
    parser.add_argument("--weather-modes", nargs="+", default=DEFAULT_WEATHER_MODES)
    parser.add_argument("--speeds", nargs="+", type=float, default=DEFAULT_SPEEDS)
    parser.add_argument("--single-bin-ms", type=float, default=None)
    args = parser.parse_args()

    ensure_dir(args.output)
    tmp_root = os.path.join(args.output, "_tmp")
    ensure_dir(tmp_root)

    gen = load_generator_module(args.generator_script)

    client = gen.connect_carla(args.host, args.port)
    world = client.load_world(args.map)
    blueprint_library = world.get_blueprint_library()

    accepted_manifest = []
    collection_stats = {}
    global_sample_idx = 0

    def run_one(speed_kmh, start_state, end_state, weather_mode, rain_value):
        nonlocal global_sample_idx
        bin_ms = args.single_bin_ms if args.single_bin_ms is not None else gen.default_bin_ms(speed_kmh)

        meta = gen.run_case(
            world=world,
            blueprint_library=blueprint_library,
            output_dir=tmp_root,
            speed_kmh=speed_kmh,
            start_state=start_state,
            end_state=end_state,
            bin_ms=bin_ms,
            rain=rain_value,
            weather_mode=weather_mode,
        )

        case_dir = os.path.join(tmp_root, meta["case_name"])
        keep = bool(meta.get("quality_ok", False))

        if keep:
            cell_id = f"{weather_mode}__{int(speed_kmh)}kmh__{start_state}_to_{end_state}"
            global_sample_idx += 1
            sample_dir = move_case_to_sample(case_dir, args.output, cell_id, global_sample_idx)

            meta["sample_path"] = os.path.relpath(sample_dir, args.output).replace("\\", "/")
            with open(os.path.join(sample_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            accepted_manifest.append(meta)
            return True, meta

        safe_rmtree(case_dir)
        return False, meta

    try:
        total_cells = len(args.weather_modes) * len(args.speeds) * len(DEFAULT_TRANSITIONS)
        cell_counter = 0

        for weather_mode in args.weather_modes:
            rain_value = DEFAULT_RAIN_BY_MODE.get(weather_mode, 25.0)

            for speed_kmh in args.speeds:
                for start_state, end_state in DEFAULT_TRANSITIONS:
                    cell_counter += 1
                    cell_key = f"{weather_mode}__{int(speed_kmh)}kmh__{start_state}_to_{end_state}"
                    collection_stats[cell_key] = {"accepted": 0, "attempts": 0, "failures": 0}

                    print(f"\n[CELL {cell_counter}/{total_cells}] {cell_key} target={args.per_cell}")

                    while collection_stats[cell_key]["accepted"] < args.per_cell:
                        if collection_stats[cell_key]["attempts"] >= args.max_attempts_per_cell:
                            print(f"[WARN] Max attempts reached for {cell_key}")
                            break

                        collection_stats[cell_key]["attempts"] += 1
                        ok, meta = run_one(speed_kmh, start_state, end_state, weather_mode, rain_value)

                        if ok:
                            collection_stats[cell_key]["accepted"] += 1
                            print(f"[KEEP] {cell_key} accepted={collection_stats[cell_key]['accepted']}/{args.per_cell}")
                        else:
                            collection_stats[cell_key]["failures"] += 1
                            print(f"[DROP] {cell_key} quality_ok={meta.get('quality_ok')} issues={meta.get('quality_issues', [])}")

        # Extra baseline samples to reach 400 total
        baseline_cells = [(s, st, en) for s in args.speeds for st, en in DEFAULT_TRANSITIONS]
        baseline_idx = 0
        extra_accepted = 0

        print(f"\n[BASELINE EXTRA] target={args.extra_baseline}")

        while extra_accepted < args.extra_baseline:
            speed_kmh, start_state, end_state = baseline_cells[baseline_idx % len(baseline_cells)]
            baseline_idx += 1

            ok, meta = run_one(
                speed_kmh,
                start_state,
                end_state,
                args.baseline_weather_mode,
                args.baseline_rain,
            )

            if ok:
                extra_accepted += 1
                print(f"[KEEP] baseline_extra accepted={extra_accepted}/{args.extra_baseline}")
            else:
                print(f"[DROP] baseline_extra quality_ok={meta.get('quality_ok')} issues={meta.get('quality_issues', [])}")

    finally:
        gen.reset_async_mode(world)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "generator_script": os.path.abspath(args.generator_script),
        "map": args.map,
        "per_cell": args.per_cell,
        "extra_baseline": args.extra_baseline,
        "weather_modes": args.weather_modes,
        "speeds": args.speeds,
        "transitions": [list(x) for x in DEFAULT_TRANSITIONS],
        "total_accepted": len(accepted_manifest),
        "accepted_samples": accepted_manifest,
        "collection_stats": collection_stats,
        "saved_files_per_sample": ["roi_rgb.png", "dvs_sequence.npy", "meta.json"],
    }

    with open(os.path.join(args.output, "dataset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[DONE] accepted={len(accepted_manifest)} output={args.output}")


if __name__ == "__main__":
    main()