import argparse
import importlib.util
import json
import os
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# DEFAULT DATASET DESIGN
# ============================================================

DEFAULT_TRANSITIONS = [
    ("Red", "Yellow", "Red_to_Yellow"),
    ("Yellow", "Green", "Yellow_to_Green"),
    ("Green", "Yellow", "Green_to_Yellow"),
    ("Yellow", "Red", "Yellow_to_Red"),
]

DEFAULT_WEATHER_MODES = [
    "day_rain",
    "sun_glare",
    "harsh_rain",
    "harsh_fog",
]

DEFAULT_SPEEDS = [
    0.0,
    30.0,
    60.0,
    80.0,
]

RAIN_BY_MODE = {
    "clear": 0.0,
    "day_rain": 25.0,
    "sun_glare": 0.0,
    "harsh_rain": 100.0,
    "harsh_fog": 100.0,
    "night_glare": 20.0,
}


# ============================================================
# BASIC HELPERS
# ============================================================

def load_generator_module(script_path):
    spec = importlib.util.spec_from_file_location("generator_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generator script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def crop_with_pad(img, bbox, pad=8):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)

    if x2 <= x1 or y2 <= y1:
        return None, [x1, y1, x2, y2]

    return img[y1:y2, x1:x2].copy(), [x1, y1, x2, y2]


def resize_roi(roi, out_w, out_h):
    return cv2.resize(roi, (out_w, out_h), interpolation=cv2.INTER_AREA)


def state_to_carla(gen, state_name):
    table = {
        "Red": gen.carla.TrafficLightState.Red,
        "Yellow": gen.carla.TrafficLightState.Yellow,
        "Green": gen.carla.TrafficLightState.Green,
    }
    return table[state_name]


def carla_rgb_to_bgr(gen, image):
    if hasattr(gen, "carla_rgb_to_bgr"):
        return gen.carla_rgb_to_bgr(image)

    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    bgr = arr[:, :, :3].copy()
    return bgr


def get_weather_rain(weather_mode):
    return RAIN_BY_MODE.get(weather_mode, 25.0)


def write_video(frames_bgr, out_path, fps):
    if not frames_bgr:
        raise RuntimeError("No frames to write.")

    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {out_path}")

    for frame in frames_bgr:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)

    writer.release()


# ============================================================
# RGB SENSOR BUFFER
# ============================================================

class LatestImageBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None
        self.latest_frame = None

    def callback(self, data):
        with self.lock:
            self.latest = data
            self.latest_frame = int(data.frame)

    def get_after(self, frame_id, timeout_s=2.0):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            with self.lock:
                if self.latest is not None and self.latest_frame is not None:
                    if self.latest_frame >= frame_id:
                        return self.latest
            time.sleep(0.002)
        return None

    def clear(self):
        with self.lock:
            self.latest = None
            self.latest_frame = None


# ============================================================
# RGB SENSOR ONLY
# ============================================================

def spawn_rgb_sensor_on_spectator(gen, world, blueprint_library, image_w, image_h, fov):
    spectator = world.get_spectator()

    rgb_bp = blueprint_library.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(image_w))
    rgb_bp.set_attribute("image_size_y", str(image_h))
    rgb_bp.set_attribute("fov", str(fov))
    rgb_bp.set_attribute("motion_blur_intensity", "0.0")
    rgb_bp.set_attribute("sensor_tick", "0.0")

    attached = False

    try:
        sensor = world.spawn_actor(
            rgb_bp,
            gen.carla.Transform(),
            attach_to=spectator
        )
        attached = True
    except Exception:
        sensor = world.spawn_actor(
            rgb_bp,
            spectator.get_transform()
        )
        attached = False

    return sensor, attached


def tick_and_get_rgb(gen, world, rgb_sensor, attached, rgb_buf, follower=None, timeout_s=2.0):
    if follower is not None:
        follower.step()

    if not attached and rgb_sensor is not None:
        try:
            rgb_sensor.set_transform(world.get_spectator().get_transform())
        except Exception:
            pass

    frame_id = int(world.tick())
    rgb_data = rgb_buf.get_after(frame_id, timeout_s=timeout_s)

    if rgb_data is None:
        return None

    return carla_rgb_to_bgr(gen, rgb_data)


def warmup_sensor(gen, world, rgb_sensor, attached, rgb_buf, follower, ticks=30):
    for _ in range(ticks):
        _ = tick_and_get_rgb(
            gen,
            world,
            rgb_sensor,
            attached,
            rgb_buf,
            follower=follower,
            timeout_s=2.0
        )
    rgb_buf.clear()


# ============================================================
# COMPATIBILITY WRAPPERS
# ============================================================

def find_visible_tl(gen, world, blueprint_library, start_state, end_state, speed_kmh, bin_ms):
    try:
        return gen.find_visible_validated_tl(
            world,
            blueprint_library,
            start_state,
            end_state,
            speed_kmh,
            bin_ms
        )
    except TypeError:
        return gen.find_visible_validated_tl(
            world,
            blueprint_library,
            start_state
        )


def get_projected_bbox(gen, traffic_light, world, K):
    try:
        bbox, info = gen.get_projected_tl_bbox(
            traffic_light,
            world.get_spectator().get_transform(),
            K,
        )
        return bbox
    except Exception:
        return None


def score_final_visibility(gen, rgb_bgr, bbox, end_state):
    try:
        score, bright_px, color_info = gen.score_roi_for_state(
            rgb_bgr,
            bbox,
            end_state,
        )
        return float(score), int(bright_px), color_info
    except Exception:
        return 1.0, 999, {"note": "score_unavailable"}


# ============================================================
# SINGLE SAMPLE CAPTURE
# ============================================================

def capture_one_sample(
    gen,
    world,
    blueprint_library,
    output_dir,
    weather_mode,
    speed_kmh,
    start_state,
    end_state,
    transition_label,
    args,
):
    ensure_dir(output_dir)

    profile = gen.choose_profile(speed_kmh)
    bin_ms = gen.default_bin_ms(speed_kmh)

    gen.set_sync_mode(world, profile["world_fps"])
    gen.apply_weather(world, get_weather_rain(weather_mode), weather_mode)

    traffic_light = None
    rgb_sensor = None

    try:
        traffic_light, selected = find_visible_tl(
            gen,
            world,
            blueprint_library,
            start_state,
            end_state,
            speed_kmh,
            bin_ms,
        )

        adaptive_fov = selected["adaptive_fov"]
        approach_wp = selected["approach_wp"]

        gen.place_spectator_camera(world, selected["transform"])

        rgb_sensor, attached = spawn_rgb_sensor_on_spectator(
            gen,
            world,
            blueprint_library,
            args.image_width,
            args.image_height,
            adaptive_fov,
        )

        rgb_buf = LatestImageBuffer()
        rgb_sensor.listen(rgb_buf.callback)

        follower = gen.SpectatorPathFollower(world, approach_wp, speed_kmh)

        warmup_sensor(
            gen,
            world,
            rgb_sensor,
            attached,
            rgb_buf,
            follower,
            ticks=args.sensor_warmup_ticks,
        )

        actual_start = gen.force_light_state(
            world,
            traffic_light,
            state_to_carla(gen, start_state),
        )

        # Stabilise at start state
        for _ in range(args.state_warmup_ticks):
            _ = tick_and_get_rgb(
                gen,
                world,
                rgb_sensor,
                attached,
                rgb_buf,
                follower=follower,
                timeout_s=2.0,
            )

        rgb_buf.clear()

        K = gen.build_projection_matrix(
            args.image_width,
            args.image_height,
            adaptive_fov,
        )

        roi_frames = []
        bbox_records = []
        frame_labels = []

        def capture_roi_frame(label_name):
            rgb_bgr = tick_and_get_rgb(
                gen,
                world,
                rgb_sensor,
                attached,
                rgb_buf,
                follower=follower,
                timeout_s=2.0,
            )

            if rgb_bgr is None:
                return None, None, None

            bbox = get_projected_bbox(gen, traffic_light, world, K)
            if bbox is None:
                bbox = selected.get("coarse_bbox")

            if bbox is None:
                return None, None, None

            roi, padded_bbox = crop_with_pad(
                rgb_bgr,
                bbox,
                pad=args.roi_pad,
            )

            if roi is None or roi.size == 0:
                return None, None, None

            roi = resize_roi(
                roi,
                args.roi_width,
                args.roi_height,
            )

            return roi, padded_bbox, rgb_bgr

        # Pre-transition frames
        last_rgb_full = None
        last_bbox = None

        for i in range(args.pre_frames):
            roi, padded_bbox, rgb_full = capture_roi_frame(f"pre_{i:03d}")

            if roi is None:
                raise RuntimeError(f"Failed to capture pre frame {i}")

            roi_frames.append(roi)
            bbox_records.append(padded_bbox)
            frame_labels.append(start_state)
            last_rgb_full = rgb_full
            last_bbox = padded_bbox

        actual_end = gen.force_light_state(
            world,
            traffic_light,
            state_to_carla(gen, end_state),
        )

        # Post-transition frames
        for i in range(args.post_frames):
            roi, padded_bbox, rgb_full = capture_roi_frame(f"post_{i:03d}")

            if roi is None:
                raise RuntimeError(f"Failed to capture post frame {i}")

            roi_frames.append(roi)
            bbox_records.append(padded_bbox)
            frame_labels.append(end_state)
            last_rgb_full = rgb_full
            last_bbox = padded_bbox

        if not roi_frames:
            raise RuntimeError("No ROI frames captured.")

        video_path = Path(output_dir) / "rgb_transition_roi_video.mp4"
        write_video(roi_frames, video_path, fps=args.video_fps)

        final_score, final_bright_px, color_info = score_final_visibility(
            gen,
            last_rgb_full,
            last_bbox,
            end_state,
        )

        quality_ok = bool(final_bright_px >= args.min_bright_px)

        meta = {
            "event_source": "v2e_from_rgb_video",
            "rgb_video_file": "rgb_transition_roi_video.mp4",
            "weather_mode": weather_mode,
            "rain_intensity": get_weather_rain(weather_mode),
            "speed_kmh": float(speed_kmh),
            "start_state": start_state,
            "end_state": end_state,
            "transition_label": transition_label,
            "actual_start_state": gen.state_to_name(actual_start),
            "actual_end_state": gen.state_to_name(actual_end),
            "pre_frames": int(args.pre_frames),
            "post_frames": int(args.post_frames),
            "total_frames": int(len(roi_frames)),
            "transition_after_frame_index": int(args.pre_frames - 1),
            "frame_gt_labels": frame_labels,
            "video_fps": float(args.video_fps),
            "image_width": int(args.image_width),
            "image_height": int(args.image_height),
            "roi_width": int(args.roi_width),
            "roi_height": int(args.roi_height),
            "roi_pad": int(args.roi_pad),
            "bbox_padded_xyxy_per_frame": bbox_records,
            "traffic_light_id": int(traffic_light.id),
            "adaptive_fov_deg": float(adaptive_fov),
            "distance_m": float(selected.get("actual_dist", -1.0)),
            "profile": profile,
            "quality_ok": quality_ok,
            "quality_issues": [] if quality_ok else ["low_final_rgb_visibility"],
            "final_rgb_validation_score": final_score,
            "final_rgb_validation_bright_pixels": final_bright_px,
            "final_rgb_validation_color_info": color_info,
            "note": "Minimal RGB-only CARLA clip for later v2e conversion. No CARLA DVS sensor used.",
        }

        meta_path = Path(output_dir) / "capture_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return quality_ok, meta

    finally:
        if traffic_light is not None:
            try:
                traffic_light.freeze(False)
            except Exception:
                pass

        if rgb_sensor is not None:
            try:
                rgb_sensor.stop()
            except Exception:
                pass

            try:
                rgb_sensor.destroy()
            except Exception:
                pass

        try:
            gen.reset_async_mode(world)
        except Exception:
            pass


# ============================================================
# MAIN COLLECTION LOOP
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fast RGB-only CARLA transition clip collector for v2e DVS dataset generation."
    )

    parser.add_argument(
        "--generator-script",
        type=str,
        default="./recommended_dvs_transition_roi_matrix_harsh_conditions.py",
    )
    parser.add_argument("--output", type=str, default="./v2e_transition_dataset_raw")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", type=str, default="Town01")

    parser.add_argument("--samples-per-cell", type=int, default=5)
    parser.add_argument("--max-attempts-per-cell", type=int, default=50)

    parser.add_argument("--speeds", nargs="+", type=float, default=DEFAULT_SPEEDS)
    parser.add_argument("--weather-modes", nargs="+", default=DEFAULT_WEATHER_MODES)

    parser.add_argument("--pre-frames", type=int, default=10)
    parser.add_argument("--post-frames", type=int, default=20)
    parser.add_argument("--video-fps", type=float, default=60.0)

    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)

    parser.add_argument("--roi-width", type=int, default=96)
    parser.add_argument("--roi-height", type=int, default=160)
    parser.add_argument("--roi-pad", type=int, default=8)

    parser.add_argument("--sensor-warmup-ticks", type=int, default=30)
    parser.add_argument("--state-warmup-ticks", type=int, default=20)
    parser.add_argument("--min-bright-px", type=int, default=20)

    args = parser.parse_args()

    output_root = Path(args.output)
    ensure_dir(output_root)

    gen = load_generator_module(args.generator_script)

    client = gen.connect_carla(args.host, args.port)
    client.set_timeout(60.0)

    print("[INFO] Loading world:", args.map, flush=True)
    world = client.load_world(args.map)
    blueprint_library = world.get_blueprint_library()

    manifest = []
    stats = {}

    total_cells = len(args.weather_modes) * len(args.speeds) * len(DEFAULT_TRANSITIONS)
    target_total = total_cells * args.samples_per_cell

    print("\n" + "=" * 100)
    print("FAST V2E RGB DATASET COLLECTION")
    print("=" * 100)
    print("Output:", output_root.resolve())
    print("Weather modes:", args.weather_modes)
    print("Speeds:", args.speeds)
    print("Transitions:", [t[2] for t in DEFAULT_TRANSITIONS])
    print("Samples per cell:", args.samples_per_cell)
    print("Target total samples:", target_total)
    print("Saved per sample: rgb_transition_roi_video.mp4 + capture_meta.json")
    print("=" * 100)

    accepted_total = 0
    cell_counter = 0

    try:
        for weather_mode in args.weather_modes:
            for speed_kmh in args.speeds:
                for start_state, end_state, transition_label in DEFAULT_TRANSITIONS:
                    cell_counter += 1

                    cell_id = f"{weather_mode}__{int(speed_kmh)}kmh__{transition_label}"
                    stats[cell_id] = {
                        "accepted": 0,
                        "attempts": 0,
                        "failures": 0,
                    }

                    print("\n" + "#" * 100)
                    print(f"[CELL {cell_counter}/{total_cells}] {cell_id}")
                    print("#" * 100)

                    sample_idx = 1

                    while stats[cell_id]["accepted"] < args.samples_per_cell:
                        if stats[cell_id]["attempts"] >= args.max_attempts_per_cell:
                            print(f"[WARN] Max attempts reached for {cell_id}", flush=True)
                            break

                        output_dir = (
                            output_root
                            / weather_mode
                            / f"{int(speed_kmh)}kmh"
                            / transition_label
                            / f"sample_{sample_idx:04d}"
                        )

                        video_path = output_dir / "rgb_transition_roi_video.mp4"
                        meta_path = output_dir / "capture_meta.json"

                        if video_path.exists() and meta_path.exists():
                            print(f"[SKIP] Existing sample: {output_dir}", flush=True)

                            try:
                                with open(meta_path, "r") as f:
                                    meta = json.load(f)
                                manifest.append(meta)
                            except Exception:
                                pass

                            stats[cell_id]["accepted"] += 1
                            accepted_total += 1
                            sample_idx += 1
                            continue

                        stats[cell_id]["attempts"] += 1

                        print(
                            f"[TRY] {cell_id} "
                            f"accepted={stats[cell_id]['accepted']}/{args.samples_per_cell} "
                            f"attempt={stats[cell_id]['attempts']}",
                            flush=True,
                        )

                        try:
                            quality_ok, meta = capture_one_sample(
                                gen=gen,
                                world=world,
                                blueprint_library=blueprint_library,
                                output_dir=str(output_dir),
                                weather_mode=weather_mode,
                                speed_kmh=speed_kmh,
                                start_state=start_state,
                                end_state=end_state,
                                transition_label=transition_label,
                                args=args,
                            )

                            if not quality_ok:
                                stats[cell_id]["failures"] += 1
                                print(f"[DROP] quality issue: {output_dir}", flush=True)
                                continue

                            meta["sample_path"] = str(
                                output_dir.relative_to(output_root)
                            ).replace("\\", "/")

                            with open(meta_path, "w") as f:
                                json.dump(meta, f, indent=2)

                            manifest.append(meta)
                            stats[cell_id]["accepted"] += 1
                            accepted_total += 1

                            print(
                                f"[KEEP] {cell_id} "
                                f"accepted={stats[cell_id]['accepted']}/{args.samples_per_cell} "
                                f"total={accepted_total}/{target_total}",
                                flush=True,
                            )

                            sample_idx += 1
                            time.sleep(0.3)

                        except Exception as e:
                            stats[cell_id]["failures"] += 1
                            print(f"[DROP] exception: {e}", flush=True)
                            time.sleep(1.0)

        summary = {
            "dataset_type": "CARLA_RGB_transition_clips_for_v2e",
            "output_root": str(output_root),
            "map": args.map,
            "total_accepted": accepted_total,
            "target_total": target_total,
            "weather_modes": args.weather_modes,
            "speeds": args.speeds,
            "transitions": [
                {
                    "start_state": s,
                    "end_state": e,
                    "label": label,
                }
                for s, e, label in DEFAULT_TRANSITIONS
            ],
            "samples_per_cell": args.samples_per_cell,
            "pre_frames": args.pre_frames,
            "post_frames": args.post_frames,
            "video_fps": args.video_fps,
            "roi_width": args.roi_width,
            "roi_height": args.roi_height,
            "roi_pad": args.roi_pad,
            "collection_stats": stats,
            "samples": manifest,
            "next_step": "Upload this folder to Colab/Kaggle and batch convert rgb_transition_roi_video.mp4 with v2e.",
        }

        with open(output_root / "dataset_manifest.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 100)
        print("[DONE]")
        print("Accepted samples:", accepted_total)
        print("Manifest:", output_root / "dataset_manifest.json")
        print("=" * 100)

    finally:
        try:
            gen.reset_async_mode(world)
        except Exception:
            pass


if __name__ == "__main__":
    main()