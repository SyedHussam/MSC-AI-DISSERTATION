import subprocess
import sys
from pathlib import Path

SCRIPT = "capture_dvs_transition_video.py"
ROOT = Path("./v2e_15_raw")

SPEED = 30
WEATHER = "day_rain"
PRE_FRAMES = 10
POST_FRAMES = 20
TICKS_PER_FRAME = 1
VIDEO_FPS = 60
ROI_PAD = 8
SAMPLES_PER_TRANSITION = 5

TRANSITIONS = [
    ("Red", "Green", "Red_to_Green"),
    ("Green", "Yellow", "Green_to_Yellow"),
    ("Yellow", "Red", "Yellow_to_Red"),
]

def run_command(cmd):
    print("\n" + "=" * 90)
    print("RUNNING:")
    print(" ".join(cmd))
    print("=" * 90)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\nERROR: command failed.")
        print("Stopping so you can check the problem.")
        sys.exit(result.returncode)

def main():
    for start_state, end_state, label in TRANSITIONS:
        for i in range(1, SAMPLES_PER_TRANSITION + 1):
            output_dir = ROOT / label / f"sample_{i:04d}"

            cmd = [
                sys.executable,
                SCRIPT,
                "--output", str(output_dir),
                "--speed", str(SPEED),
                "--start-state", start_state,
                "--end-state", end_state,
                "--weather-mode", WEATHER,
                "--pre-frames", str(PRE_FRAMES),
                "--post-frames", str(POST_FRAMES),
                "--ticks-per-frame", str(TICKS_PER_FRAME),
                "--video-fps", str(VIDEO_FPS),
                "--roi-pad", str(ROI_PAD),
            ]

            run_command(cmd)

    print("\nDONE.")
    print("Captured dataset folder:")
    print(ROOT.resolve())

if __name__ == "__main__":
    main()