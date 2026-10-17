"""正常終了したテレオペログを図・CSV・姿勢候補へ整理する。実機接続・再生は行わない。"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

from src.control.action_guard import MOTOR_IDS


ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("leader", "sent_target", "observation")


def parse_frames(text):
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if (not records or records[-1].get("event") != "complete"
            or records[-1].get("executed") is not True
            or any(row.get("event") == "fault" for row in records)):
        raise ValueError("Use an executed teleop log that completed without FAULT")
    plans = [row for row in records if row.get("event") == "plan"]
    releases = [row for row in records if row.get("event") == "torque_off"]
    if (len(plans) != 1 or len(releases) != 1
            or set(releases[0].get("verified", [])) != set(MOTOR_IDS)
            or releases[0].get("unconfirmed") != [] or releases[0].get("failures")):
        raise ValueError("Expected one plan and confirmed torque OFF for all six axes")
    frames = [row for row in records if row.get("event") == "frame"]
    if not frames:
        raise ValueError("No teleop frames")
    previous = None
    for frame in frames:
        timestamp = frame["read_started_at"]
        if (type(timestamp) not in (int, float) or not math.isfinite(timestamp)
                or (previous is not None and timestamp <= previous)):
            raise ValueError("Expected increasing finite observation timestamps")
        previous = timestamp
        for field in FIELDS:
            if set(frame[field]) != {name+".pos" for name in MOTOR_IDS}:
                raise ValueError(f"Expected six joint values in {field}")
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in frame[field].values()):
                raise ValueError(f"Expected finite joint values in {field}")
    return plans[0], frames


def load_frames(path):
    return parse_frames(path.read_text(encoding="utf-8"))


def select_poses(frames, requests):
    times = [row["read_started_at"]-frames[0]["read_started_at"] for row in frames]
    poses = []
    names = set()
    for request in requests:
        name, separator, seconds = request.partition("=")
        if not separator or not name or name in names:
            raise ValueError("Use unique --pose NAME=SECONDS values")
        seconds = float(seconds)
        if not math.isfinite(seconds) or not 0 <= seconds <= times[-1]:
            raise ValueError(f"Pose time must be between 0 and {times[-1]:.3f} seconds")
        index = min(range(len(frames)), key=lambda i: abs(times[i]-seconds))
        row = frames[index]
        poses.append({"name": name, "requested_time_s": seconds, "sampled_time_s": times[index],
                      "frame_index": row["index"], "observation": row["observation"],
                      "sent_target": row["sent_target"]})
        names.add(name)
    return poses


def save_plot(frames, poses, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = [row["read_started_at"]-frames[0]["read_started_at"] for row in frames]
    figure, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, layout="constrained")
    series = (("leader", "Leader", "#9ca3af", ":"),
              ("sent_target", "Sent target", "#d97706", "--"),
              ("observation", "Follower observed", "#2563eb", "-"))
    try:
        for axis, (motor, motor_id) in zip(axes.flat, MOTOR_IDS.items()):
            for field, label, color, style in series:
                axis.plot(times, [row[field][motor+".pos"] for row in frames],
                          label=label, color=color, linestyle=style, linewidth=1.5)
            for pose in poses:
                axis.axvline(pose["sampled_time_s"], color="#7c3aed", linewidth=0.8, alpha=0.6)
                if motor == "gripper":
                    axis.text(pose["sampled_time_s"], 0.98, pose["name"],
                              transform=axis.get_xaxis_transform(), rotation=90,
                              va="top", ha="right", fontsize=8, color="#7c3aed")
            unit = "points (0-100)" if motor == "gripper" else "degrees"
            axis.set_title(f"F{motor_id} {motor}")
            axis.set_ylabel(unit)
            axis.grid(alpha=0.25)
        for axis in axes[-1]:
            axis.set_xlabel("Seconds from first frame observation")
        axes[0, 0].legend(loc="best", fontsize=8)
        figure.suptitle("Successful teleop: source for fixed-motion preparation")
        figure.savefig(path, dpi=150)
    finally:
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="新規の保存先（省略時は日時付き）")
    parser.add_argument("--pose", action="append", default=[], metavar="NAME=SECONDS",
                        help="最も近いフレームを姿勢候補として抽出。複数指定可。時刻は最初の観測から")
    args = parser.parse_args(argv)
    try:
        plan, frames = load_frames(args.log)
        poses = select_poses(frames, args.pose)
        output = args.output_dir or ROOT / "outputs" / f"motion_review_{datetime.now():%Y%m%d_%H%M%S_%f}"
        output.mkdir(parents=True, exist_ok=False)
        times = [row["read_started_at"]-frames[0]["read_started_at"] for row in frames]
        with (output / "joint_timeline.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time_s", "frame_index"] +
                            [f"{field}.{motor}" for field in FIELDS for motor in MOTOR_IDS])
            for seconds, row in zip(times, frames):
                writer.writerow([seconds, row["index"]] +
                                [row[field][motor+".pos"] for field in FIELDS for motor in MOTOR_IDS])
        report = {"source_log": str(args.log), "execution_ready": False,
                  "frames": len(frames), "duration_s": times[-1],
                  "follower_initial": plan["follower_initial"], "poses": poses,
                  "note": "Review candidates only. Observations precede their frame's command; "
                          "grasp/release times and physical paths require review."}
        (output / "selected_poses.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        save_plot(frames, poses, output / "joint_timeline.png")
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"ログ整理に失敗しました: {exc}", file=sys.stderr)
        return 1
    print(f"Frames: {len(frames)}, duration: {times[-1]:.2f} s")
    for name in ("joint_timeline.png", "joint_timeline.csv", "selected_poses.json"):
        print("保存先:", (output / name).resolve())
    for pose in poses:
        print(f"姿勢候補 {pose['name']}: {pose['sampled_time_s']:.2f} s / frame {pose['frame_index']}")
    print("実機接続・指令送信は行っていません。抽出した姿勢は再生用の確定データではありません。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
