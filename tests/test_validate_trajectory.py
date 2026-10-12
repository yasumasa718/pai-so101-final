"""Unit and small integration tests for the offline SO-101 validator."""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from src.control import validate_trajectory as vt


@pytest.fixture
def sample() -> tuple[vt.TrajectoryData, tuple[vt.JointMapping, ...], dict[str, vt.MotorCalibration]]:
    names = tuple(f"{name}.pos" for name in vt.EXPECTED_MOTOR_IDS)
    metadata = vt.DatasetMetadata("v3.0", "so_follower", 30.0, names, names)
    calibration = {
        name: vt.MotorCalibration(
            id=motor_id,
            drive_mode=0,
            homing_offset=0,
            range_min=833 if name == "wrist_flex" else 0,
            range_max=3196 if name == "wrist_flex" else 4095,
        )
        for name, motor_id in vt.EXPECTED_MOTOR_IDS.items()
    }
    mapping = vt.validate_joint_mapping(metadata, calibration)
    commands = (
        (0.0, 0.0, 0.0, -50.0, 0.0, 40.0),
        (0.0, 0.0, 0.0, -108.0, 0.0, 45.0),
        (0.0, 0.0, 0.0, -60.0, 0.0, 50.0),
    )
    traj = vt.TrajectoryData(
        parquet_path=Path("fake.parquet"), metadata=metadata,
        actions=commands,
        observation_states=(commands[0], commands[0], commands[1]),
        timestamps=(0.0, 1/30, 2/30),
        frame_indices=(0, 1, 2), episode_indices=(0, 0, 0),
        indices=(0, 1, 2), task_indices=(0, 0, 0),
    )
    return traj, mapping, calibration


def test_limits_and_known_f4_violation(sample):
    traj, mappings, calibration = sample
    limits = vt.calculate_joint_limits(calibration, mappings)
    f4_limits = next(x for x in limits if x.motor_name == "wrist_flex")
    assert f4_limits.minimum == pytest.approx(-103.8681, abs=0.02)
    checks = vt.check_joint_limits(traj, mappings, limits)
    f4_check = next(x for x in checks if x.motor_name == "wrist_flex")
    assert f4_check.configured_range_violations == 1
    assert f4_check.first_problem_row == 1
    assert f4_check.first_problem_timestamp == pytest.approx(1 / 30)
    assert f4_check.max_configured_excess == pytest.approx(108 + f4_limits.minimum)
    assert sum(c.total_problems for c in checks if c.motor_name != "wrist_flex") == 0


def test_integrity_clean(sample):
    traj, _, _ = sample
    quality = vt.check_data_integrity(traj)
    assert quality.passed
    assert quality.max_observed_dt_seconds == pytest.approx(1 / 30)


def test_integrity_catches_missing_nan_time_and_frame(sample):
    traj, _, _ = sample
    bad_actions = list(traj.actions)
    bad_actions[1] = (math.nan, 0, 0, 0, 0, None)
    bad = replace(
        traj, actions=tuple(bad_actions),
        observation_states=(traj.observation_states[0], None, traj.observation_states[2]),
        timestamps=(0.0, 0.0, None),
        frame_indices=(0, 2, 3),
    )
    result = vt.check_data_integrity(bad)
    assert not result.passed
    assert result.issue_counts["action.nonfinite_value"] == 1
    assert result.issue_counts["action.missing_value"] == 1
    assert result.issue_counts["observation.state.missing_vector"] == 1
    assert result.issue_counts["timestamp.not_increasing"] == 1
    assert result.issue_counts["timestamp.invalid"] == 1
    assert result.issue_counts["frame_index.discontinuous"] == 1


def test_integrity_gap_threshold_opt_in(sample):
    traj, _, _ = sample
    assert vt.check_data_integrity(traj).passed
    bad = vt.check_data_integrity(traj, max_gap_seconds=0.03)
    assert bad.issue_counts["timestamp.excessive_gap"] == 2
    with pytest.raises(ValueError):
        vt.check_data_integrity(traj, max_gap_seconds=-1)


def test_jumps_no_threshold_not_evaluated_and_threshold(sample):
    traj, mappings, _ = sample
    diagnostics = vt.check_position_jumps(traj, mappings)
    f4 = next(x for x in diagnostics if x.motor_name == "wrist_flex")
    assert f4.status == "NOT_EVALUATED"
    assert f4.max_step == 58
    checked = vt.check_position_jumps(traj, mappings, max_step_deg=40)
    f4_checked = next(x for x in checked if x.motor_name == "wrist_flex")
    assert f4_checked.status == "FAIL"
    assert f4_checked.step_violations == 2
    assert f4_checked.first_violation_row == 1


def test_tracking_lag_is_descriptive(sample):
    traj, mappings, _ = sample
    results = vt.analyze_tracking_difference(traj, mappings, max_lag_frames=2)
    f4 = [x for x in results if x.motor_name == "wrist_flex"]
    assert f4[0].lag_frames == 0
    assert f4[1].compared_samples == 2
    assert f4[1].mean_abs_difference == pytest.approx(0.0)
    with pytest.raises(ValueError):
        vt.analyze_tracking_difference(traj, mappings, max_lag_frames=-1)


def test_report_paths_protect_source_data(tmp_path, sample):
    traj, mapping, cal = sample
    dataset_root = tmp_path / "dataset"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text("{}")
    parquet = dataset_root / "sample.parquet"
    parquet.write_bytes(b"dummy")
    calibration_path = tmp_path / "cal.json"
    calibration_path.write_text("{}")
    config = vt.ValidationConfig(dataset_root, calibration_path)
    inputs = vt.ValidationInputs(traj.metadata, cal, replace(traj, parquet_path=parquet), mapping)
    limits = vt.calculate_joint_limits(cal, mapping)
    range_checks = vt.check_joint_limits(traj, mapping, limits)
    integrity = vt.check_data_integrity(traj)
    jumps = vt.check_position_jumps(traj, mapping)
    tracking = vt.analyze_tracking_difference(traj, mapping)
    report = vt.build_validation_report(config, inputs, limits, range_checks, integrity, jumps, tracking)
    assert report["verdict"] == "FAIL"
    assert not report["auto_replay_eligible"]
    output = tmp_path / "outputs" / "validation.json"
    vt.save_json_report(report, output, config)
    assert json.loads(output.read_text())["joint_range_checks"][3]["configured_range_violations"] == 1
    with pytest.raises(ValueError):
        vt.save_json_report(report, parquet, config)
    with pytest.raises(ValueError):
        vt.save_json_report(report, calibration_path, config)


def test_cli_on_parquet_and_json_report(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    if not hasattr(pa, "Table"):
        pytest.skip("Full pyarrow is not installed in the test runner")

    dataset = tmp_path / "datasets" / "test"
    (dataset / "meta").mkdir(parents=True)
    dest = dataset / "data" / "chunk-000"
    dest.mkdir(parents=True)
    names = [f"{name}.pos" for name in vt.EXPECTED_MOTOR_IDS]
    features = {
        key: {"dtype": "float32", "shape": [6], "names": names}
        for key in ("action", "observation.state")
    }
    (dataset / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "robot_type": "so_follower",
        "fps": 30, "features": features,
    }))
    calibration = {
        name: {"id": motor_id, "drive_mode": 0, "homing_offset": 0,
               "range_min": 833 if name == "wrist_flex" else 0,
               "range_max": 3196 if name == "wrist_flex" else 4095}
        for name, motor_id in vt.EXPECTED_MOTOR_IDS.items()
    }
    calibration_path = tmp_path / "follower.json"
    calibration_path.write_text(json.dumps(calibration))
    action = [[0,0,0,-50,0,40], [0,0,0,-108,0,40], [0,0,0,-60,0,40]]
    table = pa.table({
        "action": pa.array(action, type=pa.list_(pa.float32(),6)),
        "observation.state": pa.array(action, type=pa.list_(pa.float32(),6)),
        "timestamp": pa.array([0,1/30,2/30], type=pa.float32()),
        "frame_index": pa.array([0,1,2], type=pa.int64()),
        "episode_index": pa.array([0,0,0], type=pa.int64()),
        "index": pa.array([0,1,2], type=pa.int64()),
        "task_index": pa.array([0,0,0], type=pa.int64()),
    })
    pq.write_table(table, dest / "file-000.parquet")
    output = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", [
        "validate_trajectory", "--dataset-root", str(dataset),
        "--calibration", str(calibration_path), "--report-json", str(output),
    ])
    assert vt.main() == 3
    report = json.loads(output.read_text())
    assert report["joint_range_checks"][3]["configured_range_violations"] == 1
    assert report["data_integrity"]["passed"]
    assert report["auto_replay_eligible"] is False


def test_episode_boundary_can_reset_timestamp_and_frame_index(sample):
    traj, _, _ = sample
    new = replace(
        traj,
        timestamps=(0.0, 1/30, 0.0),
        frame_indices=(0, 1, 0),
        episode_indices=(0, 0, 1),
    )
    assert vt.check_data_integrity(new).passed
