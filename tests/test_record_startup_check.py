"""一時ファイルと通信機能のない模擬ロボットで、開始前の設定確認を試す。"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from src.control.action_guard import ActionGuard, MOTOR_IDS
from src.control.dataset_record_writer import DatasetRecordWriter
from src.control.record_startup_check import RecordStartupCheck, StartupCheckError
from src.control.safe_record_loop import RecordState, RecordStateError, SafeRecordLoop, SendStatus
from src.control.safe_record_session import SafeRecordSession, SessionState, SessionStateError

JOINTS = tuple(f"{motor}.pos" for motor in MOTOR_IDS)


@dataclass
class FakeConfig:
    use_degrees: object = True


@dataclass
class FakeRobot:
    config: Any
    calibration: Any
    sent: list[dict[str, float]] = field(default_factory=list)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.sent.append(dict(action))
        return dict(action)

    def connect(self) -> None:
        pytest.fail("Startup must never connect hardware")

    def disconnect(self) -> None:
        pytest.fail("Startup must never disconnect/release torque")


@dataclass
class FakeDataset:
    features: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        key: {"dtype": "float32", "shape": (6,), "names": list(JOINTS)}
        for key in ("action", "observation.state")
    })
    pending: list[dict[str, Any]] = field(default_factory=list)
    episodes: list[list[dict[str, Any]]] = field(default_factory=list)

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.pending.append(frame)

    def save_episode(self, episode_data: dict | None = None, parallel_encoding: bool = True) -> None:
        self.episodes.append(list(self.pending))
        self.pending.clear()

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        self.pending.clear()

    def finalize(self) -> None:
        pass


@dataclass
class StartupCase:
    path: Path
    calibration: dict[str, dict[str, int]]
    check: RecordStartupCheck
    guard: ActionGuard
    robot: FakeRobot


@pytest.fixture
def case(tmp_path: Path) -> StartupCase:
    ranges = {
        "shoulder_pan": (792, 3306), "shoulder_lift": (826, 3209),
        "elbow_flex": (846, 3047), "wrist_flex": (800, 3140),
        "wrist_roll": (0, 4095), "gripper": (2035, 3481),
    }
    calibration = {
        motor: {"id": motor_id, "drive_mode": 0, "homing_offset": 0,
                "range_min": ranges[motor][0], "range_max": ranges[motor][1]}
        for motor, motor_id in MOTOR_IDS.items()
    }
    payload = json.dumps(calibration).encode("utf-8")
    path = tmp_path / "current_follower.json"
    path.write_bytes(payload)
    check = RecordStartupCheck(
        calibration_path=path, expected_sha256=hashlib.sha256(payload).hexdigest(),
        margin_deg=2.0, margin_gripper=3.0,
    )
    guard = ActionGuard.from_calibration(path, margin_deg=2.0, margin_gripper=3.0)
    return StartupCase(path, calibration, check, guard, FakeRobot(FakeConfig(), copy.deepcopy(calibration)))


def session_for(case: StartupCase, *, checked: bool = True) -> SafeRecordSession:
    writer = DatasetRecordWriter(FakeDataset(), "Offline startup test")
    loop = SafeRecordLoop(case.guard, lambda payload: payload[0], lambda payload: payload[0], case.robot, writer)
    return SafeRecordSession(loop, writer, startup_check=case.check if checked else None)


def assert_start_rejected(case: StartupCase, match: str) -> SafeRecordSession:
    session = session_for(case)
    with pytest.raises(StartupCheckError, match=match):
        session.start()
    assert session.state is SessionState.FAULT
    assert session.loop.state is RecordState.FAULT
    assert session.startup_result is None
    assert session.fault is not None
    assert session.fault.stage == "startup_check"
    assert session.fault.send_status is SendStatus.NOT_ATTEMPTED
    assert not session.fault.commit_uncertain
    action = {key: 50.0 if key == "gripper.pos" else 0.0 for key in JOINTS}
    for operation in (session.start, session.save_episode, lambda: session.record_frame(action, action)):
        with pytest.raises(SessionStateError):
            operation()
    with pytest.raises(RecordStateError):
        session.loop.process_frame(action, action)
    assert case.robot.sent == []
    assert session.writer.frames_in_episode == 0
    assert session.writer.dataset.pending == session.writer.dataset.episodes == []
    return session


def test_matching_startup_records_evidence_without_modifying_inputs(case: StartupCase) -> None:
    before_bytes = case.path.read_bytes()
    before_loaded = copy.deepcopy(case.robot.calibration)
    session = session_for(case)
    session.start()
    assert session.state is SessionState.RECORDING
    assert session.startup_result is not None
    assert session.startup_result.calibration_sha256 == hashlib.sha256(before_bytes).hexdigest()
    assert session.startup_result.margin_deg == 2.0
    assert session.startup_result.margin_gripper == 3.0
    assert session.startup_result.bounds == case.guard.bounds
    assert case.robot.sent == []  # 開始前の検査に通っても、指令は送信されないことを確認する。
    action = {key: 50.0 if key == "gripper.pos" else 0.0 for key in JOINTS}
    session.record_frame(action, action)
    session.save_episode()
    session.finalize()
    assert len(case.robot.sent) == 1
    assert case.path.read_bytes() == before_bytes
    assert case.robot.calibration == before_loaded


def test_legacy_offline_session_has_no_startup_evidence(case: StartupCase) -> None:
    session = session_for(case, checked=False)
    session.start()
    assert session.startup_result is None
    assert case.robot.sent == []
    session.abort_episode()
    session.finalize()


def test_changed_file_after_session_construction_rejects_fixed_hash(case: StartupCase) -> None:
    session = session_for(case)
    # 空白だけの変更でも、事前に確認したファイルと同一とはみなさない。
    case.path.write_bytes(case.path.read_bytes() + b"\n")
    with pytest.raises(StartupCheckError, match="SHA-256 mismatch"):
        session.start()
    assert session.state is SessionState.FAULT and session.startup_result is None
    assert case.robot.sent == []


def test_startup_hash_and_guard_use_one_file_read(case: StartupCase, monkeypatch: pytest.MonkeyPatch) -> None:
    read_bytes = Path.read_bytes
    reads = 0

    def read_once(path: Path) -> bytes:
        nonlocal reads
        if path == case.path:
            reads += 1
            assert reads == 1, "Hash and guard must not read different file revisions"
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_once)
    session_for(case).start()
    assert reads == 1 and case.robot.sent == []


def test_missing_file_rejects_start_and_can_explicitly_abort(case: StartupCase) -> None:
    case.check = replace(case.check, calibration_path=case.path.parent / "missing.json")
    session = assert_start_rejected(case, "Cannot read")
    session.abort_episode()
    session.finalize()
    assert session.state is SessionState.FINALIZED


@pytest.mark.parametrize("payload", [b"{", b"null", b"{}", b"\xff"])
def test_malformed_calibration_rejects_even_with_matching_hash(case: StartupCase, payload: bytes) -> None:
    case.path.write_bytes(payload)
    case.check = replace(case.check, expected_sha256=hashlib.sha256(payload).hexdigest())
    assert_start_rejected(case, "Invalid calibration")


@pytest.mark.parametrize("digest", ["", "a" * 63, "a" * 65, "g" * 64, None])
def test_invalid_expected_digest_is_not_accepted(case: StartupCase, digest: Any) -> None:
    with pytest.raises(ValueError, match="64-character"):
        replace(case.check, expected_sha256=digest)
    assert case.robot.sent == []


@pytest.mark.parametrize("use_degrees", [False, None, 1, "true"])
def test_incompatible_or_ambiguous_units_reject_start(case: StartupCase, use_degrees: object) -> None:
    case.robot.config.use_degrees = use_degrees
    assert_start_rejected(case, "use_degrees")


def test_missing_sender_configuration_rejects_start(case: StartupCase) -> None:
    case.robot.config = None
    assert_start_rejected(case, "use_degrees")


@pytest.mark.parametrize("joint, attribute", [
    ("shoulder_lift", "range_min"), ("wrist_flex", "range_max"),
    ("shoulder_pan", "homing_offset"), ("wrist_roll", "drive_mode"), ("gripper", "id"),
])
def test_stale_loaded_calibration_rejects_even_when_guard_and_file_match(
    case: StartupCase, joint: str, attribute: str,
) -> None:
    case.robot.calibration[joint][attribute] += 1
    assert_start_rejected(case, "Loaded Follower calibration mismatch")


@pytest.mark.parametrize("loaded", [None, {}, {"leader": {}}, {motor: {} for motor in MOTOR_IDS}])
def test_missing_or_malformed_loaded_calibration_rejects_start(case: StartupCase, loaded: Any) -> None:
    case.robot.calibration = loaded
    assert_start_rejected(case, "calibration")


def test_boolean_calibration_field_is_not_an_integer(case: StartupCase) -> None:
    case.robot.calibration["shoulder_pan"]["id"] = True
    assert_start_rejected(case, "shoulder_pan.id")


@pytest.mark.parametrize("joint", ["shoulder_lift.pos", "wrist_flex.pos"])
def test_guard_range_expansion_is_rejected(case: StartupCase, joint: str) -> None:
    case.guard = ActionGuard(tuple(
        replace(bound, maximum=bound.maximum + 0.01) if bound.action_key == joint else bound
        for bound in case.guard.bounds
    ))
    assert_start_rejected(case, "ActionGuard ranges")


@pytest.mark.parametrize("unit", ["radian", "normalized_m100_100"])
def test_guard_unit_mismatch_is_rejected(case: StartupCase, unit: str) -> None:
    case.guard = ActionGuard(tuple(
        replace(bound, unit=unit) if bound.action_key == "wrist_flex.pos" else bound
        for bound in case.guard.bounds
    ))
    assert_start_rejected(case, "units")


def test_different_margin_policy_is_rejected(case: StartupCase) -> None:
    case.check = replace(case.check, margin_deg=1.0)
    assert_start_rejected(case, "margins")


@pytest.mark.parametrize("margin", [-1.0, float("nan"), float("inf"), True, 1000.0])
def test_invalid_margin_policy_rejects_start(case: StartupCase, margin: Any) -> None:
    case.check = replace(case.check, margin_gripper=margin)
    assert_start_rejected(case, "Invalid calibration or margins")


def test_reordered_guard_joints_are_accepted(case: StartupCase) -> None:
    case.guard = ActionGuard(tuple(reversed(case.guard.bounds)))
    result = case.check.verify(case.guard, case.robot)
    assert {bound.action_key for bound in result.bounds} == set(JOINTS)


def test_hash_case_does_not_change_digest_meaning(case: StartupCase) -> None:
    case.check = replace(case.check, expected_sha256=case.check.expected_sha256.upper())
    assert case.check.verify(case.guard, case.robot).calibration_sha256 == case.check.expected_sha256.lower()


def test_next_episode_rechecks_sender_metadata(case: StartupCase) -> None:
    first = session_for(case)
    first.start()
    action = {key: 50.0 if key == "gripper.pos" else 0.0 for key in JOINTS}
    first.record_frame(action, action)
    first.save_episode()
    case.robot.config.use_degrees = False
    loop = SafeRecordLoop(case.guard, lambda payload: payload[0], lambda payload: payload[0], case.robot, first.writer)
    second = SafeRecordSession(loop, first.writer, startup_check=case.check)
    with pytest.raises(StartupCheckError, match="use_degrees"):
        second.start()
    assert second.state is SessionState.FAULT and second.startup_result is None
    assert len(case.robot.sent) == 1
    assert first.writer.saved_episodes == 1
    second.abort_episode()
    second.finalize()
    assert len(first.writer.dataset.episodes) == 1


def test_lerobot_metadata_and_dataset_supported_without_real_robot(case: StartupCase, tmp_path: Path) -> None:
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.motors import MotorCalibration
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

    # 設定の入れ物だけを作る。SOFollower、通信機能、ポート、カメラは作らない。
    case.robot.config = SO101FollowerConfig(port="offline_unused", use_degrees=True)
    case.robot.calibration = {
        motor: MotorCalibration(**entry) for motor, entry in case.calibration.items()
    }
    ds = LeRobotDataset.create(
        repo_id="local/so101_offline_startup_test", fps=30, features=FakeDataset().features,
        root=tmp_path / "startup_dataset", robot_type="so_follower", use_videos=False,
    )
    try:
        writer = DatasetRecordWriter(ds, "Offline startup test")
        loop = SafeRecordLoop(case.guard, lambda payload: payload[0], lambda payload: payload[0], case.robot, writer)
        session = SafeRecordSession(loop, writer, startup_check=case.check)
        session.start()
        assert session.startup_result is not None and session.startup_result.bounds == case.guard.bounds
        for value in (45.123456, 55.654321):
            action = {key: value if key == "gripper.pos" else 0.0 for key in JOINTS}
            session.record_frame(action, action)
        session.save_episode()
        session.finalize()
        paths = list((ds.root / "data").rglob("*.parquet"))
        assert len(paths) == 1
        actions = pq.read_table(paths[0], columns=["action"])["action"].to_pylist()
        for stored, sent in zip(actions, case.robot.sent, strict=True):
            assert stored == pytest.approx([sent[key] for key in JOINTS], rel=1e-7, abs=1e-6)
        assert len(actions) == len(case.robot.sent) == 2
    finally:
        ds.finalize()
