"""指令の変化量・変化速度・観測の鮮度・送信所要時間を模擬ロボットで検証する。

FakeClockの時刻を直接動かし、境界ちょうど・超過・逆行を再現する。送信前の拒否と
送信後の異常を区別し、最後に実LeRobotDatasetへの画像付き保存との結合も確認する。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.control.action_guard import ActionGuard, MOTOR_IDS
from src.control.dataset_record_writer import DatasetRecordWriter
from src.control.motion_guard import JointMotionLimit, MotionGuard
from src.control.record_startup_check import RecordStartupCheck
from src.control.safe_record_loop import RecordFault, RecordState, RecordStateError, SafeRecordLoop, SendStatus
from src.control.safe_record_session import SafeRecordSession, SessionState, SessionStateError

JOINTS = tuple(f"{name}.pos" for name in MOTOR_IDS)
CAPTURE = 9.9375  # 2 進数で正確に表せる値を使い、丸め誤差で境界の判定が変わるのを防ぐ。


def positions(**changes: float) -> dict[str, float]:
    """腕0度・爪50ポイントの基準姿勢へ、指定した関節値だけを上書きして返す。"""
    return {**{key: 50.0 if key == "gripper.pos" else 0.0 for key in JOINTS}, **changes}


def policy() -> MotionGuard:
    """テスト専用の上限値を作る。実機で使う推奨値ではない。"""
    return MotionGuard(
        limits=tuple(JointMotionLimit(
            key, "normalized_0_100" if key == "gripper.pos" else "degree",
            4.0 if key == "gripper.pos" else 2.0,
            64.0 if key == "gripper.pos" else 32.0,
        ) for key in JOINTS),
        min_interval_s=1 / 128, max_interval_s=0.25,
        max_observation_age_s=0.125, max_send_duration_s=0.0625,
    )


@dataclass
class FakeClock:
    """nowをそのまま返す模擬時計。テストや模擬送信先が明示的に時刻を進める。"""
    now: float = 10.0

    def __call__(self) -> float:
        return self.now


@dataclass
class FakeConfig:
    """開始前照合に必要なuse_degreesだけを持つ、実機接続のない設定。"""
    use_degrees: bool = True


@dataclass
class FakeRobot:
    """送信履歴・所要時間・例外・爪の戻り値を制御できる送信先。接続と切断は呼ばれたら失敗にする。"""
    clock: FakeClock
    calls: list[dict[str, float]] = field(default_factory=list)
    returned_gripper: float | None = None
    delay: float = 0.0
    error: BaseException | None = None
    calibration: dict[str, Any] = field(default_factory=dict)
    config: FakeConfig = field(default_factory=FakeConfig)

    def send_action(self, candidate: dict[str, float]) -> dict[str, float]:
        self.calls.append(dict(candidate))
        self.clock.now += self.delay
        if self.error is not None:
            raise self.error
        result = dict(candidate)
        if self.returned_gripper is not None:
            result["gripper.pos"] = self.returned_gripper
        return result

    def connect(self) -> None:
        pytest.fail("Motion policy must not connect hardware")

    def disconnect(self) -> None:
        pytest.fail("FAULT must not release torque/disconnect")


@dataclass
class Context:
    """開始済みループと、その送信先・時計・保存履歴を一緒に操作するための入れ物。"""
    loop: SafeRecordLoop
    robot: FakeRobot
    clock: FakeClock
    saved: list[dict[str, float]]


@pytest.fixture
def context() -> Context:
    """校正範囲とテスト用の動作制限を組み合わせ、開始済みループを用意するfixture。"""
    path = Path(__file__).resolve().parents[1] / "configs/calibration_snapshots/my_follower_for_004.json"
    gate = ActionGuard.from_calibration(path)
    clock = FakeClock()
    robot = FakeRobot(clock, calibration=json.loads(path.read_bytes()))
    saved: list[dict[str, float]] = []
    loop = SafeRecordLoop(gate, lambda p: p[0], lambda p: p[0], robot, lambda a: saved.append(dict(a)),
                          motion_guard=policy(), clock=clock)
    loop.start()
    return Context(loop, robot, clock, saved)


def rejected(context: Context, action: dict[str, float], capture: Any = CAPTURE,
             observation: dict[str, Any] | None = None, stage: str = "motion_guard") -> RecordFault:
    """送信前の拒否に共通する条件を検査する。指定段階でFAULTとなり、送信・保存回数が増えず再開もできないことを確かめる。"""
    sends, writes = len(context.robot.calls), len(context.saved)
    with pytest.raises(RecordFault) as caught:
        context.loop.process_frame(action, positions() if observation is None else observation, observation_time=capture)
    fault = caught.value
    assert fault.fault.stage == stage
    assert fault.fault.send_status is SendStatus.NOT_ATTEMPTED
    assert context.loop.state is RecordState.FAULT
    assert len(context.robot.calls) == sends and len(context.saved) == writes
    for operation in (context.loop.start, context.loop.finish,
                      lambda: context.loop.process_frame(positions(), positions(), observation_time=CAPTURE)):
        with pytest.raises(RecordStateError):
            operation()
    assert len(context.robot.calls) == sends
    return fault


def test_valid_multiple_frames_use_real_capture_time_and_returned_target(context: Context) -> None:
    """送信戻り値を記録・履歴に使い、呼出し側が返却辞書を変更しても次回の基準が壊れないことを確認する。"""
    context.robot.returned_gripper = 49.0
    first = context.loop.process_frame(positions(), positions(), observation_time=CAPTURE)
    assert first.sent_action["gripper.pos"] == 49.0
    # 返された結果を呼び出し側で変更しても、次のフレーム用の履歴が変わらないことを確認する。
    first.sent_action["gripper.pos"] = 99.0
    context.robot.returned_gripper = None
    context.clock.now += 0.0625
    second = context.loop.process_frame(positions(**{"gripper.pos": 52.0}), positions(), observation_time=10.03125)
    assert second.frame_index == 1
    assert context.saved[0]["gripper.pos"] == 49.0
    assert context.saved[1]["gripper.pos"] == 52.0
    assert len(context.robot.calls) == context.loop.frames_written == 2


@pytest.mark.parametrize("joint, value", [("shoulder_lift.pos", 2.0), ("wrist_flex.pos", -2.0), ("gripper.pos", 54.0)])
def test_step_and_rate_boundaries_are_inclusive(context: Context, joint: str, value: float) -> None:
    """腕・爪の変化量と速度が上限ちょうどの指令は受理することを確認する。"""
    result = context.loop.process_frame(positions(**{joint: value}), positions(), observation_time=CAPTURE)
    assert result.sent_action[joint] == value


@pytest.mark.parametrize("joint, value", [("shoulder_lift.pos", 2.01), ("wrist_flex.pos", -2.01), ("gripper.pos", 54.01)])
def test_initial_step_violation_is_refused_without_clipping(context: Context, joint: str, value: float) -> None:
    """初回観測からの変化量が上限を少し越えると、入力を補正せず送信前に拒否することを確認する。"""
    action = positions(**{joint: value})
    before = dict(action)
    fault = rejected(context, action, capture=9.875)
    assert "target step" in fault.fault.message
    assert action == before


@pytest.mark.parametrize("joint, value", [("shoulder_lift.pos", 1.0), ("wrist_flex.pos", -1.0), ("gripper.pos", 52.0)])
def test_initial_rate_violation_with_valid_small_step(context: Context, joint: str, value: float) -> None:
    """変化量自体は小さくても経過時間が短く速度上限を越える場合、送信前に拒否することを確認する。"""
    fault = rejected(context, positions(**{joint: value}), capture=9.984375)
    assert "target rate" in fault.fault.message


def test_first_reference_is_measured_observation_not_zero(context: Context) -> None:
    """初回の基準を0度と仮定せず、その場の観測位置に置くことを確認する。"""
    obs = positions(**{"shoulder_lift.pos": 10.0, "gripper.pos": 70.0})
    result = context.loop.process_frame(obs, obs, observation_time=CAPTURE)
    assert result.sent_action == obs


def test_later_step_uses_returned_target_instead_of_previous_request(context: Context) -> None:
    """次回の変化量は前回要求50ではなく送信戻り値46を基準にし、目標51への5ポイント差を拒否することを確認する。"""
    context.robot.returned_gripper = 46.0
    context.loop.process_frame(positions(), positions(), observation_time=CAPTURE)
    context.robot.returned_gripper = None
    context.clock.now += 0.125
    # 前回の要求値 50 との差は 1 だが、実際の戻り値 46 との差は 5 になる。
    fault = rejected(context, positions(**{"gripper.pos": 51.0}), capture=10.0625)
    assert "target step" in fault.fault.message


def test_later_rate_uses_previous_send_return_time(context: Context) -> None:
    """前回の送信開始でなく戻り時刻から速度を評価し、通信待ち時間で許容量を増やさないことを確認する。"""
    context.robot.delay = 0.03125
    context.loop.process_frame(positions(), positions(), observation_time=CAPTURE)
    context.robot.delay = 0.0
    context.clock.now = 10.046875
    # 前回の送信が戻ってからは 1/64 秒しかない。送信開始時刻を使うと誤って検査に通ってしまう。
    fault = rejected(context, positions(**{"wrist_flex.pos": 1.0}), capture=10.0390625)
    assert "target rate" in fault.fault.message


def test_processor_modified_final_candidate_is_checked(context: Context) -> None:
    """ロボット側の加工が作った大きな最終指令も、送信直前の動作検査で拒否することを確認する。"""
    context.loop.robot_action_processor = lambda p: {**p[0], "wrist_flex.pos": 3.0}
    rejected(context, positions())


def test_processor_cannot_replace_initial_measured_reference(context: Context) -> None:
    """加工処理が受け取った観測辞書を変更しても、事前に保持した比較用の実測位置は変わらないことを確認する。"""
    def altered(payload: tuple[dict[str, float], dict[str, Any]]) -> dict[str, float]:
        action, obs = payload
        obs["wrist_flex.pos"] = 3.0
        return {**action, "wrist_flex.pos": 3.0}

    context.loop.robot_action_processor = altered
    rejected(context, positions())


def test_time_is_checked_after_slow_processors(context: Context) -> None:
    """加工処理の間に観測が古くなった場合、加工前の時刻で通さず送信直前に拒否することを確認する。"""
    def slow(payload: tuple[dict[str, float], dict[str, Any]]) -> dict[str, float]:
        context.clock.now += 0.25
        return payload[0]

    context.loop.robot_action_processor = slow
    assert "stale" in rejected(context, positions()).fault.message


@pytest.mark.parametrize("capture", [None, True, "9.9", float("nan"), float("inf"), 10.01, 9.75])
def test_missing_invalid_future_or_stale_observation_time(context: Context, capture: Any) -> None:
    """取得時刻の欠損・型不正・非有限値・未来・期限切れを送信前に拒否することを確認する。"""
    rejected(context, positions(), capture=capture)


@pytest.mark.parametrize("capture", [10.0, 9.99609375])
def test_zero_or_too_short_initial_interval_rejects_even_same_target(context: Context, capture: float) -> None:
    """目標を変えない場合でも、初回の観測から検査までの時間が0または短すぎれば拒否することを確認する。"""
    assert "interval" in rejected(context, positions(), capture=capture).fault.message


def test_reused_observation_is_refused(context: Context) -> None:
    """前フレームと同じ取得時刻の観測を再利用して次の指令を送れないことを確認する。"""
    context.loop.process_frame(positions(), positions(), observation_time=CAPTURE)
    context.clock.now = 10.03125
    assert "reused" in rejected(context, positions(), capture=CAPTURE).fault.message


def test_newer_observation_predating_previous_send_return_is_refused(context: Context) -> None:
    """前の観測より新しくても、前回送信完了より前の観測は次の指令に使えないことを確認する。"""
    context.loop.process_frame(positions(), positions(), observation_time=CAPTURE)
    context.clock.now = 10.03125
    assert "predates" in rejected(context, positions(), capture=9.984375).fault.message


@pytest.mark.parametrize("advance", [0.0, 1 / 256, 0.5, -0.0625])
def test_invalid_send_interval_never_resumes_after_pause(context: Context, advance: float) -> None:
    """連続送信間隔が0・短すぎ・長すぎ・逆行なら、同じ目標でも自動継続できないことを確認する。"""
    context.loop.process_frame(positions(), positions(), observation_time=CAPTURE)
    context.clock.now += advance
    rejected(context, positions(), capture=context.clock.now)


@pytest.mark.parametrize("clock_value", [float("nan"), float("inf"), True])
def test_invalid_monotonic_clock_rejects_before_send(context: Context, clock_value: Any) -> None:
    """制御側の時計がNaN・無限大・boolを返した場合、送信前に拒否することを確認する。"""
    context.clock.now = clock_value
    rejected(context, positions())


@pytest.mark.parametrize("observation", [
    {key: value for key, value in positions().items() if key != "wrist_flex.pos"},
    positions(**{"wrist_flex.pos": float("nan")}),
    positions(**{"shoulder_lift.pos": 999.0}),
])
def test_invalid_measured_positions_are_refused(context: Context, observation: dict[str, Any]) -> None:
    """比較基準の観測に欠損・NaN・範囲外があれば、指令が正常でも送信前に拒否することを確認する。"""
    rejected(context, positions(), observation=observation, stage="motion_observation_guard")


def test_camera_fields_do_not_become_motion_reference(context: Context) -> None:
    """画像を含む観測でも6軸だけを動作検査へ取り出し、画像を関節値と誤認しないことを確認する。"""
    context.loop.process_frame(positions(), {**positions(), "top": np.zeros((4, 4, 3), dtype=np.uint8)}, observation_time=CAPTURE)
    assert len(context.robot.calls) == 1


@pytest.mark.parametrize("boundary", ["initial_minimum", "initial_max_age", "subsequent_maximum", "send_duration"])
def test_timing_limit_boundaries_are_inclusive(context: Context, boundary: str) -> None:
    """初回最短間隔・観測最大鮮度・次回最長間隔・最大送信時間の境界ちょうどを受理することを確認する。"""
    capture = CAPTURE
    if boundary == "initial_minimum":
        capture = context.clock.now - policy().min_interval_s
    elif boundary == "initial_max_age":
        capture = context.clock.now - policy().max_observation_age_s
    elif boundary == "send_duration":
        context.robot.delay = policy().max_send_duration_s
    context.loop.process_frame(positions(), positions(), observation_time=capture)
    if boundary == "subsequent_maximum":
        context.clock.now += policy().max_interval_s
        context.loop.process_frame(positions(), positions(), observation_time=context.clock.now - 0.03125)
    assert context.loop.state is RecordState.RECORDING
    assert len(context.robot.calls) == (2 if boundary == "subsequent_maximum" else 1)


@pytest.mark.parametrize("delay", [0.125, -0.015625, float("nan")])
def test_delayed_or_invalid_send_completion_faults_after_one_attempt(context: Context, delay: float) -> None:
    """送信戻りの遅延・逆行・NaNは1回の送信後にFAULTとなり、保存や再送を防ぐことを確認する。"""
    context.robot.delay = delay
    with pytest.raises(RecordFault) as caught:
        context.loop.process_frame(positions(), positions(), observation_time=CAPTURE)
    assert caught.value.fault.stage == "send_timing_guard"
    assert caught.value.fault.send_status is SendStatus.RETURNED
    assert len(context.robot.calls) == 1 and context.saved == []
    with pytest.raises(RecordStateError):
        context.loop.process_frame(positions(), positions(), observation_time=CAPTURE)
    assert len(context.robot.calls) == 1


def test_returned_target_motion_violation_cannot_be_recorded(context: Context) -> None:
    """位置範囲内でも送信戻り値の変化量が大きすぎる場合、送信済みの状態を保持しつつフレーム保存を拒否することを確認する。"""
    context.robot.returned_gripper = 60.0  # 位置の範囲内だが、変化量の検査には通らない条件にする。
    with pytest.raises(RecordFault) as caught:
        context.loop.process_frame(positions(), positions(), observation_time=CAPTURE)
    assert caught.value.fault.stage == "sent_motion_guard"
    assert caught.value.fault.send_status is SendStatus.RETURNED
    assert len(context.robot.calls) == 1 and context.saved == []


def test_send_latency_does_not_relax_returned_target_rate_budget(context: Context) -> None:
    """送信戻り値の速度検査でも送信前に確定した時間枠を使い、送信遅延で許容量を広げないことを確認する。"""
    context.robot.delay = 0.0625
    context.robot.returned_gripper = 51.5
    with pytest.raises(RecordFault) as caught:
        context.loop.process_frame(positions(), positions(), observation_time=9.984375)
    assert caught.value.fault.stage == "sent_motion_guard"
    assert "target rate" in caught.value.fault.message
    assert caught.value.fault.send_status is SendStatus.RETURNED
    assert context.saved == [] and len(context.robot.calls) == 1


@pytest.mark.parametrize("error", [OSError("uncertain serial send"), KeyboardInterrupt()])
def test_motion_gate_does_not_retry_or_disconnect_on_send_failure(context: Context, error: BaseException) -> None:
    """通信例外・割込み時は送信結果不明を保持し、ループ自身の再送・切断を行わないことを確認する。"""
    context.robot.error = error
    expected = KeyboardInterrupt if isinstance(error, KeyboardInterrupt) else RecordFault
    with pytest.raises(expected):
        context.loop.process_frame(positions(), positions(), observation_time=CAPTURE)
    assert context.loop.fault is not None
    assert context.loop.fault.send_status is SendStatus.UNCERTAIN
    assert len(context.robot.calls) == 1 and context.saved == []
    with pytest.raises(RecordStateError):
        context.loop.process_frame(positions(), positions(), observation_time=CAPTURE)


def test_frame_write_failure_does_not_erase_physical_send_history(context: Context) -> None:
    """保存失敗でも検査済み送信目標と戻り時刻の履歴は残り、未保存フレーム数とは区別されることを確認する。"""
    def failed_write(action: dict[str, float]) -> None:
        raise OSError("partial write")

    context.loop.frame_writer = failed_write
    with pytest.raises(RecordFault):
        context.loop.process_frame(positions(), positions(), observation_time=CAPTURE)
    assert context.loop._previous_sent_action == positions()
    assert context.loop._previous_sent_at == 10.0
    assert context.loop.frames_written == 0 and len(context.robot.calls) == 1


def test_fresh_loop_sharing_policy_has_no_previous_episode_history(context: Context) -> None:
    """同じ制限設定を共有する新ループが、前エピソードの送信履歴を引き継がず新しい観測を初期基準にすることを確認する。"""
    context.loop.process_frame(positions(), positions(), observation_time=CAPTURE)
    context.loop.finish()
    other = SafeRecordLoop(context.loop.guard, lambda p: p[0], lambda p: p[0], context.robot,
                           lambda a: context.saved.append(dict(a)), motion_guard=context.loop.motion_guard, clock=context.clock)
    other.start()
    far = positions(**{"wrist_flex.pos": 20.0})
    other.process_frame(far, far, observation_time=CAPTURE)
    assert other.frames_written == 1


# デコレーターを重ねると全組合せを試す。4項目×5種類の不正値で20ケースになる。
@pytest.mark.parametrize("field_name", ["min_interval_s", "max_interval_s", "max_observation_age_s", "max_send_duration_s"])
@pytest.mark.parametrize("value", [0.0, -1.0, True, float("nan"), float("inf")])
def test_invalid_time_policy_is_rejected(field_name: str, value: Any) -> None:
    """4種類の時間制限に0・負数・bool・非有限値を指定できないことを確認する。"""
    with pytest.raises(ValueError):
        replace(policy(), **{field_name: value})


@pytest.mark.parametrize("field_name", ["max_step", "max_speed"])
@pytest.mark.parametrize("value", [0.0, -1.0, True, float("nan"), float("inf")])
def test_invalid_joint_threshold_is_rejected(field_name: str, value: Any) -> None:
    """軸ごとの変化量・速度上限に0・負数・bool・非有限値を指定できないことを確認する。"""
    configured = policy()
    with pytest.raises(ValueError):
        replace(configured, limits=(replace(configured.limits[0], **{field_name: value}), *configured.limits[1:]))


@pytest.mark.parametrize("change", ["missing", "duplicate", "units", "impossible_timing"])
def test_incomplete_or_incompatible_policy_is_rejected(change: str) -> None:
    """軸不足・重複・単位不一致・矛盾した時間条件を、動作制限の構築時に拒否することを確認する。"""
    configured = policy()
    with pytest.raises(ValueError):
        if change == "missing":
            replace(configured, limits=configured.limits[:-1])
        elif change == "duplicate":
            replace(configured, limits=(configured.limits[0], *configured.limits[:-1]))
        elif change == "units":
            replace(configured, limits=(replace(configured.limits[0], unit="radian"), *configured.limits[1:]))
        else:
            replace(configured, min_interval_s=0.5)


def test_action_guard_units_must_match_motion_policy(context: Context) -> None:
    """位置範囲検査器と動作制限の単位が違う場合、ループを構築できないことを確認する。"""
    gate = ActionGuard(tuple(replace(bound, unit="normalized_m100_100") for bound in context.loop.guard.bounds))
    with pytest.raises(ValueError, match="units"):
        SafeRecordLoop(gate, lambda p: p[0], lambda p: p[0], context.robot, lambda a: None, motion_guard=policy())


@pytest.mark.parametrize("finish", ["save", "reject_then_abort", "send_timing_fault"])
def test_motion_guard_real_image_dataset_session_lifecycle(tmp_path: Path, finish: str) -> None:
    """実画像付きデータセットで正常保存と次エピソードの送信前・送信後FAULTを試し、保存済み2フレームだけが残ることを確認する。

    画像、送信戻り値、メタデータを読み戻し、異常なエピソードの明示破棄後も前の保存結果を保持する。
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image
    import io

    path = Path(__file__).resolve().parents[1] / "configs/calibration_snapshots/my_follower_for_004.json"
    payload = path.read_bytes()
    gate = ActionGuard.from_calibration_bytes(payload)
    clock = FakeClock()
    robot = FakeRobot(clock, returned_gripper=49.0, calibration=json.loads(payload))
    features = {key: {"dtype": "float32", "shape": (6,), "names": list(JOINTS)}
                for key in ("action", "observation.state")}
    camera = "observation.images.top"
    features[camera] = {"dtype": "image", "shape": (8, 8, 3), "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(repo_id="local/offline_motion_guard", fps=30, features=features,
                              root=tmp_path / "motion_dataset", use_videos=False, image_writer_threads=1)
    check = RecordStartupCheck(calibration_path=path, expected_sha256=hashlib.sha256(payload).hexdigest(),
                               margin_deg=0.0, margin_gripper=0.0)
    writer = DatasetRecordWriter(ds, "Offline motion policy")

    def session() -> SafeRecordSession:
        loop = SafeRecordLoop(gate, lambda p: p[0], lambda p: p[0], robot, writer, motion_guard=policy(), clock=clock)
        return SafeRecordSession(loop, writer, startup_check=check)

    active = session()
    try:
        active.start()
        expected_images = []
        for index in range(2):
            capture = clock.now - 0.03125
            pixels = np.full((8, 8, 3), 40 + index * 60, dtype=np.uint8)
            expected_images.append(pixels.copy())
            active.record_frame(positions(), {**positions(), "top": pixels}, observation_time=capture)
            clock.now += 0.0625
        active.save_episode()
        assert active.startup_result is not None
        if finish != "save":
            active = session()
            active.start()
            active.record_frame(positions(), {**positions(), "top": np.zeros((8, 8, 3), dtype=np.uint8)},
                                observation_time=clock.now - 0.03125)
            clock.now += 0.0625
            if finish == "send_timing_fault":
                robot.delay = 0.125
            target = positions(**{"wrist_flex.pos": 3.0}) if finish == "reject_then_abort" else positions()
            with pytest.raises(RecordFault) as caught:
                active.record_frame(target,
                                    {**positions(), "top": np.zeros((8, 8, 3), dtype=np.uint8)},
                                    observation_time=clock.now - 0.03125)
            assert caught.value.fault.stage == ("motion_guard" if finish == "reject_then_abort" else "send_timing_guard")
            assert active.state is SessionState.FAULT
            assert active.fault is not None
            assert active.fault.send_status is (SendStatus.NOT_ATTEMPTED if finish == "reject_then_abort" else SendStatus.RETURNED)
            with pytest.raises(SessionStateError):
                active.save_episode()
            active.abort_episode()
        active.finalize()
        files = list((ds.root / "data").rglob("*.parquet"))
        assert len(files) == 1
        table = pq.read_table(files[0])
        assert table.num_rows == ds.meta.total_frames == 2
        assert ds.meta.total_episodes == 1
        np.testing.assert_allclose(table["action"].to_pylist(), [[0, 0, 0, 0, 0, 49]] * 2, rtol=1e-7, atol=1e-6)
        assert table["episode_index"].to_pylist() == [0, 0]
        for saved, pixels in zip(table[camera].to_pylist(), expected_images, strict=True):
            with Image.open(io.BytesIO(saved["bytes"])) as image:
                np.testing.assert_array_equal(np.asarray(image), pixels)
        assert not list(ds.root.rglob("*.png"))
        assert len(robot.calls) == {"save": 2, "reject_then_abort": 3, "send_timing_fault": 4}[finish]
    finally:
        ds.finalize()  # テストで使った資源を解放するだけで、セッションの異常復旧ではない。
