"""実機なしで読取り区間・FAULT・制御ループへの伝達を確認する。"""

import pytest

from src.control.action_guard import ActionGuard
from src.control.timed_observation import TimedJointReader
from src.control.motion_guard import MotionRejectedError
from src.control.safe_record_loop import RecordFault, RecordState
from test_motion_guard import FakeClock, FakeRobot, policy, positions


@pytest.fixture
def guard():
    from pathlib import Path
    return ActionGuard.from_calibration(Path("configs/calibration_snapshots/my_follower_for_004.json"))


class ReadBus:
    def __init__(self, values=None, error=None):
        self.values = {key.removesuffix(".pos"): value for key, value in positions().items()} if values is None else values
        self.error = error
        self.calls = []

    def sync_read(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if self.error:
            raise self.error
        return self.values


def reader_for(bus, guard, ticks):
    return TimedJointReader(bus, guard, max_read_duration_s=0.125, clock=iter(ticks).__next__)


def test_copies_positions_and_records_read_interval(guard):
    bus = ReadBus()
    sample = reader_for(bus, guard, [10.0, 10.0625]).read()
    assert sample.read_started_at == 10
    assert sample.read_completed_at == 10.0625
    assert sample.read_duration_s == 0.0625
    bus.values["shoulder_pan"] = 99
    copy = sample.as_observation()
    copy["shoulder_pan.pos"] = 88
    assert sample.as_observation() == positions()
    assert bus.calls == [("Present_Position", {"normalize": True, "num_retry": 0})]


@pytest.mark.parametrize("ticks", [[10, 9], [10, 10.25], [True], [float("nan")], [10, float("inf")]])
def test_invalid_read_times_latch_fault(guard, ticks):
    bus = ReadBus()
    reader = reader_for(bus, guard, ticks)
    with pytest.raises(ValueError):
        reader.read()
    count = len(bus.calls)
    with pytest.raises(RuntimeError, match="faulted"):
        reader.read()
    assert len(bus.calls) == count


@pytest.mark.parametrize("values", [{}, {"shoulder_pan": 0}, {**ReadBus().values, "extra": 0},
                                    {**ReadBus().values, "wrist_flex": 200},
                                    {**ReadBus().values, "gripper": float("nan")}])
def test_invalid_observations_rejected(guard, values):
    reader = reader_for(ReadBus(values), guard, [10, 10.01])
    with pytest.raises(ValueError):
        reader.read()
    assert reader.fault is not None


@pytest.mark.parametrize("error", [ConnectionError("secret port"), KeyboardInterrupt()])
def test_read_failure_is_not_retried(guard, error):
    bus = ReadBus(error=error)
    reader = reader_for(bus, guard, [10])
    with pytest.raises(type(error)):
        reader.read()
    assert reader.fault is error
    with pytest.raises(RuntimeError):
        reader.read()
    assert len(bus.calls) == 1


@pytest.mark.parametrize("next_start", [9, 10, 10.03125])
def test_overlapping_or_reused_reads_are_rejected_before_io(guard, next_start):
    bus = ReadBus()
    reader = reader_for(bus, guard, [10, 10.0625, next_start])
    reader.read()
    with pytest.raises(ValueError):
        reader.read()
    assert len(bus.calls) == 1


def test_motion_uses_start_for_age_and_end_for_initial_rate():
    check = policy().check(positions(), positions(), observed_at=9.9375,
                           observation_end_time=9.96875, now=10)
    assert check.observation_age_s == 0.0625
    assert check.interval_s == 0.03125
    assert check.observation_completed_at == 9.96875
    # 開始時刻を速度の基準にすると 2 度を許すが、終了からは 1 度しか許さない。
    with pytest.raises(MotionRejectedError, match="rate"):
        policy().check(positions(**{"wrist_flex.pos": 1.5}), positions(),
                       observed_at=9.9375, observation_end_time=9.96875, now=10)


def test_old_start_is_not_refreshed_by_recent_completion():
    with pytest.raises(MotionRejectedError, match="stale"):
        policy().check(positions(), positions(), observed_at=9,
                       observation_end_time=9.96875, now=10)


@pytest.mark.parametrize("end", [9.9, 10.1, True, float("nan"), float("inf")])
def test_motion_rejects_invalid_end(end):
    with pytest.raises(MotionRejectedError):
        policy().check(positions(), positions(), observed_at=9.9375, observation_end_time=end, now=10)


def test_second_frame_rate_still_uses_previous_send_completion():
    check = policy().check(positions(), positions(), observed_at=9.96875,
                           observation_end_time=9.984375, now=10,
                           previous_action=positions(), previous_sent_at=9.9375,
                           previous_observed_at=9.875)
    assert check.interval_s == 0.0625
    assert check.reference_source == "previous_sent"


def test_session_passes_interval_and_rejects_before_send(guard):
    from src.control.dataset_record_writer import DatasetRecordWriter
    from src.control.safe_record_loop import SafeRecordLoop
    from src.control.safe_record_session import SafeRecordSession, SessionState
    from test_record_startup_check import FakeDataset

    clock = FakeClock(10)
    robot = FakeRobot(clock)
    writer = DatasetRecordWriter(FakeDataset(), "Interval test")
    loop = SafeRecordLoop(guard, lambda p: p[0], lambda p: p[0], robot, writer,
                          motion_guard=policy(), clock=clock)
    session = SafeRecordSession(loop, writer)
    session.start()
    sample = reader_for(ReadBus(), guard, [9.9375, 9.96875]).read()
    with pytest.raises(RecordFault, match="rate"):
        session.record_frame(positions(**{"wrist_flex.pos": 1.5}), sample.as_observation(),
                             observation_time=sample.read_started_at,
                             observation_end_time=sample.read_completed_at)
    assert session.state is SessionState.FAULT
    assert loop.state is RecordState.FAULT
    assert robot.calls == []
    assert writer.frames_in_episode == 0


def test_session_records_valid_interval_observations(guard):
    from src.control.dataset_record_writer import DatasetRecordWriter
    from src.control.safe_record_loop import SafeRecordLoop
    from src.control.safe_record_session import SafeRecordSession
    from test_record_startup_check import FakeDataset

    clock = FakeClock(10)
    robot = FakeRobot(clock)
    dataset = FakeDataset()
    writer = DatasetRecordWriter(dataset, "Interval test")
    loop = SafeRecordLoop(guard, lambda p: p[0], lambda p: p[0], robot, writer,
                          motion_guard=policy(), clock=clock)
    session = SafeRecordSession(loop, writer)
    session.start()
    reader = reader_for(ReadBus(), guard, [9.9375, 9.96875, 10.03125, 10.0625])
    for now in (10, 10.09375):
        clock.now = now
        sample = reader.read()
        session.record_frame(positions(), sample.as_observation(),
                             observation_time=sample.read_started_at,
                             observation_end_time=sample.read_completed_at)
    session.save_episode()
    assert robot.calls == [positions(), positions()]
    assert len(dataset.episodes[0]) == 2
    session.finalize()
