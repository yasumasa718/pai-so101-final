"""実機へ接続せず、F6限定の書込み順序・開始拒否・失敗時の解除を検証する。

GripperBusは目標とトルクをメモリで保持し、書込みが適用された後の応答喪失も再現する。
腕5軸へ書き込むと模擬バスのassertで失敗するため、操作対象の限定も検査できる。
"""

import io

import pytest

from src.control import gripper_smoke_test as gripper
from test_diagnose_observation import ReadOnlyBus, hardware_args, records


class GripperBus(ReadOnlyBus):
    """F6だけの読取り・位置指令・トルク操作を模擬し、全書込みを順序付きで記録する。"""
    def __init__(self):
        super().__init__()
        self.torque = {name: 0 for name in self.calibration}
        self.position = 2100
        self.goal = 3000  # 古い目標は実行前に置換されること。
        self.writes = []
        self.fail_write = None
        self.bad_goal_readback = False

    def read(self, register, motor, *, normalize, num_retry):
        assert not normalize and num_retry == 0
        if register == "Torque_Enable":
            return self.torque[motor]
        if register == "Present_Position":
            assert motor == "gripper"
            return self.position
        if register == "Goal_Position":
            return self.goal + int(self.bad_goal_readback)
        return super().read(register, motor, normalize=normalize, num_retry=num_retry)

    def sync_read(self, *args, **kwargs):
        values = super().sync_read(*args, **kwargs)
        cal = self.calibration["gripper"]
        values["gripper"] = (self.position-cal["range_min"]) * 100 / (cal["range_max"]-cal["range_min"])
        return values

    def write(self, register, motor, value, *, normalize, num_retry):
        """爪だけの書込みを受け付け、指定時は適用後に例外を出して「届いたが応答なし」を再現する。"""
        assert motor == "gripper"
        assert register in ("Goal_Position", "Torque_Enable")
        assert not normalize and num_retry == 0
        self.writes.append((register, value))
        if register == "Torque_Enable":
            self.torque[motor] = value
            if value:
                self.position = self.goal
        else:
            self.goal = value
            if self.torque[motor]:
                self.position = value
        # 適用されたが応答を失ったケース。
        if (register, value) == self.fail_write:
            raise ConnectionError("SECRET_PORT")


def run(tmp_path, monkeypatch, bus, execute=True):
    """バスと待機を差し替えて単発テストの子側処理を実行し、終了コードとログを返す。"""
    args = hardware_args(tmp_path)
    args.execute = execute
    monkeypatch.setattr(gripper, "make_hardware_bus", lambda *args: bus)
    monkeypatch.setattr(gripper.time, "sleep", lambda seconds: None)
    stream = io.StringIO()
    status = gripper.run_worker(args, stream)
    return status, records(stream.getvalue())


@pytest.mark.parametrize("execute", [False, True])
def test_preview_or_single_open_and_f6_only_cleanup(tmp_path, monkeypatch, execute):
    """読取り専用では書かず、実行時は現在値保持→ON→小さな開き→OFFの順でF6だけを操作することを確認する。"""
    bus = GripperBus()
    status, events = run(tmp_path, monkeypatch, bus, execute)
    assert status == 0
    if execute:
        assert bus.writes == [("Goal_Position", 2100), ("Torque_Enable", 1),
                              ("Goal_Position", 2128), ("Torque_Enable", 0)]
        assert next(e for e in events if e["event"] == "result")["reached"] is True
        assert events[-2]["event"] == "torque_off_verified"
    else:
        assert bus.writes == []
    assert all(value == 0 for value in bus.torque.values())
    assert bus.calls[-1] == "close_without_torque_change"
    assert events[-1] == {"event": "complete", "executed": execute}


def test_active_arm_refuses_without_any_writes(tmp_path, monkeypatch):
    """開始時に腕の1軸でもトルクONなら、書込みせずレジスタ検査で失敗することを確認する。"""
    bus = GripperBus()
    bus.torque["shoulder_lift"] = 1
    status, events = run(tmp_path, monkeypatch, bus)
    assert status == 1 and bus.writes == []
    assert next(e for e in events if e["event"] == "fault")["stage"] == "register_check"


@pytest.mark.parametrize("raw", [2034, 3475])
def test_invalid_initial_or_target_refuses_before_writes(tmp_path, monkeypatch, raw):
    """現在位置または開き目標が校正範囲外になる場合、書込み前に拒否することを確認する。"""
    bus = GripperBus()
    bus.position = raw
    status, events = run(tmp_path, monkeypatch, bus)
    assert status == 1 and bus.writes == []
    assert not any(e["event"] == "complete" for e in events)


@pytest.mark.parametrize("failed_write", [("Torque_Enable", 1), ("Goal_Position", 2128), ("Torque_Enable", 0)])
def test_uncertain_write_attempts_f6_off_once_and_never_retries(tmp_path, monkeypatch, failed_write):
    """ON・目標送信・OFFの応答を失っても再送せず、F6解除を試み、正常完了と報告しないことを確認する。"""
    bus = GripperBus()
    bus.fail_write = failed_write
    status, events = run(tmp_path, monkeypatch, bus)
    assert status == 1
    assert bus.writes.count(failed_write) == 1
    assert bus.writes[-1] == ("Torque_Enable", 0)
    assert bus.torque["gripper"] == 0
    assert bus.calls[-1] == "close_without_torque_change"
    assert "SECRET_PORT" not in str(events)
    assert not any(e["event"] == "complete" for e in events)


def test_goal_mismatch_never_enables_torque(tmp_path, monkeypatch):
    """現在位置を目標へ書いた後の読戻しが一致しなければ、トルクONへ進まないことを確認する。"""
    bus = GripperBus()
    bus.bad_goal_readback = True
    status, events = run(tmp_path, monkeypatch, bus)
    assert status == 1
    assert bus.writes == [("Goal_Position", 2100)]
    assert not any(e["event"] == "complete" for e in events)
