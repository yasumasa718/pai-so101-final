"""保存済みのSO-101経路を、実機へ接続せずに検査する。

LeRobotDataset v3.0のメタデータ・Parquet・Follower校正JSONを読み、
関節名の対応→指令範囲→欠損・時刻→指令の変化量→指令と観測の差を調べる。
guarded_teleopのJSONLを使う固定再生とは別の、データセット検証用の補助プログラム。

入力値は修正せず、欠損や非有限値も後段の検査まで保持する。各検査のPASSは
その項目に違反がないという意味で、総合判定はFAILまたは人の確認が必要なREVIEW。
実機の衝突・把持・再生可否を判定せず、入力の変更やモーター通信も行わない。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


logger = logging.getLogger(__name__)


# メタデータの関節名と校正のIDを照合するための、SO-101 Followerの対応表。
EXPECTED_MOTOR_IDS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}

# LeRobot 0.6.1ではSTS3215の4096段階を0〜4095で表し、腕の角度換算に4095を使う。
STS3215_MAX_POSITION = 4095

TRAJECTORY_COLUMNS = (
    "action",
    "observation.state",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)

# 6軸ベクトル全体、またはその一要素がNoneの場合も保持する。
# 読込み時に欠損・NaNを補間してしまうと、後段で元データの異常を検出できなくなる。
JointVector = tuple[float | None, ...] | None


CALIBRATION_FIELDS = (
    "id",
    "drive_mode",
    "homing_offset",
    "range_min",
    "range_max",
)


@dataclass(frozen=True)
class MotorCalibration:
    """1軸の校正。range_min/range_maxとhoming_offsetはモーターの整数tick値。"""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


@dataclass(frozen=True)
class ValidationConfig:
    """入力先・内側余白・任意の検査上限。Noneの上限は未指定であり、無制限の承認ではない。"""

    dataset_root: Path
    calibration_path: Path
    margin_deg: float = 0.0
    margin_gripper: float = 0.0
    max_step_deg: float | None = None
    max_step_gripper: float | None = None
    max_speed_deg_s: float | None = None
    max_speed_gripper_s: float | None = None
    max_gap_seconds: float | None = None
    tracking_max_lag_frames: int = 3
    report_json: Path | None = None


@dataclass(frozen=True)
class DatasetMetadata:
    """データセットの版・機種・fps・関節の並び。指令と観測の並びは別々に保持する。"""

    codebase_version: str
    robot_type: str
    fps: float
    action_names: tuple[str, ...]
    observation_state_names: tuple[str, ...]


@dataclass(frozen=True)
class TrajectoryData:
    """1つのParquetファイルから読み込んだ列を、元の順序と欠損を保って保持する。"""

    parquet_path: Path
    metadata: DatasetMetadata
    actions: tuple[JointVector, ...]
    observation_states: tuple[JointVector, ...]
    timestamps: tuple[float | None, ...]
    frame_indices: tuple[int | None, ...]
    episode_indices: tuple[int | None, ...]
    indices: tuple[int | None, ...]
    task_indices: tuple[int | None, ...]

    @property
    def num_frames(self) -> int:
        """読み込んだaction列の行数を返す。他の列との長さの一致は別途検査する。"""
        return len(self.actions)


@dataclass(frozen=True)
class JointMapping:
    """モーター名・IDと、指令配列・観測配列それぞれの添字との照合済み対応。

    expected_unitは想定する設定であり、データセットから推定した単位ではない。
    収録時にその単位が使われたかは別途確認する必要がある。
    """

    motor_name: str
    motor_id: int
    action_index: int
    state_index: int
    expected_unit: str


@dataclass(frozen=True)
class JointLimits:
    """校正由来の範囲と、両端からmarginを引いた検査用の有効範囲。

    F1〜F5は度、F6は0〜100のポイント。機械的な限界や衝突しない範囲の保証ではない。
    """

    motor_name: str
    motor_id: int
    unit: str
    minimum: float
    maximum: float
    margin: float
    effective_minimum: float
    effective_maximum: float


@dataclass(frozen=True)
class JointLimitCheck:
    """1軸の記録指令（action）の範囲検査結果。観測位置の検査結果ではない。

    違反は校正範囲外、内側余白だけの違反、不正値のいずれかに分類し、重複計上しない。
    最初の問題の行・フレーム・時刻と最大超過量を残す。observed_minimum/maximumは
    検査中に見つかった指令値の最小・最大を表し、observation.stateの値ではない。
    この項目のPASSだけで再生を許可することはできない。
    """

    motor_name: str
    motor_id: int
    unit: str
    frames_checked: int
    configured_range_violations: int
    margin_only_violations: int
    invalid_values: int
    observed_minimum: float | None
    observed_maximum: float | None
    first_problem_row: int | None
    first_problem_frame_index: int | None
    first_problem_timestamp: float | None
    first_problem_type: str | None
    max_configured_excess: float
    max_effective_excess: float

    @property
    def total_problems(self) -> int:
        """3種類の違反数を足し、この軸の問題サンプル総数を返す。"""
        return (
            self.configured_range_violations
            + self.margin_only_violations
            + self.invalid_values
        )

    @property
    def passed(self) -> bool:
        """この軸の全記録指令が範囲検査を通ったかを返す。"""
        return self.total_problems == 0


@dataclass(frozen=True)
class IntegrityIssue:
    """データ品質の問題例。rowはParquet内の0始まりの行番号で、frame_indexとは別。"""

    kind: str
    row: int
    detail: str


@dataclass(frozen=True)
class DataIntegrityResult:
    """欠損・番号・時刻などの品質検査結果。関節範囲や実機の安全性とは別に評価する。"""

    issue_counts: dict[str, int]
    examples: tuple[IntegrityIssue, ...]
    max_observed_dt_seconds: float | None
    max_gap_threshold_seconds: float | None

    @property
    def passed(self) -> bool:
        """データの欠損や時刻などの検査で、問題が見つからなかったかを返す。実機の安全判定ではない。"""
        return not self.issue_counts


@dataclass(frozen=True)
class JointJumpCheck:
    """隣接指令の変化量・変化速度と違反数。上限値は利用者が指定した検査条件。"""

    motor_name: str
    motor_id: int
    unit: str
    compared_pairs: int
    velocity_pairs: int
    max_step: float | None
    max_speed_per_second: float | None
    step_threshold: float | None
    speed_threshold: float | None
    step_violations: int
    speed_violations: int
    first_violation_row: int | None

    @property
    def status(self) -> str:
        """指令の変化量・変化速度の検査結果を返す。違反は FAIL、上限未指定は NOT_EVALUATED、それ以外は PASS。"""
        if self.step_violations or self.speed_violations:
            return "FAIL"
        if self.step_threshold is None and self.speed_threshold is None:
            return "NOT_EVALUATED"
        return "PASS"


@dataclass(frozen=True)
class TrackingLagMetric:
    """観測と数フレーム前の指令との差をまとめた、時間ずれの探索用指標。

    同期したサーボの追従誤差測定ではなく、再生可否の判定にも使わない。
    """

    motor_name: str
    motor_id: int
    lag_frames: int
    compared_samples: int
    mean_abs_difference: float | None
    max_abs_difference: float | None
    mean_timestamp_offset_seconds: float | None
    unit: str


@dataclass(frozen=True)
class ValidationInputs:
    """読込み済みデータと関節の対応表をまとめ、後段の検査へ渡す。

    このオブジェクトを作れた段階では、指令の範囲・時刻等はまだ未検査。
    """

    metadata: DatasetMetadata
    calibrations: dict[str, MotorCalibration]
    trajectory: TrajectoryData
    mappings: tuple[JointMapping, ...]


def load_calibration(
    path: Path,
) -> dict[str, MotorCalibration]:
    """SO-101の6軸分の校正JSONを検査し、モーター名→MotorCalibrationを返す。

    ID、必須整数フィールド、drive_mode、生値の範囲を確認する。読取りのみで、
    ファイルの更新や実機照合は行わない。ファイル読込みはOSError、JSON形式は
    JSONDecodeError、校正内容の不整合はValueErrorとして呼出し側へ伝える。
    """
    logger.info("Loading calibration: %s", path)

    with path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    if not isinstance(raw_data, dict):
        raise ValueError(
            "Calibration JSON must contain an object."
        )

    expected_names = set(EXPECTED_MOTOR_IDS)
    actual_names = set(raw_data)

    missing = expected_names - actual_names
    unexpected = actual_names - expected_names

    if missing or unexpected:
        raise ValueError(
            f"Invalid motor names: "
            f"missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    calibrations: dict[str, MotorCalibration] = {}

    for name, expected_id in EXPECTED_MOTOR_IDS.items():
        motor_data = raw_data[name]

        if not isinstance(motor_data, dict):
            raise ValueError(
                f"{name}: calibration must be an object."
            )

        values: dict[str, int] = {}

        for field in CALIBRATION_FIELDS:
            value = motor_data.get(field)

            if type(value) is not int:
                raise ValueError(
                    f"{name}.{field}: expected an integer, "
                    f"got {value!r}"
                )

            values[field] = value

        calibration = MotorCalibration(**values)

        if calibration.id != expected_id:
            raise ValueError(
                f"{name}: expected motor ID {expected_id}, "
                f"got {calibration.id}"
            )

        if calibration.drive_mode not in (0, 1):
            raise ValueError(
                f"{name}: invalid drive_mode "
                f"{calibration.drive_mode}"
            )

        if not (
            0 <= calibration.range_min
            < calibration.range_max
            <= 4095
        ):
            raise ValueError(
                f"{name}: invalid calibration range "
                f"{calibration.range_min} to "
                f"{calibration.range_max}"
            )

        calibrations[name] = calibration

        logger.info(
            "Loaded %s: id=%d, range=%d..%d",
            name,
            calibration.id,
            calibration.range_min,
            calibration.range_max,
        )

    return calibrations


def _read_feature_names(
    features: dict[str, object],
    feature_key: str,
) -> tuple[str, ...]:
    """float32・6要素の特徴量定義を検査し、重複のない関節名を記録順で返す。"""
    feature = features.get(feature_key)

    if not isinstance(feature, dict):
        raise ValueError(f"Missing or invalid feature: {feature_key}")

    if feature.get("dtype") != "float32":
        raise ValueError(f"{feature_key}: expected dtype float32")

    if feature.get("shape") != [6]:
        raise ValueError(f"{feature_key}: expected shape [6]")

    names = feature.get("names")
    if not isinstance(names, list) or len(names) != 6:
        raise ValueError(f"{feature_key}: expected six joint names")

    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{feature_key}: invalid joint name")

    if len(set(names)) != len(names):
        raise ValueError(f"{feature_key}: duplicate joint names")

    return tuple(names)


def load_dataset_metadata(dataset_root: Path) -> DatasetMetadata:
    """dataset_root/meta/info.jsonを読み、対応する版・機種・特徴量かを検査する。

    v3.0、so_follower、正のfps、6関節の指令・観測が必要。入力は変更しない。
    読込み不可はOSError、不正JSONはJSONDecodeError、未対応・不足項目はValueError。
    """
    metadata_path = dataset_root / "meta" / "info.json"
    logger.info("Loading dataset metadata: %s", metadata_path)

    with metadata_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    if not isinstance(raw_data, dict):
        raise ValueError("Dataset metadata must contain a JSON object.")

    codebase_version = raw_data.get("codebase_version")
    robot_type = raw_data.get("robot_type")
    fps = raw_data.get("fps")
    features = raw_data.get("features")

    if codebase_version != "v3.0":
        raise ValueError(f"Unsupported dataset version: {codebase_version}")

    if robot_type != "so_follower":
        raise ValueError(f"Unexpected robot type: {robot_type}")

    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid dataset FPS: {fps}")

    if not isinstance(features, dict):
        raise ValueError("Dataset features must be an object.")

    action_names = _read_feature_names(features, "action")
    observation_state_names = _read_feature_names(
        features, "observation.state"
    )

    metadata = DatasetMetadata(
        codebase_version=codebase_version,
        robot_type=robot_type,
        fps=float(fps),
        action_names=action_names,
        observation_state_names=observation_state_names,
    )

    logger.info(
        "Loaded dataset metadata: version=%s, robot=%s, fps=%s",
        metadata.codebase_version,
        metadata.robot_type,
        metadata.fps,
    )
    return metadata


def _read_joint_vectors(
    values: list[list[float | None] | None],
    column_name: str,
) -> tuple[JointVector, ...]:
    """関節列をtupleへ変換する。非欠損の行は6要素を要求し、NoneやNaNは保持する。"""
    vectors: list[JointVector] = []

    for row_index, value in enumerate(values):
        if value is None:
            vectors.append(None)
            continue

        if len(value) != 6:
            raise ValueError(
                f"{column_name}: row {row_index} has {len(value)} "
                "elements; expected 6"
            )

        vectors.append(tuple(value))

    return tuple(vectors)


def load_trajectory(
    parquet_path: Path,
    metadata: DatasetMetadata,
) -> TrajectoryData:
    """単一のParquetを読み、記録順を保ったTrajectoryDataを返す。

    parquet_pathはdata配下の1ファイル、metadataはload_dataset_metadataの戻り値。
    必須列とArrowの型・6軸の構造を確認する。NaN、欠損、時刻順、指令範囲は
    後段で検査するため、値の修復・行の並べ替えはしない。
    読込み不可はOSError、型や構造の不整合はValueError、復号失敗はArrowException。
    """
    if len(metadata.action_names) != 6 or len(metadata.observation_state_names) != 6:
        raise ValueError("Dataset metadata must describe six joint positions")

    logger.info("Loading trajectory: %s", parquet_path)

    # 既存ファイルを読むだけで、LeRobotの実機オブジェクトは作らない。
    parquet_file = pq.ParquetFile(parquet_path)
    schema = parquet_file.schema_arrow

    missing = set(TRAJECTORY_COLUMNS) - set(schema.names)
    if missing:
        raise ValueError(f"Missing Parquet columns: {sorted(missing)}")

    for column_name in ("action", "observation.state"):
        data_type = schema.field(column_name).type
        if not (
            pa.types.is_fixed_size_list(data_type)
            and data_type.list_size == 6
            and pa.types.is_float32(data_type.value_type)
        ):
            raise ValueError(
                f"{column_name}: expected fixed_size_list<float32>[6], "
                f"got {data_type}"
            )

    timestamp_type = schema.field("timestamp").type
    if not pa.types.is_floating(timestamp_type):
        raise ValueError(f"timestamp: expected float, got {timestamp_type}")

    for column_name in ("frame_index", "episode_index", "index", "task_index"):
        data_type = schema.field(column_name).type
        if not pa.types.is_integer(data_type):
            raise ValueError(f"{column_name}: expected integer, got {data_type}")

    table = parquet_file.read(columns=list(TRAJECTORY_COLUMNS))
    if table.num_rows == 0:
        raise ValueError(f"Parquet file contains no frames: {parquet_path}")

    trajectory = TrajectoryData(
        parquet_path=parquet_path,
        metadata=metadata,
        actions=_read_joint_vectors(table.column("action").to_pylist(), "action"),
        observation_states=_read_joint_vectors(
            table.column("observation.state").to_pylist(), "observation.state"
        ),
        timestamps=tuple(table.column("timestamp").to_pylist()),
        frame_indices=tuple(table.column("frame_index").to_pylist()),
        episode_indices=tuple(table.column("episode_index").to_pylist()),
        indices=tuple(table.column("index").to_pylist()),
        task_indices=tuple(table.column("task_index").to_pylist()),
    )

    logger.info("Loaded %d trajectory frames", trajectory.num_frames)
    return trajectory


def _motor_indices_from_feature_names(
    names: tuple[str, ...],
    feature_key: str,
) -> dict[str, int]:
    """<motor>.pos形式の名前から、モーター名→0始まりの列内添字を作る。

    F1〜F6の固定順を仮定せず、未知・重複・不足した関節はValueErrorにする。
    """
    indices: dict[str, int] = {}

    for index, feature_name in enumerate(names):
        if not isinstance(feature_name, str) or not feature_name.endswith(".pos"):
            raise ValueError(
                f"{feature_key}[{index}]: expected '<motor>.pos', "
                f"got {feature_name!r}"
            )

        motor_name = feature_name.removesuffix(".pos")
        if motor_name in indices:
            raise ValueError(
                f"{feature_key}: duplicate motor name: {motor_name}"
            )
        indices[motor_name] = index

    expected = set(EXPECTED_MOTOR_IDS)
    actual = set(indices)
    if expected != actual:
        raise ValueError(
            f"{feature_key}: incorrect joint set: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )

    return indices


def validate_joint_mapping(
    metadata: DatasetMetadata,
    calibrations: dict[str, MotorCalibration],
) -> tuple[JointMapping, ...]:
    """関節名・校正IDを照合し、指令と観測の添字を別々に求めた対応表を返す。

    出力順はmeta/info.jsonのaction_namesに従う。observation.stateの配列順が
    異なっていても、同じモーターの値を比較できるよう名前で結びつける。
    関節名の不足・余分・重複・形式不正やID不一致はValueError。
    ここでは経路の値や、収録時の単位の正しさは確認しない。
    """
    if metadata.robot_type != "so_follower":
        raise ValueError(
            f"Unexpected robot type for mapping: {metadata.robot_type}"
        )

    expected = set(EXPECTED_MOTOR_IDS)
    actual = set(calibrations)
    if actual != expected:
        raise ValueError(
            "Calibration motor names do not match SO-101 Follower: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )

    action_indices = _motor_indices_from_feature_names(
        metadata.action_names, "action"
    )
    state_indices = _motor_indices_from_feature_names(
        metadata.observation_state_names, "observation.state"
    )

    mappings: list[JointMapping] = []
    for motor_name, action_index in sorted(
        action_indices.items(), key=lambda pair: pair[1]
    ):
        calibration = calibrations[motor_name]
        expected_id = EXPECTED_MOTOR_IDS[motor_name]
        if not isinstance(calibration, MotorCalibration):
            raise ValueError(f"{motor_name}: invalid calibration object")
        if calibration.id != expected_id:
            raise ValueError(
                f"{motor_name}: expected ID {expected_id}, "
                f"got {calibration.id}"
            )

        mappings.append(
            JointMapping(
                motor_name=motor_name,
                motor_id=calibration.id,
                action_index=action_index,
                state_index=state_indices[motor_name],
                expected_unit=(
                    "normalized_0_100" if motor_name == "gripper" else "degree"
                ),
            )
        )

    logger.info("Verified joint mapping for %d motors", len(mappings))
    return tuple(mappings)


def calculate_joint_limits(
    calibrations: dict[str, MotorCalibration],
    mappings: tuple[JointMapping, ...],
    *,
    margin_deg: float = 0.0,
    margin_gripper: float = 0.0,
) -> tuple[JointLimits, ...]:
    """校正からLeRobot 0.6.1の単位で範囲を求め、内側余白付きのJointLimitsを返す。

    F1〜F5の換算は(raw_position - (range_min + range_max)/2) * 360 / 4095。
    F6は0〜100で、drive_mode=1でも向きが反転するだけで区間は変わらない。
    homing_offsetは機器の位置レジスタへ反映済みなので二重に加算しない。

    データセットは正規化モードを明示しないため、収録時にuse_degrees=Trueであり、
    校正が有効だったことは呼出し側で別途確認する。marginは各軸の単位で非負とする。
    この関数は指令値を検査せず、再生も許可しない。
    """
    for option_name, value in (
        ("margin_deg", margin_deg),
        ("margin_gripper", margin_gripper),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{option_name} must be a finite non-negative number: {value!r}")

    if len(mappings) != len(EXPECTED_MOTOR_IDS):
        raise ValueError(f"Expected {len(EXPECTED_MOTOR_IDS)} mapped joints")
    if {joint.motor_name for joint in mappings} != set(EXPECTED_MOTOR_IDS):
        raise ValueError("Mappings must contain each SO-101 Follower joint exactly once")
    if set(calibrations) != set(EXPECTED_MOTOR_IDS):
        raise ValueError("Calibration motor set does not match SO-101 Follower")

    results: list[JointLimits] = []
    for joint in mappings:
        motor_name = joint.motor_name
        cal = calibrations[motor_name]
        if cal.id != joint.motor_id or cal.id != EXPECTED_MOTOR_IDS[motor_name]:
            raise ValueError(f"{motor_name}: motor ID mismatch")
        if not (0 <= cal.range_min < cal.range_max <= STS3215_MAX_POSITION):
            raise ValueError(f"{motor_name}: invalid calibration range")

        if motor_name == "gripper":
            if joint.expected_unit != "normalized_0_100":
                raise ValueError("gripper: unexpected unit")
            # drive_modeによる向きの違いがあっても、検査区間は0〜100で共通。
            minimum, maximum = 0.0, 100.0
            margin = float(margin_gripper)
        else:
            if joint.expected_unit != "degree":
                raise ValueError(f"{motor_name}: expected degree unit")
            # 校正の中点を0度とし、機種の分解能から1を引いた値を換算の分母に使う。
            midpoint = (cal.range_min + cal.range_max) / 2.0
            minimum = (cal.range_min - midpoint) * 360.0 / STS3215_MAX_POSITION
            maximum = (cal.range_max - midpoint) * 360.0 / STS3215_MAX_POSITION
            margin = float(margin_deg)

        effective_minimum = minimum + margin
        effective_maximum = maximum - margin
        if effective_minimum >= effective_maximum:
            raise ValueError(
                f"{motor_name}: margin={margin} leaves no valid position interval "
                f"({minimum:.4f} .. {maximum:.4f} {joint.expected_unit})"
            )

        results.append(
            JointLimits(
                motor_name=motor_name,
                motor_id=joint.motor_id,
                unit=joint.expected_unit,
                minimum=minimum,
                maximum=maximum,
                margin=margin,
                effective_minimum=effective_minimum,
                effective_maximum=effective_maximum,
            )
        )

    return tuple(results)


def check_joint_limits(
    trajectory: TrajectoryData,
    mappings: tuple[JointMapping, ...],
    limits: tuple[JointLimits, ...],
) -> tuple[JointLimitCheck, ...]:
    """全記録指令を関節の校正範囲・内側余白と比較し、軸ごとの検査結果を返す。

    有限な値はまず元の範囲、次に余白付き範囲と比較する。不正値は補正せず違反に数える。
    observation.stateは別の時点の実測値なので、この指令範囲検査の対象にはしない。
    ロボットの衝突・トルク・実測速度は確認できず、結果を再生許可として扱わない。
    """
    if trajectory.num_frames == 0:
        raise ValueError("Trajectory contains no frames")
    if (
        len(trajectory.timestamps) != trajectory.num_frames
        or len(trajectory.frame_indices) != trajectory.num_frames
    ):
        raise ValueError("Trajectory timestamp/frame_index lengths do not match actions")

    expected = set(EXPECTED_MOTOR_IDS)
    if (
        len(mappings) != len(expected)
        or {joint.motor_name for joint in mappings} != expected
        or len(limits) != len(expected)
        or {item.motor_name for item in limits} != expected
    ):
        raise ValueError("Exactly one mapping and limit is required per SO-101 joint")

    limits_by_name = {item.motor_name: item for item in limits}
    if len({joint.action_index for joint in mappings}) != len(expected):
        raise ValueError("Joint action indices must be unique")

    results: list[JointLimitCheck] = []
    for joint in mappings:
        bounds = limits_by_name[joint.motor_name]
        if (
            bounds.motor_id != joint.motor_id
            or bounds.unit != joint.expected_unit
            or joint.motor_id != EXPECTED_MOTOR_IDS[joint.motor_name]
        ):
            raise ValueError(f"{joint.motor_name}: mapping/limit mismatch")
        if not (0 <= joint.action_index < len(expected)):
            raise ValueError(f"{joint.motor_name}: invalid action_index")
        if not (
            math.isfinite(bounds.minimum)
            and math.isfinite(bounds.maximum)
            and math.isfinite(bounds.effective_minimum)
            and math.isfinite(bounds.effective_maximum)
            and bounds.minimum <= bounds.effective_minimum
            < bounds.effective_maximum <= bounds.maximum
        ):
            raise ValueError(f"{joint.motor_name}: invalid joint bounds")

        configured_violations = 0
        margin_only_violations = 0
        invalid_values = 0
        observed_minimum: float | None = None
        observed_maximum: float | None = None
        first_problem_row: int | None = None
        first_problem_frame_index: int | None = None
        first_problem_timestamp: float | None = None
        first_problem_type: str | None = None
        max_configured_excess = 0.0
        max_effective_excess = 0.0

        for row, action_vector in enumerate(trajectory.actions):
            if action_vector is not None and len(action_vector) != len(expected):
                raise ValueError(f"action row {row}: expected six joint values")

            value = (
                None if action_vector is None
                else action_vector[joint.action_index]
            )
            problem_type: str | None = None

            if type(value) not in (int, float) or not math.isfinite(value):
                invalid_values += 1
                problem_type = "invalid_value"
            else:
                observed_minimum = (
                    value if observed_minimum is None
                    else min(observed_minimum, value)
                )
                observed_maximum = (
                    value if observed_maximum is None
                    else max(observed_maximum, value)
                )

                effective_excess = max(
                    bounds.effective_minimum - value,
                    value - bounds.effective_maximum,
                    0.0,
                )
                max_effective_excess = max(max_effective_excess, effective_excess)

                configured_excess = max(
                    bounds.minimum - value,
                    value - bounds.maximum,
                    0.0,
                )
                if configured_excess > 0:
                    configured_violations += 1
                    max_configured_excess = max(
                        max_configured_excess, configured_excess
                    )
                    problem_type = "configured_range"
                elif effective_excess > 0:
                    margin_only_violations += 1
                    problem_type = "margin_only"

            if problem_type is not None and first_problem_row is None:
                first_problem_row = row
                frame_index = trajectory.frame_indices[row]
                first_problem_frame_index = (
                    frame_index if type(frame_index) is int else None
                )
                timestamp = trajectory.timestamps[row]
                first_problem_timestamp = (
                    float(timestamp)
                    if type(timestamp) in (int, float) and math.isfinite(timestamp)
                    else None
                )
                first_problem_type = problem_type

        results.append(
            JointLimitCheck(
                motor_name=joint.motor_name,
                motor_id=joint.motor_id,
                unit=joint.expected_unit,
                frames_checked=trajectory.num_frames,
                configured_range_violations=configured_violations,
                margin_only_violations=margin_only_violations,
                invalid_values=invalid_values,
                observed_minimum=observed_minimum,
                observed_maximum=observed_maximum,
                first_problem_row=first_problem_row,
                first_problem_frame_index=first_problem_frame_index,
                first_problem_timestamp=first_problem_timestamp,
                first_problem_type=first_problem_type,
                max_configured_excess=max_configured_excess,
                max_effective_excess=max_effective_excess,
            )
        )

    return tuple(results)



def _valid_number(value: object) -> bool:
    """bool・None・NaN・無限大を除く、有限なint/floatかを判定する。"""
    return type(value) in (float, int) and math.isfinite(value)


def _validate_optional_positive(name: str, value: float | None) -> None:
    """Noneなら上限未指定とし、指定された場合は有限かつ正の数を要求する。"""
    if value is not None and (not _valid_number(value) or value <= 0):
        raise ValueError(f"{name} must be finite and > 0 when provided: {value!r}")


def check_data_integrity(
    trajectory: TrajectoryData,
    *,
    max_gap_seconds: float | None = None,
) -> DataIntegrityResult:
    """指令・観測の欠損、行番号、時刻の整合を検査し、件数と問題例を返す。

    各エピソード内では時刻の厳密な増加と、0から始まるframe_indexの連番を要求する。
    データセット全体のindexは連番、task_indexは非負整数でなければならない。
    時間間隔の上限検査はmax_gap_secondsを指定した場合だけ行う。
    問題例は先頭30件まで保持するが、違反の件数は全行で集計する。値の修復はしない。
    """
    _validate_optional_positive("max_gap_seconds", max_gap_seconds)
    n = trajectory.num_frames
    if n == 0:
        raise ValueError("Empty trajectory")
    columns = (
        trajectory.observation_states, trajectory.timestamps,
        trajectory.frame_indices, trajectory.episode_indices,
        trajectory.indices, trajectory.task_indices,
    )
    if any(len(column) != n for column in columns):
        raise ValueError("Trajectory columns have inconsistent lengths")

    counts: dict[str, int] = {}
    examples: list[IntegrityIssue] = []

    def issue(kind: str, row: int, detail: str) -> None:
        """種類別の件数を増やし、レポートが巨大にならない範囲で具体例を残す。"""
        counts[kind] = counts.get(kind, 0) + 1
        if len(examples) < 30:
            examples.append(IntegrityIssue(kind, row, detail))

    previous_episode: int | None = None
    previous_frame: int | None = None
    previous_index: int | None = None
    previous_time: float | None = None
    max_dt: float | None = None

    for row in range(n):
        for field, vector in (
            ("action", trajectory.actions[row]),
            ("observation.state", trajectory.observation_states[row]),
        ):
            if vector is None:
                issue(f"{field}.missing_vector", row, "Null joint vector")
            elif len(vector) != 6:
                issue(f"{field}.wrong_length", row, f"Got {len(vector)}, expected 6")
            else:
                for joint_index, value in enumerate(vector):
                    if value is None:
                        issue(f"{field}.missing_value", row, f"joint_index={joint_index}")
                    elif not _valid_number(value):
                        issue(
                            f"{field}.nonfinite_value", row,
                            f"joint_index={joint_index}, value={value!r}",
                        )

        episode = trajectory.episode_indices[row]
        frame = trajectory.frame_indices[row]
        global_index = trajectory.indices[row]
        task = trajectory.task_indices[row]
        timestamp = trajectory.timestamps[row]

        valid_episode = type(episode) is int and episode >= 0
        valid_frame = type(frame) is int and frame >= 0
        valid_global = type(global_index) is int and global_index >= 0
        if not valid_episode:
            issue("episode_index.invalid", row, repr(episode))
        if not valid_frame:
            issue("frame_index.invalid", row, repr(frame))
        if not valid_global:
            issue("index.invalid", row, repr(global_index))
        if type(task) is not int or task < 0:
            issue("task_index.invalid", row, repr(task))

        if valid_global and previous_index is not None and global_index != previous_index + 1:
            issue("index.discontinuous", row, f"{previous_index} -> {global_index}")
        previous_index = global_index if valid_global else None

        # エピソードが替わるとframe_indexとtimestampはリセットされる。
        # その境界を時間逆行や指令の連続動作として扱わない。
        same_episode = (
            valid_episode and previous_episode is not None
            and episode == previous_episode
        )
        if valid_episode and previous_episode is not None and episode < previous_episode:
            issue("episode_index.decreased", row, f"{previous_episode} -> {episode}")
        if valid_frame:
            if same_episode and previous_frame is not None and frame != previous_frame + 1:
                issue("frame_index.discontinuous", row, f"{previous_frame} -> {frame}")
            elif not same_episode and frame != 0:
                issue("frame_index.not_zero_at_episode_start", row, repr(frame))
        previous_frame = frame if valid_frame else None

        valid_timestamp = _valid_number(timestamp) and timestamp >= 0
        if not valid_timestamp:
            issue("timestamp.invalid", row, repr(timestamp))
        elif same_episode and previous_time is not None:
            dt = float(timestamp) - previous_time
            if dt <= 0:
                issue("timestamp.not_increasing", row, f"dt={dt:.9g}")
            else:
                max_dt = dt if max_dt is None else max(max_dt, dt)
                if max_gap_seconds is not None and dt > max_gap_seconds:
                    issue("timestamp.excessive_gap", row, f"dt={dt:.9g}s")
        previous_time = float(timestamp) if valid_timestamp else None
        previous_episode = episode if valid_episode else None

    return DataIntegrityResult(
        issue_counts=counts,
        examples=tuple(examples),
        max_observed_dt_seconds=max_dt,
        max_gap_threshold_seconds=max_gap_seconds,
    )


def check_position_jumps(
    trajectory: TrajectoryData,
    mappings: tuple[JointMapping, ...],
    *,
    max_step_deg: float | None = None,
    max_step_gripper: float | None = None,
    max_speed_deg_s: float | None = None,
    max_speed_gripper_s: float | None = None,
) -> tuple[JointJumpCheck, ...]:
    """同じエピソードの隣接指令について、変化量と時間差から求めた速度を集計する。

    変化量は|action[i] - action[i-1]|、速度はそれを記録時刻の差で割った値。
    実測したモーター速度ではない。腕と爪の上限は単位が異なるため別々に指定する。
    不正なサンプルは比較から除き、別途check_data_integrityで報告する。
    上限未指定でも最大値を集計するが、違反判定済みのPASSとは区別する。
    """
    for name, value in (
        ("max_step_deg", max_step_deg),
        ("max_step_gripper", max_step_gripper),
        ("max_speed_deg_s", max_speed_deg_s),
        ("max_speed_gripper_s", max_speed_gripper_s),
    ):
        _validate_optional_positive(name, value)

    results: list[JointJumpCheck] = []
    for joint in mappings:
        is_gripper = joint.motor_name == "gripper"
        step_limit = max_step_gripper if is_gripper else max_step_deg
        speed_limit = max_speed_gripper_s if is_gripper else max_speed_deg_s
        max_step: float | None = None
        max_speed: float | None = None
        compared = velocity_pairs = 0
        step_violations = speed_violations = 0
        first_violation_row: int | None = None
        for row in range(1, trajectory.num_frames):
            if trajectory.episode_indices[row] != trajectory.episode_indices[row - 1]:
                continue
            before, after = trajectory.actions[row - 1], trajectory.actions[row]
            if before is None or after is None:
                continue
            if not (len(before) == 6 and len(after) == 6):
                continue
            a, b = before[joint.action_index], after[joint.action_index]
            if not _valid_number(a) or not _valid_number(b):
                continue
            step = abs(float(b) - float(a))
            max_step = step if max_step is None else max(max_step, step)
            compared += 1
            violated = False
            if step_limit is not None and step > step_limit:
                step_violations += 1
                violated = True
            ta, tb = trajectory.timestamps[row - 1], trajectory.timestamps[row]
            if _valid_number(ta) and _valid_number(tb) and tb > ta:
                dt = float(tb) - float(ta)
                speed = step / dt
                max_speed = speed if max_speed is None else max(max_speed, speed)
                velocity_pairs += 1
                if speed_limit is not None and speed > speed_limit:
                    speed_violations += 1
                    violated = True
            if violated and first_violation_row is None:
                first_violation_row = row
        results.append(
            JointJumpCheck(
                motor_name=joint.motor_name,
                motor_id=joint.motor_id,
                unit=joint.expected_unit,
                compared_pairs=compared,
                velocity_pairs=velocity_pairs,
                max_step=max_step,
                max_speed_per_second=max_speed,
                step_threshold=step_limit,
                speed_threshold=speed_limit,
                step_violations=step_violations,
                speed_violations=speed_violations,
                first_violation_row=first_violation_row,
            )
        )
    return tuple(results)


def analyze_tracking_difference(
    trajectory: TrajectoryData,
    mappings: tuple[JointMapping, ...],
    *,
    max_lag_frames: int = 3,
) -> tuple[TrackingLagMetric, ...]:
    """0〜max_lag_framesだけ指令を遡り、観測との差の平均・最大を軸別に返す。

    遅れLならstate[i]とaction[i-L]を比較する。同じエピソードで有効な値だけを使い、
    比較可能な値がなければ統計値はNone。添字は関節名の対応表からそれぞれ求める。
    取得時刻のずれとサーボ応答の遅れが混ざるため、同期した追従誤差や再生可否ではなく、
    指令と観測の対応を調べる診断値として使う。
    """
    if type(max_lag_frames) is not int or not 0 <= max_lag_frames <= 30:
        raise ValueError("max_lag_frames must be an integer from 0 to 30")
    results: list[TrackingLagMetric] = []
    for joint in mappings:
        for lag in range(max_lag_frames + 1):
            difference_sum = max_difference = offset_sum = 0.0
            count = offset_count = 0
            for row in range(lag, trajectory.num_frames):
                src = row - lag
                if trajectory.episode_indices[row] != trajectory.episode_indices[src]:
                    continue
                action, state = trajectory.actions[src], trajectory.observation_states[row]
                if action is None or state is None or len(action) != 6 or len(state) != 6:
                    continue
                a, b = action[joint.action_index], state[joint.state_index]
                if not _valid_number(a) or not _valid_number(b):
                    continue
                difference = abs(float(a) - float(b))
                difference_sum += difference
                max_difference = max(max_difference, difference)
                count += 1
                ts_src, ts_state = trajectory.timestamps[src], trajectory.timestamps[row]
                if _valid_number(ts_src) and _valid_number(ts_state) and ts_state >= ts_src:
                    offset_sum += float(ts_state) - float(ts_src)
                    offset_count += 1
            results.append(
                TrackingLagMetric(
                    motor_name=joint.motor_name,
                    motor_id=joint.motor_id,
                    lag_frames=lag,
                    compared_samples=count,
                    mean_abs_difference=difference_sum / count if count else None,
                    max_abs_difference=max_difference if count else None,
                    mean_timestamp_offset_seconds=(offset_sum / offset_count if offset_count else None),
                    unit=joint.expected_unit,
                )
            )
    return tuple(results)


def _sha256_file(path: Path) -> str:
    """ファイルを1 MiBずつ読み、レポートで入力を識別するSHA-256を求める。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_validation_report(
    config: ValidationConfig,
    inputs: ValidationInputs,
    limits: tuple[JointLimits, ...],
    range_checks: tuple[JointLimitCheck, ...],
    integrity: DataIntegrityResult,
    jumps: tuple[JointJumpCheck, ...],
    tracking: tuple[TrackingLagMetric, ...],
) -> dict[str, object]:
    """各検査の結果・条件・入力ハッシュを、JSONに保存できる診断辞書へまとめる。

    範囲・データ品質・指定上限のいずれかに違反があればFAIL、それ以外もREVIEWとする。
    追従差は診断値として載せるだけで、判定には使わない。auto_replay_eligibleは常にFalse。
    ハッシュはこの関数を呼んだ時点でファイルを読み直して求めるため、検査中は入力を
    変更しないこと。読込み時と同一バイト列であることを保証する仕組みではない。
    """
    failed_range = any(not item.passed for item in range_checks)
    failed_jumps = any(item.status == "FAIL" for item in jumps)
    status = "FAIL" if failed_range or not integrity.passed or failed_jumps else "REVIEW"
    reasons: list[str] = []
    if failed_range:
        reasons.append("action_command_out_of_configured_or_effective_range")
    if not integrity.passed:
        reasons.append("data_integrity_issues")
    if failed_jumps:
        reasons.append("command_jump_threshold_exceeded")
    if any(item.status == "NOT_EVALUATED" for item in jumps):
        reasons.append("command_jump_thresholds_not_set")
    reasons.extend(("collision_and_physical_safety_not_evaluated", "calibration_at_recording_unverified"))

    return {
        "schema_version": "1.0",
        "verdict": status,
        "auto_replay_eligible": False,
        "reasons": reasons,
        "disclaimer": (
            "Offline diagnostic only. Calibration-derived ranges do not certify "
            "mechanical limits, collision clearance or safe robot playback."
        ),
        "dataset": {
            "root": str(config.dataset_root),
            "parquet": str(inputs.trajectory.parquet_path),
            "parquet_sha256": _sha256_file(inputs.trajectory.parquet_path),
            "metadata_sha256": _sha256_file(config.dataset_root / "meta" / "info.json"),
            "frames": inputs.trajectory.num_frames,
            "fps": inputs.metadata.fps,
        },
        "calibration": {
            "path": str(config.calibration_path),
            "sha256": _sha256_file(config.calibration_path),
            "recording_time_match_verified": False,
        },
        "configuration": {
            "margin_deg": config.margin_deg,
            "margin_gripper": config.margin_gripper,
            "max_step_deg": config.max_step_deg,
            "max_step_gripper": config.max_step_gripper,
            "max_speed_deg_s": config.max_speed_deg_s,
            "max_speed_gripper_s": config.max_speed_gripper_s,
            "max_gap_seconds": config.max_gap_seconds,
            "tracking_max_lag_frames": config.tracking_max_lag_frames,
        },
        "mapping": [asdict(x) for x in inputs.mappings],
        "limits": [asdict(x) for x in limits],
        "joint_range_checks": [asdict(x) | {"passed": x.passed} for x in range_checks],
        "data_integrity": {
            "passed": integrity.passed,
            "issue_counts": integrity.issue_counts,
            "examples": [asdict(x) for x in integrity.examples],
            "max_observed_dt_seconds": integrity.max_observed_dt_seconds,
            "max_gap_threshold_seconds": integrity.max_gap_threshold_seconds,
        },
        "command_jumps": [asdict(x) | {"status": x.status} for x in jumps],
        "tracking_diagnostics": [asdict(x) for x in tracking],
    }


def save_json_report(report: dict[str, object], output_path: Path, config: ValidationConfig) -> None:
    """明示した出力先にレポートを書く。データセット配下と校正ファイルへの出力は拒否する。

    パスを解決してから比較する。出力先の親フォルダーは作成し、既存レポートは上書きする。
    """
    destination = output_path.expanduser().resolve()
    dataset_dir = config.dataset_root.expanduser().resolve()
    calibration_path = config.calibration_path.expanduser().resolve()
    if destination.is_relative_to(dataset_dir) or destination == calibration_path:
        raise ValueError("Report path must not overwrite dataset or calibration input")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _find_single_parquet(dataset_root: Path) -> Path:
    """data配下のParquetを探し、ちょうど1ファイルの場合だけそのパスを返す。

    複数ファイルから先頭だけを選ぶと一部しか検査しない結果になるため、
    複数ファイルのデータセットは明示的に拒否する。
    """
    data_dir = dataset_root / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset data directory not found: {data_dir}")

    files = sorted(data_dir.rglob("*.parquet"))
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one Parquet file in {data_dir}, "
            f"found {len(files)}. Multi-file datasets are not supported yet."
        )
    return files[0]


def load_validation_inputs(
    dataset_root: Path,
    calibration_path: Path,
) -> ValidationInputs:
    """メタデータ・校正・関節対応・経路をまとめて読み、ValidationInputsを返す。

    dataset_rootはLeRobotDataset v3.0のルート、calibration_pathはFollowerの校正JSON。
    Parquetは1ファイルに限る。入力を読み取るだけで指令範囲検査や実機通信は行わない。
    アクセス不可はOSError、不正な形式・対応はValueError、Parquet復号失敗はArrowException。
    """
    metadata = load_dataset_metadata(dataset_root)
    calibrations = load_calibration(calibration_path)
    mappings = validate_joint_mapping(metadata, calibrations)

    parquet_path = _find_single_parquet(dataset_root)
    trajectory = load_trajectory(parquet_path, metadata)

    logger.info(
        "Offline inputs loaded: %d frames, %d mapped joints",
        trajectory.num_frames,
        len(mappings),
    )
    return ValidationInputs(
        metadata=metadata,
        calibrations=calibrations,
        trajectory=trajectory,
        mappings=mappings,
    )


def build_parser() -> argparse.ArgumentParser:
    """オフライン検査の入力と任意の上限を定義する。上限は実機安全性の認証値ではない。"""
    parser = argparse.ArgumentParser(description="Offline SO-101 trajectory validator")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--margin-deg", type=float, default=0.0)
    parser.add_argument("--margin-gripper", type=float, default=0.0)
    parser.add_argument("--max-step-deg", type=float, default=None)
    parser.add_argument("--max-step-gripper", type=float, default=None)
    parser.add_argument("--max-speed-deg-s", type=float, default=None)
    parser.add_argument("--max-speed-gripper-s", type=float, default=None)
    parser.add_argument("--max-gap-seconds", type=float, default=None)
    parser.add_argument("--tracking-max-lag-frames", type=int, default=3)
    parser.add_argument("--report-json", type=Path, default=None)
    return parser


def main() -> int:
    """一連の検査と結果表示を行い、指定時だけJSONレポートも保存するCLI。

    終了コードは入力・実行エラー1、要確認REVIEWが2、違反FAILが3。
    正常に検査できたことを実機再生の許可と誤認しないよう、0は返さない。
    """
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    config = ValidationConfig(
        dataset_root=args.dataset_root,
        calibration_path=args.calibration,
        margin_deg=args.margin_deg,
        margin_gripper=args.margin_gripper,
        max_step_deg=args.max_step_deg,
        max_step_gripper=args.max_step_gripper,
        max_speed_deg_s=args.max_speed_deg_s,
        max_speed_gripper_s=args.max_speed_gripper_s,
        max_gap_seconds=args.max_gap_seconds,
        tracking_max_lag_frames=args.tracking_max_lag_frames,
        report_json=args.report_json,
    )
    try:
        inputs = load_validation_inputs(config.dataset_root, config.calibration_path)
        limits = calculate_joint_limits(
            inputs.calibrations, inputs.mappings,
            margin_deg=config.margin_deg, margin_gripper=config.margin_gripper,
        )
        ranges = check_joint_limits(inputs.trajectory, inputs.mappings, limits)
        integrity = check_data_integrity(
            inputs.trajectory, max_gap_seconds=config.max_gap_seconds,
        )
        jumps = check_position_jumps(
            inputs.trajectory, inputs.mappings,
            max_step_deg=config.max_step_deg,
            max_step_gripper=config.max_step_gripper,
            max_speed_deg_s=config.max_speed_deg_s,
            max_speed_gripper_s=config.max_speed_gripper_s,
        )
        tracking = analyze_tracking_difference(
            inputs.trajectory, inputs.mappings,
            max_lag_frames=config.tracking_max_lag_frames,
        )
        report = build_validation_report(
            config, inputs, limits, ranges, integrity, jumps, tracking,
        )
        if config.report_json is not None:
            save_json_report(report, config.report_json, config)
    except (OSError, ValueError, TypeError, pa.ArrowException) as error:
        logger.error("Offline validation failed: %s", error)
        print("Auto-replay eligible: NO")
        return 1

    print("=== Offline SO-101 Trajectory Validation ===")
    print(f"Dataset: {config.dataset_root}")
    print(f"Calibration: {config.calibration_path}")
    print(f"Frames: {inputs.trajectory.num_frames}, FPS: {inputs.metadata.fps}")
    print("\n=== Joint Command Range Checks (action) ===")
    for check in ranges:
        status = "PASS" if check.passed else "FAIL"
        unit = "deg" if check.unit == "degree" else "normalized"
        print(
            f"F{check.motor_id} {check.motor_name:16s} {status} "
            f"configured={check.configured_range_violations} "
            f"margin_only={check.margin_only_violations} "
            f"invalid={check.invalid_values} "
            f"max_excess={check.max_configured_excess:.2f} {unit}"
        )
        if not check.passed:
            print(
                f"    First issue row={check.first_problem_row}, "
                f"frame_index={check.first_problem_frame_index}, "
                f"timestamp={check.first_problem_timestamp}, "
                f"type={check.first_problem_type}"
            )

    print("\n=== Data Integrity ===")
    print("Status:", "PASS" if integrity.passed else "FAIL")
    print("Max observed frame delta:", integrity.max_observed_dt_seconds, "s")
    for kind, count in sorted(integrity.issue_counts.items()):
        print(f"  {kind}: {count}")
    for issue in integrity.examples[:10]:
        print(f"  example row={issue.row}: {issue.kind} - {issue.detail}")

    print("\n=== Command Step / Speed Diagnostics ===")
    for jump in jumps:
        print(
            f"F{jump.motor_id} {jump.motor_name:16s} {jump.status:13s} "
            f"max_step={jump.max_step} max_speed/s={jump.max_speed_per_second} "
            f"step_violations={jump.step_violations} "
            f"speed_violations={jump.speed_violations}"
        )
    print("  Thresholds must be specified by the operator and do not establish safe speed.")

    print("\n=== Action/State Time-lag Diagnostics (not tracking errors) ===")
    for joint in inputs.mappings:
        joint_metrics = [m for m in tracking if m.motor_name == joint.motor_name]
        detail = " ".join(
            f"lag={m.lag_frames}:MAE={m.mean_abs_difference:.2f}"
            for m in joint_metrics[:2] if m.mean_abs_difference is not None
        )
        print(f"F{joint.motor_id} {joint.motor_name:16s} {detail}")

    print("\nVerdict:", report["verdict"])
    print("Auto-replay eligible: NO")
    if config.report_json is not None:
        print("JSON report:", config.report_json)
    print("Physical collision/clearance and recording-time calibration are NOT verified.")
    return 3 if report["verdict"] == "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
