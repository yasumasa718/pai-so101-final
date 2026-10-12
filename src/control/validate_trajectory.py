
"""Offline validation of recorded SO-101 trajectories.

This module does not connect to robot hardware.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


logger = logging.getLogger(__name__)


# SO-101 Follower motor IDs.
EXPECTED_MOTOR_IDS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}

TRAJECTORY_COLUMNS = (
    "action",
    "observation.state",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)

# A whole vector or an individual element may be missing (None).
# Missing and non-finite values are NOT repaired by the loader.
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
    """Calibration data for a single motor."""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for offline trajectory validation."""

    dataset_root: Path
    calibration_path: Path
    margin_deg: float = 0.0
    margin_gripper: float = 0.0
    max_step_deg: float | None = None
    max_step_gripper: float | None = None


@dataclass(frozen=True)
class DatasetMetadata:
    """Essential metadata for validating a recorded LeRobotDataset."""

    codebase_version: str
    robot_type: str
    fps: float
    action_names: tuple[str, ...]
    observation_state_names: tuple[str, ...]


@dataclass(frozen=True)
class TrajectoryData:
    """Recorded values from one Parquet file (not a safety verdict)."""

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
        """Number of loaded records."""
        return len(self.actions)


@dataclass(frozen=True)
class JointMapping:
    """Verified correspondence between feature indices and a follower motor.

    expected_unit is a configuration assumption, NOT a unit inferred from
    the LeRobotDataset metadata or a physical-safety certification.
    """

    motor_name: str
    motor_id: int
    action_index: int
    state_index: int
    expected_unit: str


@dataclass(frozen=True)
class ValidationInputs:
    """Data and verified joint mapping needed by later safety checks.

    Successfully creating this object is NOT a trajectory safety approval.
    """

    metadata: DatasetMetadata
    calibrations: dict[str, MotorCalibration]
    trajectory: TrajectoryData
    mappings: tuple[JointMapping, ...]


def load_calibration(
    path: Path,
) -> dict[str, MotorCalibration]:
    """Load and validate SO-101 Follower calibration JSON.

    This function only reads the file.
    It never modifies calibration data or connects to hardware.

    Raises:
        OSError: If the file cannot be opened.
        json.JSONDecodeError: If the JSON is malformed.
        ValueError: If calibration data is invalid.
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
    """Validate a joint-position feature and return its ordered names."""
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
    """Read and validate LeRobotDataset v3 metadata without modifying it.

    Args:
        dataset_root: Directory containing meta/info.json.

    Raises:
        OSError: If meta/info.json cannot be opened.
        json.JSONDecodeError: If the JSON is malformed.
        ValueError: If required metadata is missing or unsupported.
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
    """Keep joint data in recorded order, including missing values."""
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
    """Read one trajectory Parquet file without modifying any data.

    Checks required columns and Arrow column types, but DOES NOT approve
    the trajectory for replay. NaN, nulls, time order, and physical limits
    are inspected in later validation steps; all values are preserved.

    Args:
        parquet_path: Path to a single data/chunk-XXX/file-YYY.parquet.
        metadata: Dataset metadata obtained with load_dataset_metadata().

    Returns:
        TrajectoryData in recorded row and column order.

    Raises:
        OSError: File cannot be opened.
        ValueError: Required columns/types or frame structure are invalid.
        pyarrow.ArrowException: Parquet cannot be decoded.
    """
    if len(metadata.action_names) != 6 or len(metadata.observation_state_names) != 6:
        raise ValueError("Dataset metadata must describe six joint positions")

    logger.info("Loading trajectory: %s", parquet_path)

    # ParquetFile only accesses an existing file; it doesn't connect to LeRobot.
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
    """Map `<motor>.pos` metadata names to zero-based feature indices.

    Does not assume an order of motors or silently drop unknown features.
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
    """Verify motor names/IDs and derive both feature column indices.

    The output order follows `action_names` in meta/info.json, not a
    hard-coded F1-F6 array order. `observation.state` can have its own
    order, so its index is resolved separately by motor name.

    Does not inspect trajectory positions or establish unit correctness.
    Raises ValueError on missing/extra/duplicate/malformed joint names or
    calibration ID mismatches.
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


def _find_single_parquet(dataset_root: Path) -> Path:
    """Find the sole Parquet data file; reject ambiguous multi-file inputs.

    Multi-file dataset support is intentionally postponed. Selecting the
    first file silently could incorrectly approve only part of a dataset.
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
    """Load all offline inputs and verify motor mapping in one operation.

    Input files are read only. No motor communication, command sending,
    trajectory-limit check or replay approval takes place in this function.

    Args:
        dataset_root: Root of a LeRobotDataset v3.0.
        calibration_path: Saved Follower calibration JSON file.

    The dataset must contain exactly one Parquet data file.

    Raises:
        OSError: Input files or directories cannot be accessed.
        ValueError: Metadata, calibration, mapping or Parquet format invalid.
        pyarrow.ArrowException: Parquet decode error.
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
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Offline SO-101 trajectory validator"
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the LeRobotDataset directory",
    )

    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Path to the follower calibration JSON",
    )

    return parser


def main() -> int:
    """Load and display offline inputs; never approve a replay."""
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()

    config = ValidationConfig(
        dataset_root=args.dataset_root,
        calibration_path=args.calibration,
    )

    try:
        inputs = load_validation_inputs(
            dataset_root=config.dataset_root,
            calibration_path=config.calibration_path,
        )
    except (OSError, ValueError, pa.ArrowException) as error:
        logger.error("Input loading failed: %s", error)
        return 1

    print("=== Offline Trajectory Inputs ===")
    print(f"Dataset: {config.dataset_root}")
    print(f"Calibration: {config.calibration_path}")
    print(f"Parquet: {inputs.trajectory.parquet_path}")
    print(f"Frames: {inputs.trajectory.num_frames}")
    print(f"FPS: {inputs.metadata.fps}")
    print(f"Mapped joints: {len(inputs.mappings)}")
    for joint in inputs.mappings:
        print(
            f"F{joint.motor_id} {joint.motor_name:16s} "
            f"action_index={joint.action_index} "
            f"state_index={joint.state_index} "
            f"unit={joint.expected_unit}"
        )

    print("\nInput loading: PASS (structure only)")
    print("Trajectory safety: NOT VALIDATED")
    print("Auto-replay eligible: NO")
    logger.warning(
        "Safety checks are not implemented. No trajectory is approved."
    )
    # Nonzero exit status prevents an unfinished safety check being treated
    # as a successful safety validation by a shell or automation.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
