#!/usr/bin/env python3
"""Convert UF850 raw episodes to no-op-filtered OpenVLA-OFT RLDS/TFDS.

The converter intentionally keeps the TFDS dataset name and feature schema used
by the local OpenVLA-OFT XArm training code.  It scans JSON first, filters idle
steps, and only then opens/crops the images that will be written.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


DATASET_NAME = "utokyo_xarm_pick_and_place_converted_externally_to_rlds"
DATASET_VERSION = "1.0.0"
DEFAULT_POS_THRESHOLD_CM = 0.02
DEFAULT_ROT_THRESHOLD_RAD = 0.002
DEFAULT_GRIPPER_THRESHOLD = 425.0
EXPECTED_IMAGE_SIZE = (1920, 1080)
OUTPUT_IMAGE_SIZE = (224, 224)
EXTERNAL_CROP = (540, 0, 1620, 1080)
WRIST_CROP = (760, 0, 1840, 1080)

RAW_HF_SOURCES = {
    "Press_buttons": {
        "repo_id": "AAyano/uf850-vr-teleop-press-buttons-raw",
        "revision": "495ebc9ff6bf301cccdcfe6765539bce4b9533e7",
    },
    "Place_corn_in_bowl": {
        "repo_id": "AAyano/uf850-vr-teleop-place-corn-in-bowl-raw",
        "revision": "6e6da9b719b0806c68e88b5f2e162ec18579e395",
    },
}

EPISODE_RE = re.compile(r"episode_(\d{3})$")
STEP_RE = re.compile(r"step_(\d{5})$")


class ConversionError(RuntimeError):
    """Raised when raw input or converted output is not safe to use."""


@dataclass(frozen=True)
class StepRecord:
    source_index: int
    step_dir: Path
    state: np.ndarray
    action: np.ndarray
    gripper_state: float


@dataclass(frozen=True)
class EpisodePlan:
    source_index: int
    episode_dir: Path
    instruction: str
    raw_steps: int
    kept_steps: tuple[StepRecord, ...]
    used_fallback: bool
    source_json_sha256: str


@dataclass(frozen=True)
class ScanResult:
    dataset_root: Path
    episodes: tuple[EpisodePlan, ...]
    raw_steps: int
    kept_steps: int
    dropped_steps: int
    fallback_episodes: int
    instruction: str
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]
    action_min: tuple[float, ...]
    action_max: tuple[float, ...]


def _runtime_dependencies() -> tuple[Any, Any]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    missing: list[str] = []
    try:
        import tensorflow_datasets as tfds
    except ModuleNotFoundError:
        tfds = None
        missing.append("tensorflow-datasets")
    try:
        from PIL import Image
    except ModuleNotFoundError:
        Image = None
        missing.append("pillow")
    if missing:
        raise SystemExit(
            "Missing conversion dependencies: "
            + " ".join(missing)
            + f"\nRun this script with {sys.executable!s} only after installing them."
        )
    return tfds, Image


def _float_vector(value: Any, *, size: int, field: str, path: Path) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ConversionError(f"{path}: {field} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ConversionError(f"{path}: {field} contains a non-finite value")
    return array


def wrap_degrees(values: Sequence[float]) -> np.ndarray:
    """Wrap degree deltas into [-180, 180), matching the OFT action convention."""
    array = np.asarray(values, dtype=np.float64)
    return (array + 180.0) % 360.0 - 180.0


def binary_gripper(position: float, *, threshold: float = DEFAULT_GRIPPER_THRESHOLD) -> float:
    """Map the xArm raw gripper position to OFT: +1 closed, -1 open."""
    position = float(position)
    if not math.isfinite(position):
        raise ConversionError("gripper position is not finite")
    return 1.0 if position <= threshold else -1.0


def transformed_state_and_action(
    data: dict[str, Any],
    *,
    path: Path,
    gripper_threshold: float = DEFAULT_GRIPPER_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, float]:
    observations = data.get("observations")
    action_data = data.get("action")
    meta = data.get("meta")
    if not isinstance(observations, dict) or not isinstance(action_data, dict):
        raise ConversionError(f"{path}: missing observations/action object")
    if not isinstance(meta, dict):
        meta = {}

    joints_deg = _float_vector(
        observations.get("joint_pos"), size=6, field="observations.joint_pos", path=path
    )
    raw_delta = _float_vector(
        action_data.get("delta_ee_pos"), size=6, field="action.delta_ee_pos", path=path
    )
    gripper_position = observations.get("gripper_pos")
    if not isinstance(gripper_position, (int, float)) or not math.isfinite(float(gripper_position)):
        raise ConversionError(f"{path}: observations.gripper_pos must be finite")

    wrapped_from_raw = wrap_degrees(raw_delta[3:])
    wrapped_meta = meta.get("delta_ee_rotation_wrapped")
    if wrapped_meta is not None:
        wrapped_stored = _float_vector(
            wrapped_meta,
            size=3,
            field="meta.delta_ee_rotation_wrapped",
            path=path,
        )
        if not np.allclose(wrapped_stored, wrapped_from_raw, rtol=0.0, atol=1e-4):
            raise ConversionError(
                f"{path}: stored wrapped Euler delta disagrees with action.delta_ee_pos"
            )
        wrapped_deg = wrapped_stored
    else:
        wrapped_deg = wrapped_from_raw

    state = np.deg2rad(joints_deg).astype(np.float32)
    action = np.concatenate(
        (
            raw_delta[:3] / 10.0,
            np.deg2rad(wrapped_deg),
            np.asarray([binary_gripper(float(gripper_position), threshold=gripper_threshold)]),
        )
    ).astype(np.float32)
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
        raise ConversionError(f"{path}: transformed state/action contains a non-finite value")
    return state, action, float(action[6])


def is_noop(
    action: np.ndarray,
    *,
    previous_gripper: float,
    pos_threshold_cm: float = DEFAULT_POS_THRESHOLD_CM,
    rot_threshold_rad: float = DEFAULT_ROT_THRESHOLD_RAD,
) -> bool:
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (7,):
        raise ValueError(f"action must have shape (7,), got {action.shape}")
    # Actions are stored as float32 in RLDS.  Compare to float32 thresholds so
    # an action exactly on the documented boundary is retained.
    pos_boundary = float(np.float32(pos_threshold_cm))
    rot_boundary = float(np.float32(rot_threshold_rad))
    moving = (
        float(np.linalg.norm(action[:3])) >= pos_boundary
        or float(np.linalg.norm(action[3:6])) >= rot_boundary
    )
    gripper_changed = abs(float(action[6]) - float(previous_gripper)) > 1e-6
    return not moving and not gripper_changed


def _indexed_directories(root: Path, pattern: re.Pattern[str], label: str) -> list[tuple[int, Path]]:
    indexed: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.fullmatch(path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    indexed.sort(key=lambda item: item[0])
    if not indexed:
        raise ConversionError(f"{root}: no {label} directories found")
    actual = [index for index, _ in indexed]
    expected = list(range(len(indexed)))
    if actual != expected:
        raise ConversionError(f"{root}: non-contiguous {label} indices: {actual[:8]}...")
    return indexed


def _load_instruction(episode_dir: Path) -> tuple[str, bytes]:
    path = episode_dir / "episode_meta.json"
    try:
        raw = path.read_bytes()
        metadata = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"{path}: cannot read episode metadata: {exc}") from exc
    instruction = metadata.get("task")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ConversionError(f"{path}: task/instruction is missing")
    return instruction.strip(), raw


def scan_episode(
    episode_index: int,
    episode_dir: Path,
    *,
    pos_threshold_cm: float,
    rot_threshold_rad: float,
    gripper_threshold: float,
    max_steps: int | None = None,
) -> EpisodePlan:
    instruction, episode_meta_bytes = _load_instruction(episode_dir)
    indexed_steps = _indexed_directories(episode_dir, STEP_RE, "step")
    if max_steps is not None:
        indexed_steps = indexed_steps[:max_steps]
    if len(indexed_steps) < 2:
        raise ConversionError(f"{episode_dir}: an RLDS episode needs at least two steps")

    hasher = hashlib.sha256()
    hasher.update(episode_meta_bytes)
    all_steps: list[StepRecord] = []
    kept: list[StepRecord] = []
    previous_gripper: float | None = None
    for step_index, step_dir in indexed_steps:
        data_path = step_dir / "data.json"
        for image_name in ("cam_0.jpg", "cam_1.jpg"):
            if not (step_dir / image_name).is_file():
                raise ConversionError(f"{step_dir}: missing {image_name}")
        try:
            raw_json = data_path.read_bytes()
            data = json.loads(raw_json)
        except (OSError, json.JSONDecodeError) as exc:
            raise ConversionError(f"{data_path}: cannot read JSON: {exc}") from exc
        hasher.update(raw_json)
        state, action, gripper_state = transformed_state_and_action(
            data, path=data_path, gripper_threshold=gripper_threshold
        )
        record = StepRecord(
            source_index=step_index,
            step_dir=step_dir,
            state=state,
            action=action,
            gripper_state=gripper_state,
        )
        all_steps.append(record)
        if previous_gripper is None:
            previous_gripper = gripper_state
        if not is_noop(
            action,
            previous_gripper=previous_gripper,
            pos_threshold_cm=pos_threshold_cm,
            rot_threshold_rad=rot_threshold_rad,
        ):
            kept.append(record)
        previous_gripper = gripper_state

    used_fallback = len(kept) < 2
    if used_fallback:
        kept = all_steps
    return EpisodePlan(
        source_index=episode_index,
        episode_dir=episode_dir,
        instruction=instruction,
        raw_steps=len(all_steps),
        kept_steps=tuple(kept),
        used_fallback=used_fallback,
        source_json_sha256=hasher.hexdigest(),
    )


def scan_dataset(
    dataset_root: Path,
    *,
    pos_threshold_cm: float = DEFAULT_POS_THRESHOLD_CM,
    rot_threshold_rad: float = DEFAULT_ROT_THRESHOLD_RAD,
    gripper_threshold: float = DEFAULT_GRIPPER_THRESHOLD,
    max_episodes: int | None = None,
    max_steps_per_episode: int | None = None,
) -> ScanResult:
    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise ConversionError(f"dataset root does not exist: {dataset_root}")
    if pos_threshold_cm < 0 or rot_threshold_rad < 0:
        raise ConversionError("no-op thresholds must be non-negative")
    if not math.isfinite(gripper_threshold):
        raise ConversionError("gripper threshold must be finite")

    indexed_episodes = _indexed_directories(dataset_root, EPISODE_RE, "episode")
    if max_episodes is not None:
        indexed_episodes = indexed_episodes[:max_episodes]
    plans: list[EpisodePlan] = []
    for episode_index, episode_dir in indexed_episodes:
        plan = scan_episode(
            episode_index,
            episode_dir,
            pos_threshold_cm=pos_threshold_cm,
            rot_threshold_rad=rot_threshold_rad,
            gripper_threshold=gripper_threshold,
            max_steps=max_steps_per_episode,
        )
        plans.append(plan)
        print(
            f"[scan] {episode_dir.name}: kept {len(plan.kept_steps)}/{plan.raw_steps}"
            + (" (fallback: kept unfiltered)" if plan.used_fallback else ""),
            flush=True,
        )

    instructions = {plan.instruction for plan in plans}
    if len(instructions) != 1:
        raise ConversionError(
            f"{dataset_root}: expected one shared instruction, found {sorted(instructions)!r}"
        )
    actions = np.stack([step.action for plan in plans for step in plan.kept_steps])
    raw_steps = sum(plan.raw_steps for plan in plans)
    kept_steps = int(actions.shape[0])
    return ScanResult(
        dataset_root=dataset_root,
        episodes=tuple(plans),
        raw_steps=raw_steps,
        kept_steps=kept_steps,
        dropped_steps=raw_steps - kept_steps,
        fallback_episodes=sum(plan.used_fallback for plan in plans),
        instruction=next(iter(instructions)),
        action_mean=tuple(float(value) for value in actions.mean(axis=0)),
        action_std=tuple(float(value) for value in actions.std(axis=0)),
        action_min=tuple(float(value) for value in actions.min(axis=0)),
        action_max=tuple(float(value) for value in actions.max(axis=0)),
    )


def crop_and_resize(path: Path, *, role: str, image_module: Any) -> np.ndarray:
    crop = EXTERNAL_CROP if role == "external" else WRIST_CROP if role == "wrist" else None
    if crop is None:
        raise ValueError(f"unknown camera role: {role}")
    try:
        with image_module.open(path) as image:
            image.load()
            if image.size != EXPECTED_IMAGE_SIZE:
                raise ConversionError(
                    f"{path}: expected image size {EXPECTED_IMAGE_SIZE}, got {image.size}"
                )
            image = image.convert("RGB").crop(crop)
            resampling = getattr(image_module, "Resampling", image_module).LANCZOS
            image = image.resize(OUTPUT_IMAGE_SIZE, resample=resampling)
            result = np.asarray(image, dtype=np.uint8)
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError(f"{path}: cannot decode/crop image: {exc}") from exc
    if result.shape != (224, 224, 3):
        raise ConversionError(f"{path}: transformed image has unexpected shape {result.shape}")
    return result


def episode_example(plan: EpisodePlan, *, image_module: Any, dataset_name: str) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    last_index = len(plan.kept_steps) - 1
    for output_index, record in enumerate(plan.kept_steps):
        is_last = output_index == last_index
        steps.append(
            {
                "observation": {
                    "image": crop_and_resize(
                        record.step_dir / "cam_1.jpg", role="external", image_module=image_module
                    ),
                    "hand_image": crop_and_resize(
                        record.step_dir / "cam_0.jpg", role="wrist", image_module=image_module
                    ),
                    "end_effector_pose": record.state,
                },
                "action": record.action,
                "discount": np.float32(1.0),
                "reward": np.float32(1.0 if is_last else 0.0),
                "is_first": np.bool_(output_index == 0),
                "is_last": np.bool_(is_last),
                "is_terminal": np.bool_(is_last),
                "language_instruction": plan.instruction,
            }
        )
    return {
        "steps": steps,
        "episode_metadata": {
            "file_path": f"{dataset_name}/{plan.episode_dir.name}",
        },
    }


def make_builder_class(*, tfds: Any, image_module: Any, scan: ScanResult) -> type:
    plans = scan.episodes
    source_name = scan.dataset_root.name

    class UtokyoXarmPickAndPlaceConvertedExternallyToRlds(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version(DATASET_VERSION)
        RELEASE_NOTES = {
            DATASET_VERSION: "UF850 raw trajectories with OFT-compatible transforms and no-op filtering."
        }
        pkg_dir_path = Path(__file__).resolve().parent

        def _info(self) -> Any:
            return self.dataset_info_from_configs(
                features=tfds.features.FeaturesDict(
                    {
                        "steps": tfds.features.Dataset(
                            {
                                "observation": tfds.features.FeaturesDict(
                                    {
                                        "image": tfds.features.Image(
                                            shape=(224, 224, 3),
                                            dtype=np.uint8,
                                            encoding_format="jpeg",
                                        ),
                                        "hand_image": tfds.features.Image(
                                            shape=(224, 224, 3),
                                            dtype=np.uint8,
                                            encoding_format="jpeg",
                                        ),
                                        "end_effector_pose": tfds.features.Tensor(
                                            shape=(6,), dtype=np.float32
                                        ),
                                    }
                                ),
                                "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                                "discount": tfds.features.Scalar(dtype=np.float32),
                                "reward": tfds.features.Scalar(dtype=np.float32),
                                "is_first": tfds.features.Scalar(dtype=np.bool_),
                                "is_last": tfds.features.Scalar(dtype=np.bool_),
                                "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                                "language_instruction": tfds.features.Text(),
                            }
                        ),
                        "episode_metadata": tfds.features.FeaturesDict(
                            {"file_path": tfds.features.Text()}
                        ),
                    }
                ),
                description=(
                    f"UF850 Quest VR {source_name} trajectories, cropped to 224x224 and "
                    "filtered to remove no-op steps for OpenVLA-OFT."
                ),
            )

        def _split_generators(self, dl_manager: Any) -> dict[str, Any]:
            del dl_manager
            return {"train": self._generate_examples()}

        def _generate_examples(self) -> Iterator[tuple[str, dict[str, Any]]]:
            for plan in plans:
                print(
                    f"[write] {plan.episode_dir.name}: {len(plan.kept_steps)} steps",
                    flush=True,
                )
                yield plan.episode_dir.name, episode_example(
                    plan,
                    image_module=image_module,
                    dataset_name=source_name,
                )

    return UtokyoXarmPickAndPlaceConvertedExternallyToRlds


def _manifest(
    scan: ScanResult,
    *,
    output_root: Path,
    pos_threshold_cm: float,
    rot_threshold_rad: float,
    gripper_threshold: float,
) -> dict[str, Any]:
    source_hf = RAW_HF_SOURCES.get(scan.dataset_root.name)
    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": scan.dataset_root.name,
            "hugging_face_repo": source_hf["repo_id"] if source_hf else None,
            "hugging_face_revision": source_hf["revision"] if source_hf else None,
        },
        "instruction": scan.instruction,
        "tfds_dataset_name": DATASET_NAME,
        "tfds_version": DATASET_VERSION,
        "tfds_data_root": ".",
        "split": "train",
        "episodes": len(scan.episodes),
        "raw_steps": scan.raw_steps,
        "retained_steps": scan.kept_steps,
        "dropped_noop_steps": scan.dropped_steps,
        "retained_fraction": scan.kept_steps / scan.raw_steps,
        "fallback_episodes": scan.fallback_episodes,
        "no_op_filter": {
            "translation_threshold_cm": pos_threshold_cm,
            "rotation_threshold_rad": rot_threshold_rad,
            "gripper_change_is_always_kept": True,
            "actions_recomputed_after_filtering": False,
        },
        "transforms": {
            "state": "joint_pos degrees to radians",
            "action_xyz": "delta TCP XYZ millimetres divided by 10 (centimetres)",
            "action_rotation": "delta Euler degrees wrapped to [-180,180), then radians",
            "action_gripper": {
                "closed": 1.0,
                "open": -1.0,
                "raw_position_threshold": gripper_threshold,
                "rule": "closed when gripper_pos <= threshold",
            },
            "external_image": {
                "source": "cam_1.jpg",
                "crop_xyxy": list(EXTERNAL_CROP),
                "resize": list(OUTPUT_IMAGE_SIZE),
                "tfds_key": "image",
            },
            "wrist_image": {
                "source": "cam_0.jpg",
                "crop_xyxy": list(WRIST_CROP),
                "resize": list(OUTPUT_IMAGE_SIZE),
                "tfds_key": "hand_image",
            },
        },
        "action_statistics": {
            "mean": list(scan.action_mean),
            "std": list(scan.action_std),
            "min": list(scan.action_min),
            "max": list(scan.action_max),
        },
        "conversion_runtime": {
            "python": sys.version.split()[0],
            "tensorflow": importlib.metadata.version("tensorflow"),
            "tensorflow_datasets": importlib.metadata.version("tensorflow-datasets"),
            "numpy": np.__version__,
            "pillow": importlib.metadata.version("pillow"),
            "converter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "episode_details": [
            {
                "episode": plan.episode_dir.name,
                "raw_steps": plan.raw_steps,
                "retained_steps": len(plan.kept_steps),
                "dropped_noop_steps": plan.raw_steps - len(plan.kept_steps),
                "fallback_to_unfiltered": plan.used_fallback,
                "source_json_sha256": plan.source_json_sha256,
            }
            for plan in scan.episodes
        ],
    }


def write_manifest(
    scan: ScanResult,
    *,
    output_root: Path,
    pos_threshold_cm: float,
    rot_threshold_rad: float,
    gripper_threshold: float,
) -> Path:
    path = output_root / "conversion_manifest.json"
    payload = _manifest(
        scan,
        output_root=output_root,
        pos_threshold_cm=pos_threshold_cm,
        rot_threshold_rad=rot_threshold_rad,
        gripper_threshold=gripper_threshold,
    )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def build_dataset(
    scan: ScanResult,
    *,
    output_root: Path,
    overwrite: bool,
    pos_threshold_cm: float,
    rot_threshold_rad: float,
    gripper_threshold: float,
) -> Path:
    tfds, image_module = _runtime_dependencies()
    output_root = output_root.expanduser().resolve()
    dataset_dir = output_root / DATASET_NAME
    if dataset_dir.exists():
        if not overwrite:
            raise ConversionError(
                f"output already exists: {dataset_dir} (pass --overwrite to replace it)"
            )
        shutil.rmtree(dataset_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    builder_cls = make_builder_class(tfds=tfds, image_module=image_module, scan=scan)
    builder = builder_cls(data_dir=str(output_root))
    if builder.name != DATASET_NAME:
        raise ConversionError(f"unexpected TFDS name {builder.name!r}; expected {DATASET_NAME!r}")
    builder.download_and_prepare()
    manifest_path = write_manifest(
        scan,
        output_root=output_root,
        pos_threshold_cm=pos_threshold_cm,
        rot_threshold_rad=rot_threshold_rad,
        gripper_threshold=gripper_threshold,
    )
    print(f"[done] TFDS: {builder.data_dir}", flush=True)
    print(f"[done] manifest: {manifest_path}", flush=True)
    return Path(builder.data_dir)


def validate_rlds(output_root: Path, *, expected: ScanResult | None = None) -> dict[str, Any]:
    tfds, _ = _runtime_dependencies()
    output_root = output_root.expanduser().resolve()
    version_dir = output_root / DATASET_NAME / DATASET_VERSION
    if not version_dir.is_dir():
        raise ConversionError(f"missing TFDS version directory: {version_dir}")
    builder = tfds.builder_from_directory(str(version_dir))
    if builder.name != DATASET_NAME:
        raise ConversionError(f"wrong TFDS dataset name: {builder.name}")
    dataset = builder.as_dataset(split="train", shuffle_files=False)

    expected_by_name = (
        {plan.episode_dir.name: plan for plan in expected.episodes} if expected is not None else {}
    )
    seen_episode_names: set[str] = set()
    episode_count = 0
    step_count = 0
    instructions: set[str] = set()
    for episode in dataset:
        source_path = episode["episode_metadata"]["file_path"].numpy().decode("utf-8")
        episode_name = Path(source_path).name
        if episode_name in seen_episode_names:
            raise ConversionError(f"duplicate converted episode key: {episode_name}")
        seen_episode_names.add(episode_name)
        expected_plan = expected_by_name.get(episode_name)
        if expected is not None and expected_plan is None:
            raise ConversionError(f"unexpected converted episode: {episode_name}")

        first_flags: list[bool] = []
        last_flags: list[bool] = []
        terminal_flags: list[bool] = []
        rewards: list[float] = []
        count = 0
        for step in tfds.as_numpy(episode["steps"]):
            action = step["action"]
            proprio = step["observation"]["end_effector_pose"]
            if action.shape != (7,):
                raise ConversionError(f"{episode_name}: bad action shape {action.shape}")
            if proprio.shape != (6,):
                raise ConversionError(f"{episode_name}: bad proprio shape {proprio.shape}")
            for key in ("image", "hand_image"):
                shape = step["observation"][key].shape
                if shape != (224, 224, 3):
                    raise ConversionError(f"{episode_name}: bad {key} shape {shape}")
            if not np.all(np.isfinite(action)) or not np.all(np.isfinite(proprio)):
                raise ConversionError(f"{episode_name}: state/action contains a non-finite value")
            if float(action[6]) not in (-1.0, 1.0):
                raise ConversionError(f"{episode_name}: gripper action is not -1/+1")
            instruction = step["language_instruction"].decode("utf-8")
            instructions.add(instruction)
            if expected_plan is not None:
                if count >= len(expected_plan.kept_steps):
                    raise ConversionError(f"{episode_name}: converted episode has too many steps")
                expected_step = expected_plan.kept_steps[count]
                if not np.array_equal(action, expected_step.action):
                    raise ConversionError(f"{episode_name}: action mismatch at retained step {count}")
                if not np.array_equal(proprio, expected_step.state):
                    raise ConversionError(f"{episode_name}: proprio mismatch at retained step {count}")
                if instruction != expected_plan.instruction:
                    raise ConversionError(f"{episode_name}: instruction mismatch at step {count}")
            first_flags.append(bool(step["is_first"]))
            last_flags.append(bool(step["is_last"]))
            terminal_flags.append(bool(step["is_terminal"]))
            rewards.append(float(step["reward"]))
            count += 1

        if count < 2:
            raise ConversionError(f"{episode_name}: converted episode has fewer than two steps")
        if expected_plan is not None and count != len(expected_plan.kept_steps):
            raise ConversionError(
                f"{episode_name}: got {count} steps, expected {len(expected_plan.kept_steps)}"
            )
        if first_flags != [True] + [False] * (count - 1):
            raise ConversionError(f"{episode_name}: is_first flags are invalid")
        if last_flags != [False] * (count - 1) + [True]:
            raise ConversionError(f"{episode_name}: is_last flags are invalid")
        if terminal_flags != [False] * (count - 1) + [True]:
            raise ConversionError(f"{episode_name}: is_terminal flags are invalid")
        if rewards != [0.0] * (count - 1) + [1.0]:
            raise ConversionError(f"{episode_name}: rewards are invalid")
        episode_count += 1
        step_count += count

    if expected is not None:
        if episode_count != len(expected.episodes) or step_count != expected.kept_steps:
            raise ConversionError(
                "RLDS count mismatch: "
                f"got {episode_count} episodes/{step_count} steps, expected "
                f"{len(expected.episodes)}/{expected.kept_steps}"
            )
        if instructions != {expected.instruction}:
            raise ConversionError(
                f"RLDS instructions {sorted(instructions)!r} do not match source instruction"
            )
    result = {
        "dataset_name": builder.name,
        "version": str(builder.version),
        "episodes": episode_count,
        "steps": step_count,
        "instructions": sorted(instructions),
        "status": "ok",
    }
    print("[validate] " + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--max-episodes", type=_positive_int)
    parser.add_argument("--max-steps-per-episode", type=_positive_int)
    parser.add_argument("--noop-pos-threshold-cm", type=float, default=DEFAULT_POS_THRESHOLD_CM)
    parser.add_argument("--noop-rot-threshold-rad", type=float, default=DEFAULT_ROT_THRESHOLD_RAD)
    parser.add_argument("--gripper-threshold", type=float, default=DEFAULT_GRIPPER_THRESHOLD)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        scan = scan_dataset(
            args.dataset_root,
            pos_threshold_cm=args.noop_pos_threshold_cm,
            rot_threshold_rad=args.noop_rot_threshold_rad,
            gripper_threshold=args.gripper_threshold,
            max_episodes=args.max_episodes,
            max_steps_per_episode=args.max_steps_per_episode,
        )
        print(
            f"[summary] {scan.dataset_root.name}: {len(scan.episodes)} episodes, "
            f"kept {scan.kept_steps}/{scan.raw_steps} steps "
            f"({scan.kept_steps / scan.raw_steps:.1%})",
            flush=True,
        )
        if args.scan_only:
            return 0
        build_dataset(
            scan,
            output_root=args.output_root,
            overwrite=args.overwrite,
            pos_threshold_cm=args.noop_pos_threshold_cm,
            rot_threshold_rad=args.noop_rot_threshold_rad,
            gripper_threshold=args.gripper_threshold,
        )
        if not args.skip_validation:
            validate_rlds(args.output_root, expected=scan)
    except (ConversionError, OSError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
