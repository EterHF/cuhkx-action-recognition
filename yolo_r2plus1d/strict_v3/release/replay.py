#!/usr/bin/env python3
"""Rebuild a strict release CSV from raw anonymous test data.

This is the P0.1 entry point.  It never reads historical test logits: YOLO
windows, the four-channel visual cache, the H36M-17 skeleton cache, public
DSTFormer frame inputs, and all three branch logits are generated in a fresh
working directory before the fixed release contract is applied.  The working
directory may be retained for audit, but it is not an input to a later run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from yolo_r2plus1d.strict_v3.data.skeleton_cache import load_person
from yolo_r2plus1d.strict_v3.models.fusion import TemporalFusionClassifier
from yolo_r2plus1d.strict_v3.models.public_dstformer import Net as PublicDSTNet
from yolo_r2plus1d.strict_v3.paths import (
    CHECKPOINT_DIR,
    DATA_DIR,
    REPO_ROOT,
    RESULT_DIR,
)
from yolo_r2plus1d.strict_v3.release.blend import apply_gate
from yolo_r2plus1d.strict_v3.release.packing import reconstruct_state
from yolo_r2plus1d.strict_v3.training.base import MEAN, STD
from yolo_r2plus1d.strict_v3.training.public_finetune import (
    VideoDataset,
    dequantize_state,
    make_model,
)
from yolo_r2plus1d.strict_v3.training.temporal import Residual as LogitResidual

DEFAULT_PACKAGE = CHECKPOINT_DIR / "model.pt"
DEFAULT_DETECTOR = CHECKPOINT_DIR / "yolo11n.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def softmax(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    value = value - value.max(axis=1, keepdims=True)
    value = np.exp(value)
    return value / value.sum(axis=1, keepdims=True)


def anchor_package_view(
    package: dict[str, Any], anchor_contract: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Materialize the exact low-bit anchor view stored as sparse corrections."""
    release = package.get("release_contract", {})
    ensemble = release.get("probability_ensemble")
    reconstruction = release.get("anchor_reconstruction")
    if not isinstance(ensemble, dict) or not isinstance(reconstruction, dict):
        raise RuntimeError("dual release package lacks anchor reconstruction")
    anchor = copy.copy(package)
    anchor["fusion_4bit"] = {
        "model_state_packed": reconstruct_state(
            package["fusion_4bit"]["model_state_packed"],
            reconstruction["fusion"],
        ),
        "bits": int(reconstruction.get("fusion_bits", 4)),
    }
    members = package.get("visual_members")
    if not isinstance(members, list) or len(members) != 1:
        raise RuntimeError("dual release package requires one candidate visual member")
    anchor.pop("visual_member0", None)
    anchor["visual_members"] = [
        {
            "model_state_packed": reconstruct_state(
                members[0]["model_state_packed"],
                reconstruction["visual"],
            ),
            "bits": int(reconstruction.get("visual_bits", 5)),
            "weight": 1.0,
            "input_adapter": bool(members[0].get("input_adapter", False)),
        }
    ]
    anchor["temporal_residual"] = reconstruction["temporal_residual"]
    anchor_release = copy.deepcopy(release)
    if anchor_contract is None:
        anchor_contract = ensemble["anchor"]
    anchor_release.update(copy.deepcopy(anchor_contract))
    anchor_release.pop("probability_ensemble", None)
    anchor_release.pop("anchor_reconstruction", None)
    anchor["release_contract"] = anchor_release
    return anchor


def run_command(command: list[str], cwd: Path) -> None:
    print("$ " + " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def crop_scale(points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float32).copy()
    valid = value[value[:, 2] != 0.0][:, :2]
    if len(valid) < 4:
        return np.zeros_like(value)
    low, high = valid.min(0), valid.max(0)
    scale = max(float((high - low).max()), 1e-6)
    center = (low + high - scale) / 2.0
    value[:, :2] = (value[:, :2] - center) / scale * 2.0 - 1.0
    return np.clip(value, -1.0, 1.0).astype(np.float32)


def raw_dst_frames(
    test_root: Path, test_ids: np.ndarray, frames: int = 16
) -> np.ndarray:
    result: list[np.ndarray] = []
    for sample_id in test_ids:
        paths = sorted(
            (test_root / str(sample_id) / "Skeleton" / "predictions").glob("*.json")
        )
        if not paths:
            result.append(np.zeros((frames, 17, 3), dtype=np.float32))
            continue
        indices = np.linspace(0, len(paths) - 1, frames).round().astype(int)
        sequence = []
        for index in indices:
            points = load_person(paths[int(index)])
            sequence.append(
                crop_scale(
                    points
                    if points is not None
                    else np.zeros((17, 3), dtype=np.float32)
                )
            )
        result.append(np.stack(sequence))
    return np.asarray(result, dtype=np.float32)


class RawFrames(Dataset):
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, index: int):
        return torch.from_numpy(self.value[index]), index


@torch.inference_mode()
def infer_visual(
    cache: Path,
    contract: dict[str, Any],
    package: dict[str, Any],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    mean = torch.tensor(contract["mean"], dtype=torch.float32).view(1, 4, 1, 1)
    std = torch.tensor(contract["std"], dtype=torch.float32).view(1, 4, 1, 1)
    dataset = VideoDataset(
        cache,
        np.arange(len(np.load(cache, mmap_mode="r"))),
        None,
        augment=False,
        horizontal_flip=False,
        source_mean=mean,
        source_std=std,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    output = np.zeros((len(dataset), 40), dtype=np.float32)
    members = package.get("visual_members")
    if not members:
        members = [package["visual_member0"]]
    total_weight = sum(float(member.get("weight", 1.0)) for member in members)
    if total_weight <= 0:
        raise RuntimeError("visual member weights must sum to a positive value")
    for member in members:
        # A strict package may blend the ordinary RGB-stem member with a
        # separately trained four-channel input-adapter member.  The adapter
        # flag is therefore carried per member; the package-level field is a
        # backwards-compatible default for legacy single-member packages.
        model = make_model(
            input_adapter=bool(
                member.get("input_adapter", package.get("visual_input_adapter", False))
            )
        )
        model.load_state_dict(
            dequantize_state(member["model_state_packed"]), strict=True
        )
        model.to(device).eval()
        member_output = np.zeros_like(output)
        for frames, indices in loader:
            # The released visual logits were generated with the same fp16
            # inference path used by ``infer_public_checkpoint_cache``.
            # Keeping the visual branch in fp16 avoids an otherwise silent
            # bf16 rounding change at close class margins; Fusion and the
            # skeleton-logit branch retain their historical bf16 path below.
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(frames.to(device, non_blocking=True))
            member_output[indices.numpy()] = logits.float().cpu().numpy()
        output += float(member.get("weight", 1.0)) * member_output / total_weight
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    release = package.get("release_contract", {})
    scale = float(release.get("visual_logit_scale", 1.0)) * float(
        release.get("visual_package_output_scale", 1.0)
    )
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("invalid visual_logit_scale in package contract")
    return output * scale


class FusionDataset(Dataset):
    def __init__(
        self, visual: Path, skeleton: Path, mask: Path, contract: dict[str, Any]
    ) -> None:
        self.visual = np.load(visual, mmap_mode="r")
        self.skeleton = np.load(skeleton, mmap_mode="r")
        self.mask = np.load(mask, mmap_mode="r")
        self.mean = torch.tensor(contract["mean"], dtype=torch.float32).view(1, 4, 1, 1)
        self.std = torch.tensor(contract["std"], dtype=torch.float32).view(1, 4, 1, 1)
        if len(self.visual) != len(self.skeleton) or len(self.visual) != len(self.mask):
            raise ValueError("raw visual/skeleton/mask lengths differ")

    def __len__(self) -> int:
        return len(self.visual)

    def __getitem__(self, index: int):
        frames = (
            torch.from_numpy(np.array(self.visual[index], copy=True))
            .float()
            .div_(255.0)
        )
        frames = ((frames - self.mean) / self.std) * STD + MEAN
        skeleton = torch.from_numpy(np.array(self.skeleton[index], copy=True)).float()
        return frames, skeleton, bool(self.mask[index]), index


@torch.inference_mode()
def infer_fusion(
    cache: Path,
    skeleton: Path,
    mask: Path,
    contract: dict[str, Any],
    package: dict[str, Any],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    dataset = FusionDataset(cache, skeleton, mask, contract)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    model = TemporalFusionClassifier(pretrained_visual=False)
    state = dequantize_state(package["fusion_4bit"]["model_state_packed"])
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    output = np.zeros((len(dataset), 40), dtype=np.float32)
    for frames, joints, present, indices in loader:
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(
                frames.to(device, non_blocking=True),
                joints.to(device, non_blocking=True),
                present.to(device, non_blocking=True),
            )
        output[indices.numpy()] = logits.float().cpu().numpy()
    return output


@torch.inference_mode()
def infer_temporal(
    raw_frames: np.ndarray,
    package: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    residual = package.get("temporal_residual")
    frame_batch_size = (
        int(residual.get("frame_inference_batch_size", batch_size))
        if isinstance(residual, dict)
        else batch_size
    )
    if frame_batch_size <= 0:
        raise RuntimeError("invalid temporal frame inference batch size")
    dstformer = PublicDSTNet()
    dstformer.load_state_dict(package["dstformer"]["state_dict"], strict=True)
    dstformer.to(device).eval()
    frame_logits = np.zeros(
        (len(raw_frames), raw_frames.shape[1], 40), dtype=np.float32
    )
    loader = DataLoader(
        RawFrames(raw_frames),
        batch_size=frame_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    for value, indices in loader:
        batch, steps = value.shape[:2]
        flattened = value.reshape(batch * steps, 1, 17, 3).to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = dstformer(flattened)
        frame_logits[indices.numpy()] = (
            logits.reshape(batch, steps, 40).float().cpu().numpy()
        )
    # A parameter-free temporal pool is a valid strict candidate: it operates
    # on the same raw DSTFormer frame logits and does not require a target/test
    # statistic.  Packages that omit this field retain the historical TCN.
    temporal_contract = package.get("release_contract", {}).get(
        "temporal_pool", "residual_tcn"
    )
    if temporal_contract in {"top2_frame_mean", "top4_frame_mean"}:
        count = 2 if temporal_contract == "top2_frame_mean" else 4
        return (
            np.sort(frame_logits, axis=1)[:, -count:, :].mean(axis=1).astype(np.float32)
        )
    if temporal_contract == "motion_energy_softmax":
        # A fixed TDN-style, parameter-free temporal pool.  The score is the
        # mean absolute first difference across the 40 classes, smoothed with
        # a three-frame edge-replicated box.  It is intentionally computed
        # from the clip itself; no user/session/missingness/test statistic is
        # involved.  Keeping this transform here (rather than materializing a
        # test cache) makes the package contract and raw replay identical.
        delta = np.diff(frame_logits, axis=1, prepend=frame_logits[:, :1])
        energy = np.mean(np.abs(delta), axis=2)
        padded = np.pad(energy, ((0, 0), (1, 1)), mode="edge")
        energy = (padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]) / 3.0
        scaled = (energy - energy.max(axis=1, keepdims=True)) / 0.5
        weights = np.exp(scaled)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        return np.sum(weights[:, :, None] * frame_logits, axis=1).astype(np.float32)
    if temporal_contract != "residual_tcn":
        raise RuntimeError(f"unsupported temporal_pool contract: {temporal_contract!r}")
    # The strict-v3 package records the original three-layer TCN residual
    # (40->96->96->40), not the later v2 80-channel experimental residual.
    if not isinstance(residual, dict):
        raise RuntimeError("package lacks temporal_residual")
    members = residual.get("members")
    if members is None:
        if residual.get("kind") != "tcn" or not isinstance(
            residual.get("model_state"), dict
        ):
            raise RuntimeError("invalid temporal residual checkpoint")
        members = [
            {"kind": "tcn", "model_state": residual["model_state"], "weight": 1.0}
        ]
    elif not (
        residual.get("schema_version") == "cuhkx-temporal-ensemble/v1"
        and residual.get("aggregation") == "equal arithmetic mean of logits"
        and residual.get("checkpoint_selection") == "fixed_final_epoch"
        and residual.get("labels_used_for_selection") is False
        and residual.get("test_statistics_used") is False
        and isinstance(members, list)
        and len(members) >= 2
    ):
        raise RuntimeError("invalid temporal residual ensemble")
    expected_weight = 1.0 / len(members)
    residual_batch_size = int(residual.get("residual_inference_batch_size", batch_size))
    if residual_batch_size <= 0:
        raise RuntimeError("invalid temporal residual inference batch size")
    loader = DataLoader(
        RawFrames(frame_logits),
        batch_size=residual_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    outputs = []
    for member in members:
        if not (
            isinstance(member, dict)
            and member.get("kind") == "tcn"
            and isinstance(member.get("model_state"), dict)
            and np.isclose(float(member.get("weight", float("nan"))), expected_weight)
        ):
            raise RuntimeError("invalid temporal residual ensemble member")
        temporal = LogitResidual(kind="tcn").to(device)
        temporal.load_state_dict(member["model_state"], strict=True)
        temporal.eval()
        output = np.zeros((len(raw_frames), 40), dtype=np.float32)
        for value, indices in loader:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = temporal(value.to(device, non_blocking=True))
            output[indices.numpy()] = logits.float().cpu().numpy()
        outputs.append(output)
        del temporal
    # Match average_logits.py exactly: promote each stored float32 member to
    # float64, take the equal arithmetic mean, and persist float32 logits.
    return np.mean(np.stack(outputs).astype(np.float64), axis=0).astype(np.float32)


def validate_contract(
    package: dict[str, Any], contract: dict[str, Any], detector: Path
) -> None:
    release = package.get("release_contract")
    if not isinstance(release, dict):
        raise RuntimeError("package has no release_contract")
    for key, expected in (
        ("per_sample_zscore", False),
        ("thermal_weight", 0.0),
        ("skeleton_mask_used_in_blend", False),
        ("test_labels_used", False),
        ("test_statistics_used", False),
        ("timestamp_metadata_used", False),
    ):
        if release.get(key) != expected:
            raise RuntimeError(f"unsafe release contract {key}={release.get(key)!r}")
    scale = float(release.get("visual_logit_scale", 1.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("unsafe release contract visual_logit_scale")
    package_scale = float(release.get("visual_package_output_scale", 1.0))
    if not np.isfinite(package_scale) or package_scale <= 0.0:
        raise RuntimeError("unsafe release contract visual_package_output_scale")
    # New release packages embed the contract.  A small number of legacy
    # candidates predate that field; those may still be replayed when the
    # caller explicitly supplies a train-only contract (the CLI fallback
    # above).  The fallback is deliberately compared against no package
    # value, and the resulting audit summary records that the package was not
    # self-contained.  This keeps the raw replay useful for legacy candidates
    # without weakening the safety checks for modern packages.
    embedded = release.get("preprocessing_contract")
    if isinstance(embedded, dict):
        for key in ("mean", "std", "target_mean", "target_std", "channels"):
            if embedded.get(key) != contract.get(key):
                raise RuntimeError(
                    f"CLI preprocessing contract differs from package for {key}"
                )
    raw = release.get("raw_pipeline")
    if isinstance(raw, dict):
        if raw.get("frames") != 16 or raw.get("image_size") != 128:
            raise RuntimeError("package raw pipeline contract is incomplete")
    elif not isinstance(contract, dict):
        raise RuntimeError("package raw pipeline contract is incomplete")
    # Legacy candidates without a raw_pipeline are replayed with the
    # explicit command-line fallback and the historical 16x128 defaults used
    # by prepare_cache.  The output summary marks this non-self-contained
    # contract so it cannot be mistaken for a modern packaged release.
    if not detector.is_file():
        raise FileNotFoundError(f"YOLO detector not found: {detector}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument(
        "--test-root",
        type=Path,
        default=DATA_DIR / "processed/test/small_model_track_test",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=DATA_DIR / "Small-Model-Track/Testing/test_file/test.csv",
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=DATA_DIR / "Small-Model-Track/Testing/test_file/sample_submission.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=RESULT_DIR / "reproduced_submission.csv"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="retain generated raw caches; otherwise a temporary directory is removed",
    )
    parser.add_argument(
        "--preprocessing-contract",
        type=Path,
        help="legacy fallback only; new packages must embed this contract",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not args.package.is_file():
        raise FileNotFoundError(args.package)
    package = torch.load(args.package, map_location="cpu", weights_only=True)
    release = package.get("release_contract", {})
    contract = release.get("preprocessing_contract")
    embedded_contract = isinstance(contract, dict)
    if contract is None:
        if args.preprocessing_contract is None:
            raise RuntimeError(
                "package lacks embedded preprocessing contract; pass --preprocessing-contract for audit only"
            )
        contract = json.loads(args.preprocessing_contract.read_text())
    if (
        contract.get("source_split") != "train"
        or contract.get("test_statistics_used", True)
        or contract.get("timestamp_metadata_used", False)
    ):
        raise RuntimeError(
            "raw replay requires train-only, timestamp-free preprocessing"
        )
    detector = args.detector.resolve()
    validate_contract(package, contract, detector)
    if not args.test_root.is_dir():
        raise FileNotFoundError(args.test_root)
    test_ids = (
        pd.read_csv(args.test_csv)["id"].astype(str).to_numpy()
        if "id" in pd.read_csv(args.test_csv).columns
        else None
    )
    if test_ids is None:
        # The competition test.csv stores the path column.  Derive only the
        # anonymous sample directory, never a user/session identifier.
        frame = pd.read_csv(args.test_csv)
        column = "path" if "path" in frame.columns else frame.columns[0]
        test_ids = (
            frame[column].astype(str).str.rstrip("/").str.split("/").str[-1].to_numpy()
        )
    if len(test_ids) == 0:
        raise RuntimeError("test.csv has no sample IDs")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(2026)
    np.random.seed(2026)

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="cuhkx_raw_replay_")
        work = Path(temporary.name)
    else:
        work = args.work_dir.resolve()
        work.mkdir(parents=True, exist_ok=True)
    try:
        windows = work / "windows.npz"
        run_command(
            [
                sys.executable,
                "-m",
                "yolo_r2plus1d.strict_v3.data.windows",
                "--test-only",
                "--device",
                str(device.index or 0),
                "--fallback-device",
                "cpu",
                "--weights",
                str(detector),
                "--test-root",
                str(args.test_root),
                "--test-csv",
                str(args.test_csv),
                "--output",
                str(windows),
            ],
            REPO_ROOT,
        )
        visual_dir = work / "visual"
        run_command(
            [
                sys.executable,
                "-m",
                "yolo_r2plus1d.strict_v3.data.cache",
                "--windows",
                str(windows),
                "--test-root",
                str(args.test_root),
                "--output-dir",
                str(visual_dir),
                "--test-only",
                "--workers",
                str(args.workers),
            ],
            REPO_ROOT,
        )
        skeleton_dir = work / "skeleton"
        run_command(
            [
                sys.executable,
                "-m",
                "yolo_r2plus1d.strict_v3.data.skeleton_cache",
                "--metadata",
                str(RESULT_DIR / "metadata.npz"),
                "--test-root",
                str(args.test_root),
                "--output-dir",
                str(skeleton_dir),
                "--test-only",
                "--workers",
                str(args.workers),
            ],
            REPO_ROOT,
        )
        visual_logits = infer_visual(
            visual_dir / "test_depth_ir.npy",
            contract,
            package,
            device,
            args.batch_size,
            args.workers,
        )
        fusion_logits = infer_fusion(
            visual_dir / "test_depth_ir.npy",
            skeleton_dir / "test_skeleton.npy",
            skeleton_dir / "test_skeleton_mask.npy",
            contract,
            package,
            device,
            args.batch_size,
            args.workers,
        )
        raw_frames = raw_dst_frames(args.test_root, test_ids, 16)
        temporal_logits = infer_temporal(raw_frames, package, device, args.batch_size)
        branches = {
            "fusion": fusion_logits,
            "visual": visual_logits,
            "temporal": temporal_logits,
        }
        # Keep branch outputs in the retained audit directory only.  They are
        # generated in this run (never consumed as inputs on a later replay),
        # which makes quantization and missingness audits reproducible.
        np.save(work / "fusion_logits.npy", fusion_logits)
        np.save(work / "visual_logits.npy", visual_logits)
        np.save(work / "temporal_logits.npy", temporal_logits)
        np.save(work / "raw_dst_frames.npy", raw_frames)
        with np.load(windows) as window_payload:
            detection_rate = float(
                np.asarray(window_payload["test_has_crop"], dtype=bool).mean()
            )
        probability_ensemble = release.get("probability_ensemble")
        prediction_consensus = release.get("prediction_consensus")
        if isinstance(prediction_consensus, dict) and isinstance(
            prediction_consensus.get("members"), list
        ):
            portfolio_contract = prediction_consensus
            portfolio_aggregation = "majority_prediction"
        else:
            portfolio_contract = probability_ensemble
            portfolio_aggregation = "probability_mean"
        ensemble_summary = None
        if isinstance(portfolio_contract, dict) and isinstance(
            portfolio_contract.get("members"), list
        ):
            portfolio = portfolio_contract["members"]
            if len(portfolio) < 2:
                raise RuntimeError("release portfolio requires at least two members")
            if portfolio_aggregation == "majority_prediction" and len(portfolio) != 3:
                raise RuntimeError(
                    "prediction consensus requires exactly three members"
                )
            expected_weight = 1.0 / len(portfolio)
            probability = np.zeros((len(test_ids), 40), dtype=np.float64)
            total_weight = 0.0
            anchor_fusion = None
            anchor_visual_cache: dict[tuple[float, float], np.ndarray] = {}
            member_summaries = []
            member_predictions = []
            weights = None
            for member_index, member in enumerate(portfolio):
                if not isinstance(member, dict):
                    raise RuntimeError("release portfolio member must be a mapping")
                name = str(member.get("name", f"member_{member_index}"))
                view = str(member.get("view", ""))
                member_weight = float(member.get("weight", float("nan")))
                if not np.isclose(member_weight, expected_weight):
                    raise RuntimeError(
                        "release portfolio is not a fixed equal-weight ensemble"
                    )
                member_contract = member.get("release_contract")
                if not isinstance(member_contract, dict):
                    raise RuntimeError(
                        f"portfolio member {name!r} lacks a release contract"
                    )
                for key, expected_value in (
                    ("per_sample_zscore", False),
                    ("thermal_weight", 0.0),
                    ("skeleton_mask_used_in_blend", False),
                    ("test_labels_used", False),
                    ("test_statistics_used", False),
                    ("timestamp_metadata_used", False),
                ):
                    if member_contract.get(key, release.get(key)) != expected_value:
                        raise RuntimeError(
                            f"unsafe portfolio member {name!r}: {key}="
                            f"{member_contract.get(key, release.get(key))!r}"
                        )
                if view == "candidate":
                    for scale_key in (
                        "visual_logit_scale",
                        "visual_package_output_scale",
                    ):
                        if not np.isclose(
                            float(member_contract.get(scale_key, 1.0)),
                            float(release.get(scale_key, 1.0)),
                        ):
                            raise RuntimeError(
                                f"candidate portfolio member {name!r} changes {scale_key}"
                            )
                    if member_contract.get(
                        "temporal_pool", "residual_tcn"
                    ) != release.get("temporal_pool", "residual_tcn"):
                        raise RuntimeError(
                            f"candidate portfolio member {name!r} changes temporal_pool"
                        )
                    member_branches = branches
                elif view == "shared":
                    for scale_key in (
                        "visual_logit_scale",
                        "visual_package_output_scale",
                    ):
                        if not np.isclose(
                            float(member_contract.get(scale_key, 1.0)),
                            float(release.get(scale_key, 1.0)),
                        ):
                            raise RuntimeError(
                                f"shared portfolio member {name!r} changes {scale_key}"
                            )
                    member_package = copy.copy(package)
                    shared_release = copy.deepcopy(release)
                    shared_release.update(copy.deepcopy(member_contract))
                    shared_release.pop("probability_ensemble", None)
                    shared_release.pop("anchor_reconstruction", None)
                    member_package["release_contract"] = shared_release
                    member_temporal = infer_temporal(
                        raw_frames,
                        member_package,
                        device,
                        args.batch_size,
                    )
                    member_branches = {
                        "fusion": fusion_logits,
                        "visual": visual_logits,
                        "temporal": member_temporal,
                    }
                    np.save(
                        work / f"member_{member_index}_temporal_logits.npy",
                        member_temporal,
                    )
                elif view == "anchor":
                    member_package = anchor_package_view(package, member_contract)
                    if anchor_fusion is None:
                        anchor_fusion = infer_fusion(
                            visual_dir / "test_depth_ir.npy",
                            skeleton_dir / "test_skeleton.npy",
                            skeleton_dir / "test_skeleton_mask.npy",
                            contract,
                            member_package,
                            device,
                            args.batch_size,
                            args.workers,
                        )
                    scale_key = (
                        float(member_contract.get("visual_logit_scale", 1.0)),
                        float(member_contract.get("visual_package_output_scale", 1.0)),
                    )
                    if scale_key not in anchor_visual_cache:
                        anchor_visual_cache[scale_key] = infer_visual(
                            visual_dir / "test_depth_ir.npy",
                            contract,
                            member_package,
                            device,
                            args.batch_size,
                            args.workers,
                        )
                    member_temporal = infer_temporal(
                        raw_frames,
                        member_package,
                        device,
                        args.batch_size,
                    )
                    member_branches = {
                        "fusion": anchor_fusion,
                        "visual": anchor_visual_cache[scale_key],
                        "temporal": member_temporal,
                    }
                    np.save(
                        work / f"member_{member_index}_temporal_logits.npy",
                        member_temporal,
                    )
                else:
                    raise RuntimeError(f"unsupported portfolio view {view!r}")
                member_fused, member_gate_weights = apply_gate(
                    member_branches,
                    member_contract["temperatures"],
                    member_contract["weights"],
                    quality_floor=float(member_contract.get("quality_floor", 0.25)),
                    quality_gamma=float(member_contract.get("quality_gamma", 1.0)),
                )
                probability += member_weight * softmax(member_fused)
                total_weight += member_weight
                member_predictions.append(member_fused.argmax(1).astype(np.int64))
                if weights is None:
                    weights = member_gate_weights
                np.save(
                    work / f"member_{member_index}_fused_logits.npy",
                    member_fused.astype(np.float32),
                )
                member_summaries.append(
                    {
                        "name": name,
                        "view": view,
                        "weight": member_weight,
                        "mean_gate_weights": member_gate_weights.mean(0).tolist(),
                        "branch_shapes": {
                            key: list(value.shape)
                            for key, value in member_branches.items()
                        },
                    }
                )
            if not np.isclose(total_weight, 1.0) or weights is None:
                raise RuntimeError("release portfolio weights do not sum to one")
            if portfolio_aggregation == "majority_prediction":
                fallback_name = str(portfolio_contract.get("tie_fallback", ""))
                member_names = [str(member.get("name", "")) for member in portfolio]
                if fallback_name not in member_names:
                    raise RuntimeError(
                        "prediction consensus tie fallback is not a member"
                    )
                stacked_predictions = np.stack(member_predictions, axis=0)
                vote_count = np.zeros((len(test_ids), 40), dtype=np.int16)
                rows = np.arange(len(test_ids))
                for member_prediction in member_predictions:
                    vote_count[rows, member_prediction] += 1
                majority = vote_count.argmax(1).astype(np.int64)
                has_majority = vote_count.max(1) >= 2
                fallback = stacked_predictions[member_names.index(fallback_name)]
                predictions = np.where(has_majority, majority, fallback).astype(
                    np.int64
                )
                np.save(work / "ensemble_member_predictions.npy", stacked_predictions)
                np.save(work / "ensemble_predictions.npy", predictions)
            else:
                predictions = probability.argmax(1).astype(np.int64)
            if anchor_fusion is not None:
                np.save(work / "anchor_fusion_logits.npy", anchor_fusion)
            for index, value in enumerate(anchor_visual_cache.values()):
                np.save(work / f"anchor_visual_logits_{index}.npy", value)
            probability_name = (
                "diagnostic_member_probability_mean.npy"
                if portfolio_aggregation == "majority_prediction"
                else "ensemble_probabilities.npy"
            )
            np.save(work / probability_name, probability.astype(np.float32))
            ensemble_summary = {
                "mode": (
                    "fixed_three_member_release_prediction_consensus"
                    if portfolio_aggregation == "majority_prediction"
                    else "fixed_equal_release_probability_portfolio"
                ),
                "members": member_summaries,
            }
            if portfolio_aggregation == "majority_prediction":
                ensemble_summary["tie_fallback"] = fallback_name
        elif isinstance(probability_ensemble, dict):
            candidate_fused, candidate_weights = apply_gate(
                branches,
                release["temperatures"],
                release["weights"],
                quality_floor=float(release.get("quality_floor", 0.25)),
                quality_gamma=float(release.get("quality_gamma", 1.0)),
            )
            anchor_package = anchor_package_view(package)
            anchor_release = anchor_package["release_contract"]
            anchor_visual = infer_visual(
                visual_dir / "test_depth_ir.npy",
                contract,
                anchor_package,
                device,
                args.batch_size,
                args.workers,
            )
            anchor_fusion = infer_fusion(
                visual_dir / "test_depth_ir.npy",
                skeleton_dir / "test_skeleton.npy",
                skeleton_dir / "test_skeleton_mask.npy",
                contract,
                anchor_package,
                device,
                args.batch_size,
                args.workers,
            )
            anchor_temporal = infer_temporal(
                raw_frames, anchor_package, device, args.batch_size
            )
            anchor_branches = {
                "fusion": anchor_fusion,
                "visual": anchor_visual,
                "temporal": anchor_temporal,
            }
            anchor_fused, anchor_weights = apply_gate(
                anchor_branches,
                anchor_release["temperatures"],
                anchor_release["weights"],
                quality_floor=float(anchor_release.get("quality_floor", 0.25)),
                quality_gamma=float(anchor_release.get("quality_gamma", 1.0)),
            )
            anchor_weight = float(probability_ensemble["anchor_weight"])
            candidate_weight = float(probability_ensemble["candidate_weight"])
            if not np.isclose(anchor_weight + candidate_weight, 1.0):
                raise RuntimeError("probability ensemble weights do not sum to one")
            probability = anchor_weight * softmax(
                anchor_fused
            ) + candidate_weight * softmax(candidate_fused)
            predictions = probability.argmax(1).astype(np.int64)
            np.save(work / "anchor_fusion_logits.npy", anchor_fusion)
            np.save(work / "anchor_visual_logits.npy", anchor_visual)
            np.save(work / "anchor_temporal_logits.npy", anchor_temporal)
            np.save(work / "anchor_fused_logits.npy", anchor_fused.astype(np.float32))
            np.save(
                work / "candidate_fused_logits.npy", candidate_fused.astype(np.float32)
            )
            np.save(work / "ensemble_probabilities.npy", probability.astype(np.float32))
            weights = candidate_weights
            ensemble_summary = {
                "mode": "fixed_probability_ensemble",
                "anchor_weight": anchor_weight,
                "candidate_weight": candidate_weight,
                "anchor_mean_gate_weights": anchor_weights.mean(0).tolist(),
                "candidate_mean_gate_weights": candidate_weights.mean(0).tolist(),
                "anchor_branch_shapes": {
                    key: list(value.shape) for key, value in anchor_branches.items()
                },
            }
        else:
            fused, weights = apply_gate(
                branches,
                release["temperatures"],
                release["weights"],
                quality_floor=float(release.get("quality_floor", 0.25)),
                quality_gamma=float(release.get("quality_gamma", 1.0)),
            )
            predictions = fused.argmax(1).astype(np.int64)
        sample = pd.read_csv(args.sample_submission)
        if len(sample) != len(predictions) or sample.columns.tolist() != [
            "path",
            "prediction",
        ]:
            raise RuntimeError("sample submission schema/length mismatch")
        sample["prediction"] = predictions
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(args.output, index=False)
        summary = {
            "output": str(args.output),
            "rows": int(len(predictions)),
            "unique_classes": int(np.unique(predictions).size),
            "class_ids": sorted(np.unique(predictions).astype(int).tolist()),
            "raw_work_dir": str(work),
            "detector": str(detector),
            "detector_sha256": sha256(detector),
            "test_detection_rate": detection_rate,
            "branch_shapes": {
                key: list(value.shape) for key, value in branches.items()
            },
            "mean_gate_weights": weights.mean(0).tolist(),
            "probability_ensemble": ensemble_summary,
            "package_sha256": sha256(args.package),
            "preprocessing_contract_embedded": embedded_contract,
            "preprocessing_contract_fallback": (
                None if embedded_contract else str(args.preprocessing_contract)
            ),
            "test_labels_used": False,
            "test_statistics_used": False,
            "timestamp_metadata_used": False,
        }
        args.output.with_suffix(".json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
