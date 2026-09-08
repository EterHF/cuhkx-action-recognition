#!/usr/bin/env python3
"""Build external-only nested OOF inputs for the conditional corrector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from yolo_r2plus1d.strict_v3.models.dstformer import DSTformer
from yolo_r2plus1d.strict_v3.paths import REPO_ROOT, RESULT_DIR
from yolo_r2plus1d.strict_v3.release.blend import apply_gate
from yolo_r2plus1d.strict_v3.training.base import seed_everything
from yolo_r2plus1d.strict_v3.training.public_finetune import VideoDataset, make_model

FOLDS = {
    "A": (1, 6, 17, 22),
    "B": (2, 7, 18, 23),
    "C": (3, 8, 19, 24),
    "D": (4, 9, 20),
    "E": (5, 16, 21),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_external_checkpoint(path: Path, modality: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload.get("encoder_state"), dict):
        raise RuntimeError(f"{modality} source has no encoder_state")
    if payload.get("cuhkx_data_used") is not False and modality == "temporal":
        raise RuntimeError("Temporal source does not explicitly exclude CUHK-X")
    if payload.get("test_statistics_used") is not False and modality == "temporal":
        raise RuntimeError("Temporal source does not explicitly exclude test statistics")
    return payload


@torch.inference_mode()
def extract_visual_features(args: argparse.Namespace, device: torch.device) -> np.ndarray:
    payload = checked_external_checkpoint(args.visual_source, "visual")
    source_split = json.loads(args.visual_source_split.read_text(encoding="utf-8"))
    if source_split.get("target_validation_users_used") is not False:
        raise RuntimeError("Visual source provenance does not exclude target users")
    if source_split.get("test_statistics_used") is not False:
        raise RuntimeError("Visual source provenance uses test statistics")
    model = make_model()
    missing, unexpected = model.encoder.load_state_dict(payload["encoder_state"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Visual source mismatch: {missing=}, {unexpected=}")
    model.to(device).eval()
    rows = len(np.load(args.metadata)["train_y"])
    dataset = VideoDataset(
        args.visual_cache,
        np.arange(rows),
        None,
        False,
        False,
        "all",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.visual_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    output = np.empty((rows, 512), dtype=np.float16)
    for frames, indices in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16):
            features = model.forward_features(frames.to(device, non_blocking=True))
        output[indices.numpy()] = features.float().cpu().numpy().astype(np.float16)
    return output


class SkeletonSet(Dataset):
    def __init__(self, root: Path) -> None:
        self.skeleton = np.load(root / "train_skeleton.npy", mmap_mode="r")
        self.mask = np.load(root / "train_skeleton_mask.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.skeleton)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return (
            torch.from_numpy(np.array(self.skeleton[index], copy=True)),
            torch.from_numpy(np.array(self.mask[index], copy=True)),
            index,
        )


@torch.inference_mode()
def extract_temporal_features(args: argparse.Namespace, device: torch.device) -> np.ndarray:
    payload = checked_external_checkpoint(args.temporal_source, "temporal")
    config = payload["model_config"]
    model = DSTformer(
        dim_in=3,
        dim_out=0,
        dim_feat=int(config["dim_feat"]),
        dim_rep=int(config["dim_rep"]),
        depth=int(config["depth"]),
        num_heads=int(config["heads"]),
        mlp_ratio=2,
        num_joints=17,
        maxlen=int(config["frames"]),
        drop_rate=0.1,
        attn_drop_rate=0.05,
        att_fuse=True,
    )
    model.load_state_dict(payload["encoder_state"], strict=True)
    model.to(device).eval()
    dataset = SkeletonSet(args.skeleton_root)
    loader = DataLoader(
        dataset,
        batch_size=args.temporal_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    width = int(config["dim_rep"])
    frames = int(config["frames"])
    output = np.empty((len(dataset), width), dtype=np.float16)
    for skeleton, mask, indices in loader:
        skeleton = skeleton.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        relative = skeleton.clone()
        relative[:, :, 0].zero_()
        relative *= mask[:, :, None, None].to(relative.dtype)
        batch = len(skeleton)
        relative = (
            F.interpolate(
                relative.permute(0, 2, 3, 1).reshape(batch, 51, -1),
                size=frames,
                mode="linear",
                align_corners=False,
            )
            .reshape(batch, 17, 3, frames)
            .permute(0, 3, 1, 2)
        )
        resized_mask = F.interpolate(mask[:, None].float(), size=frames, mode="nearest")[:, 0]
        with torch.autocast(device.type, dtype=torch.bfloat16):
            features = model.get_representation(relative).mean(dim=2)
            features = (features * resized_mask[:, :, None]).sum(dim=1)
            features /= resized_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        output[indices.numpy()] = features.float().cpu().numpy().astype(np.float16)
    return output


class FixedHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 40))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def fit_head(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    held_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    seed_everything(seed)
    model = FixedHead(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.head_lr, weight_decay=0.01)
    dataset = TensorDataset(
        torch.from_numpy(features[train_indices].astype(np.float32)),
        torch.from_numpy(labels[train_indices]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.head_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(args.head_epochs):
        model.train()
        for values, target in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(values.to(device))
            loss = F.cross_entropy(logits, target.to(device), label_smoothing=0.02)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        return (
            model(torch.from_numpy(features[held_indices].astype(np.float32)).to(device))
            .cpu()
            .numpy()
        )


def fusion_contract(manifest: dict, fold: str) -> tuple[dict, dict]:
    record = next(item for item in manifest["fold_contracts"] if item["fold"] == fold)
    weights = dict(record["weights"])
    weights["fusion"] = 0.0
    return record["temperatures"], weights


def visual_output_scale(manifest: dict) -> float:
    scale = float(manifest.get("blend", {}).get("visual_package_output_scale", 0.0))
    if not 0.0 < scale <= 1.0:
        raise RuntimeError("release manifest has no safe Visual output scale")
    return scale


def baseline_logits(
    visual: np.ndarray, temporal: np.ndarray, temperatures: dict, weights: dict, scale: float
) -> np.ndarray:
    fused, _ = apply_gate(
        {"fusion": np.zeros_like(visual), "visual": visual * scale, "temporal": temporal},
        temperatures,
        weights,
    )
    return fused.astype(np.float32)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"].astype(np.int64)
        users = metadata["train_users"].astype(np.int64)
    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    scale = visual_output_scale(release)
    visual_features = extract_visual_features(args, device)
    temporal_features = extract_temporal_features(args, device)
    np.save(args.output_dir / "external_visual_features.npy", visual_features)
    np.save(args.output_dir / "external_temporal_features.npy", temporal_features)
    manifest = {"schema_version": "conditional-corrector-nested-oof/v1", "folds": []}
    audit = {
        "protocol": "external-only-fixed-encoder-nested-oof/v1",
        "runs": [],
        "outer_runs": [],
    }
    for outer_position, (outer, held_tuple) in enumerate(FOLDS.items()):
        outer_dir = args.output_dir / f"outer{outer}"
        outer_dir.mkdir()
        outer_users = set(held_tuple)
        outer_train = np.flatnonzero(~np.isin(users, list(outer_users)))
        outer_held = np.flatnonzero(np.isin(users, list(outer_users)))
        train_visual = np.empty((len(outer_train), 40), dtype=np.float32)
        train_temporal = np.empty_like(train_visual)
        location = {int(index): position for position, index in enumerate(outer_train)}
        assigned = np.zeros(len(outer_train), dtype=bool)
        for inner_position, (inner, inner_tuple) in enumerate(FOLDS.items()):
            if inner == outer:
                continue
            inner_held = np.flatnonzero(np.isin(users, list(inner_tuple)))
            fit_users = (
                set(int(value) for value in np.unique(users)) - outer_users - set(inner_tuple)
            )
            fit_indices = np.flatnonzero(np.isin(users, list(fit_users)))
            destinations = np.asarray([location[int(index)] for index in inner_held])
            if bool(assigned[destinations].any()):
                raise RuntimeError("inner folds overlap")
            assigned[destinations] = True
            run_seed = args.seed + outer_position * 100 + inner_position
            train_visual[destinations] = fit_head(
                visual_features, labels, fit_indices, inner_held, args, device, run_seed
            )
            train_temporal[destinations] = fit_head(
                temporal_features, labels, fit_indices, inner_held, args, device, run_seed + 50
            )
            audit["runs"].append(
                {
                    "outer": outer,
                    "inner": inner,
                    "fit_users": sorted(fit_users),
                    "excluded_users": sorted(outer_users | set(inner_tuple)),
                    "held_rows": len(inner_held),
                    "checkpoint_selection": "fixed_final_epoch",
                }
            )
        if not bool(assigned.all()):
            raise RuntimeError(f"outer {outer} inner folds do not cover outer-train")
        held_visual = fit_head(
            visual_features,
            labels,
            outer_train,
            outer_held,
            args,
            device,
            args.seed + 1000 + outer_position,
        )
        held_temporal = fit_head(
            temporal_features,
            labels,
            outer_train,
            outer_held,
            args,
            device,
            args.seed + 1050 + outer_position,
        )
        audit["outer_runs"].append(
            {
                "outer": outer,
                "fit_users": sorted(set(int(value) for value in users[outer_train])),
                "excluded_users": sorted(outer_users),
                "held_rows": len(outer_held),
                "checkpoint_selection": "fixed_final_epoch",
            }
        )
        temperatures, weights = fusion_contract(release, outer)
        train_baseline = baseline_logits(train_visual, train_temporal, temperatures, weights, scale)
        held_baseline = baseline_logits(held_visual, held_temporal, temperatures, weights, scale)
        paths = {}
        for split, indices, visual_logits, temporal_logits, base in (
            ("train", outer_train, train_visual, train_temporal, train_baseline),
            ("held", outer_held, held_visual, held_temporal, held_baseline),
        ):
            split_dir = outer_dir / split
            split_dir.mkdir()
            arrays = {
                "indices": indices,
                "visual_features": visual_features[indices],
                "visual_logits": visual_logits,
                "temporal_logits": temporal_logits,
                "baseline_logits": base,
            }
            paths[split] = {}
            for name, values in arrays.items():
                path = split_dir / f"{name}.npy"
                np.save(path, values)
                paths[split][name] = str(path.resolve())
        common = {
            "upstream_excluded_users": sorted(outer_users),
            "target_labels_used_for_upstream_selection": False,
            "external_visual_source": str(args.visual_source.resolve()),
            "external_temporal_source": str(args.temporal_source.resolve()),
        }
        manifest["folds"].append(
            {
                "fold": outer,
                "held_users": sorted(outer_users),
                "train": {**paths["train"], **common, "cross_fitted_within_outer_train": True},
                "held": {**paths["held"], **common, "cross_fitted_within_outer_train": False},
            }
        )
    audit.update(
        {
            "head_epochs": args.head_epochs,
            "head_lr": args.head_lr,
            "visual_source_sha256": sha256(args.visual_source),
            "temporal_source_sha256": sha256(args.temporal_source),
            "visual_normalization": "fixed Kinetics constants; no target/test fitted statistics",
            "visual_package_output_scale": scale,
            "held_labels_used_for_checkpoint_selection": False,
            "anonymous_test_accessed": False,
        }
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({key: value for key, value in audit.items() if key != "runs"}, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    result.add_argument(
        "--release-manifest", type=Path, default=RESULT_DIR / "release_manifest.json"
    )
    result.add_argument("--visual-cache", type=Path, required=True)
    result.add_argument(
        "--skeleton-root", type=Path, default=REPO_ROOT / ".cache/highrate_cross_attention/skeleton"
    )
    result.add_argument("--visual-source", type=Path, required=True)
    result.add_argument("--visual-source-split", type=Path, required=True)
    result.add_argument("--temporal-source", type=Path, required=True)
    result.add_argument("--head-epochs", type=int, default=60)
    result.add_argument("--head-lr", type=float, default=3e-3)
    result.add_argument("--head-batch-size", type=int, default=128)
    result.add_argument("--visual-batch-size", type=int, default=16)
    result.add_argument("--temporal-batch-size", type=int, default=64)
    result.add_argument("--workers", type=int, default=8)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--device", default="cuda:0")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
