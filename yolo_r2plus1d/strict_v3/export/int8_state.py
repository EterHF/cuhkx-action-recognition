"""Per-output-channel int8 state packing for public Omnivore branches."""

from __future__ import annotations

from pathlib import Path

import torch


def keep_fp16(name: str, value: torch.Tensor) -> bool:
    return (
        value.ndim < 2
        or name.startswith("head.")
        or "patch_embed" in name
        or name.endswith("relative_position_bias_table")
    )


def pack_state(state: dict[str, torch.Tensor]) -> dict:
    packed: dict[str, object] = {"format": "per-output-channel-int8/v1", "tensors": {}}
    tensors: dict[str, dict[str, torch.Tensor | str]] = packed["tensors"]  # type: ignore[assignment]
    for name, value in state.items():
        if not value.is_floating_point():
            continue
        value = value.float()
        if keep_fp16(name, value):
            tensors[name] = {"kind": "fp16", "value": value.half()}
            continue
        dimensions = tuple(range(1, value.ndim))
        scale = value.abs().amax(dim=dimensions).clamp_min_(1e-12).div_(127)
        view = (len(scale),) + (1,) * (value.ndim - 1)
        quantized = torch.round(value / scale.view(view)).clamp_(-127, 127).to(torch.int8)
        tensors[name] = {"kind": "int8", "value": quantized, "scale": scale.half()}
    return packed


def unpack_state(package: dict, template: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if package.get("format") != "per-output-channel-int8/v1":
        raise ValueError("unsupported int8 package")
    packed = package["tensors"]
    output = {}
    for name, reference in template.items():
        if not reference.is_floating_point():
            output[name] = reference
            continue
        record = packed[name]
        if record["kind"] == "fp16":
            output[name] = record["value"].to(reference.dtype)
        else:
            scale = record["scale"].float()
            view = (len(scale),) + (1,) * (reference.ndim - 1)
            output[name] = (record["value"].float() * scale.view(view)).to(reference.dtype)
    return output


def pack_checkpoint(source: Path, destination: Path) -> None:
    state = torch.load(source, map_location="cpu", weights_only=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack_state(state), destination)
