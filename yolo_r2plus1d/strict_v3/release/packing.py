"""Small helpers needed to reconstruct compact release members."""

from collections.abc import Mapping

import torch

from yolo_r2plus1d.strict_v3.training.public_finetune import dequantize_state


def reconstruct_state(
    source: Mapping[str, object], recipe: Mapping[str, object]
) -> dict[str, torch.Tensor]:
    output = dequantize_state(source)
    for key, record in recipe.items():
        if record["kind"] == "literal":
            output[key] = record["value"]
            continue
        shape = tuple(int(value) for value in record["shape"])
        bits = int(record["bits"])
        qmax = (1 << (bits - 1)) - 1
        scale = record["scale"].float()
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        code = (
            torch.round(output[key].float() / safe_scale)
            .clamp(-qmax - 1, qmax)
            .reshape(-1)
            .to(torch.int16)
        )
        code[record["indices"].long()] = record["values"].to(torch.int16)
        output[key] = (code.reshape(shape).float() * scale).to(torch.float32)
    return output
