"""A small binary selector that cannot invent a third class or alter confident rows."""

from __future__ import annotations

import math

import torch
from torch import nn


class BranchSelector(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 2))


def selector_inputs(visual, temporal, baseline):
    vp, tp, bp = (value.softmax(1) for value in (visual * .5, temporal, baseline))
    vi, ti = vp.argmax(1), tp.argmax(1)
    row = torch.arange(len(bp), device=bp.device)
    def entropy(p):
        return -(p * p.clamp_min(1e-12).log()).sum(1) / math.log(p.shape[1])
    features = torch.stack((tp[row, ti] - tp[row, vi], vp[row, vi] - vp[row, ti],
                            bp[row, vi] - bp[row, ti], entropy(tp), entropy(vp), bp.max(1).values), 1)
    return features, ti, vi


@torch.inference_mode()
def select_branches(model, visual, temporal, baseline, both_valid):
    if both_valid.dtype != torch.bool or both_valid.shape != (len(baseline),):
        raise ValueError("both_valid must be a row boolean mask")
    features, ti, vi = selector_inputs(visual, temporal, baseline)
    eligible = both_valid & (ti != vi) & (features[:, 5] < .8)
    result = baseline.clone()
    if bool(eligible.any()):
        choose_visual = model(features[eligible]).argmax(1).bool()
        choice = torch.where(choose_visual, vi[eligible], ti[eligible])
        rows = eligible.nonzero().flatten()
        # Preserve a finite logit representation while making the binary choice
        # explicit. This final result must not be fed back into a learned fusion.
        result[rows, choice] = baseline[eligible].max(1).values + 1
    return result


def unanimous_selection(baseline, members):
    if len(members) != 3 or any(m.shape != baseline.shape for m in members):
        raise ValueError("unanimous selector requires three matching member arrays")
    choices = torch.stack([m.argmax(1) for m in members])
    change = (choices == choices[0]).all(0) & (choices[0] != baseline.argmax(1))
    result = baseline.clone()
    result[change] = members[0][change]
    return result
