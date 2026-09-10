import torch

from yolo_r2plus1d.strict_v3.models.branch_selector import (
    BranchSelector,
    select_branches,
    unanimous_selection,
)
from yolo_r2plus1d.strict_v3.training.branch_selector import fit_selector


def test_selector_is_bounded_to_branch_choices_and_preserves_confident_or_unavailable_rows():
    model = BranchSelector().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model[-1].bias[1] = 100
    visual = torch.tensor([[0., 2., 1.]] * 4)
    temporal = torch.tensor([[2., 0., 1.]] * 4)
    base = torch.tensor([[9., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
    temporal[3] = visual[3]
    actual = select_branches(model, visual, temporal, base, torch.tensor([True, False, True, True]))
    assert torch.equal(actual[[0, 1, 3]], base[[0, 1, 3]])
    assert actual[2].argmax() == visual[2].argmax()


def test_selector_excludes_held_unavailable_agreement_and_unrepresentable_labels():
    torch.set_num_threads(1)
    visual = torch.tensor([[0., 2., 1.]] * 6)
    temporal = torch.tensor([[2., 0., 1.]] * 6)
    temporal[4] = visual[4]
    _, selected = fit_selector(visual, temporal, temporal,
                              torch.tensor([0, 1, 0, 0, 1, 2]),
                              torch.tensor([True, True, True, False, True, True]),
                              torch.tensor([True, True, False, True, True, True]), 2026)
    assert selected.tolist() == [True, True, False, False, False, False]


def test_consensus_requires_every_member_and_preserves_baseline_on_disagreement():
    base = torch.tensor([[3., 0., 1.], [3., 0., 1.], [3., 0., 1.]])
    member = torch.tensor([[0., 3., 1.], [0., 3., 1.], [30., 0., 1.]])
    third = member.clone()
    third[1] = torch.tensor([0., 1., 3.])
    result = unanimous_selection(base, [member, member, third])
    assert result[0].argmax() == 1
    assert torch.equal(result[1:], base[1:])
