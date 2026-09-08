import torch
from cafd.mistral_direction_v15 import targets

def test_targets_normalized_and_shift_invariant():
    torch.manual_seed(2027)
    inputs = [torch.randn(4, 17).log_softmax(-1) for _ in range(3)]
    q, mix, alpha = targets(*inputs)
    assert torch.allclose(q.exp().sum(-1), torch.ones(4), atol=1e-6)
    assert torch.allclose(mix.exp().sum(-1), torch.ones(4), atol=1e-6)
    assert bool(((alpha >= 0) & (alpha <= 1)).all())
    # Inputs to helper are normalized log probabilities, as returned by log_probs.
    same = targets(inputs[0], inputs[0], inputs[0])
    assert torch.equal(same[2], torch.zeros(4))
    assert torch.allclose(same[1], inputs[0], atol=1e-6)

def test_no_teacher_delta_preserves_relative_reference():
    ref = torch.tensor([[0., 1., 2.]]).log_softmax(-1)
    teacher = torch.tensor([[2., 1., 0.]]).log_softmax(-1)
    q, mix, alpha = targets(ref, teacher, teacher)
    assert torch.allclose(q, ref, atol=1e-6)
    assert torch.allclose(mix, teacher, atol=1e-6)
    assert alpha.item() == 0
