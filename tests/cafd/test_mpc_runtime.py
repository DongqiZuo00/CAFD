from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import cafd.mpc_runtime as runtime
from cafd.optimizer import FP32AdamW


def test_behavior_recorder_aligns_first_middle_last_sampled_tokens_and_is_fp32():
    recorder = runtime.BehaviorRecorder()
    prefix = torch.tensor([[1, 7], [1, 7]])
    all_scores = [
        torch.tensor([[0., 1., 2., 3.], [3., 2., 1., 0.]], dtype=torch.bfloat16),
        torch.tensor([[2., 3., 0., 1.], [1., 0., 3., 2.]], dtype=torch.bfloat16),
        torch.tensor([[1., 3., 2., 0.], [2., 0., 1., 3.]], dtype=torch.bfloat16),
    ]
    sampled = torch.tensor([[3, 0, 1], [0, 2, 3]])
    for i, scores in enumerate(all_scores):
        returned = recorder(prefix, scores)
        assert returned is scores
        assert recorder.previous.dtype == torch.float32
        prefix = torch.cat([prefix, sampled[:, i:i + 1]], dim=1)
    actual = torch.tensor(recorder.finish(prefix))
    expected = torch.stack([
        scores.float().log_softmax(-1).gather(1, sampled[:, i:i + 1]).squeeze(1)
        for i, scores in enumerate(all_scores)
    ], dim=1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert recorder.previous is None and len(recorder.selected) == 3


def test_completion_stop_records_only_first_stop_per_row_without_injecting_tokens():
    tokenizer = SimpleNamespace(eos_token_id=2,
        decode=lambda tokens, **kwargs: "CODE" + ("\n" + chr(96)*3)*2 if 6 in tokens else "CODE")
    fence = lambda ids, scores: ids[:, -1] == 6
    with patch.object(runtime, "StopStringCriteria", return_value=fence):
        stopper = runtime.CompletionStop(tokenizer, prompt_length=2)
    prefix = torch.tensor([[1, 7], [1, 7], [1, 7]])
    first = torch.tensor([[3], [6], [4]])
    ids = torch.cat([prefix, first], 1)
    before = ids.clone()
    assert stopper(ids, None).tolist() == [False, True, False]
    torch.testing.assert_close(ids, before)
    ids = torch.cat([ids, torch.tensor([[2], [0], [5]])], 1)
    assert stopper(ids, None).tolist() == [True, False, False]
    ids = torch.cat([ids, torch.tensor([[0], [0], [2]])], 1)
    assert stopper(ids, None).tolist() == [False, False, True]
    assert stopper.lengths.tolist() == [2, 1, 3]


def test_restore_optimizer_really_preserves_fp32_precision_and_next_step():
    parameter = torch.nn.Parameter(torch.tensor([1., -.5, 2.], dtype=torch.bfloat16))
    optimizer = FP32AdamW([parameter], lr=1e-5)
    for gradient in ([.011, -.013, .019], [.014, -.007, .031], [.023, .005, -.002]):
        parameter.grad = torch.tensor(gradient, dtype=torch.bfloat16)
        optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    tensor_keys = ["master", "exp_avg", "exp_avg_sq"]
    original_state = saved["state"][0]
    # Fixture catches dtype-correct but already quantized restores.
    assert any(not torch.equal(original_state[k], original_state[k].bfloat16().float()) for k in tensor_keys)
    naive_parameter = torch.nn.Parameter(parameter.detach().clone())
    naive = FP32AdamW([naive_parameter], lr=1e-5)
    naive.load_state_dict(copy.deepcopy(saved))
    assert any(not torch.equal(original_state[k], naive.state[naive_parameter][k]) for k in tensor_keys)
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = FP32AdamW([restored_parameter], lr=.4)
    runtime.restore_optimizer_exact(restored, saved)
    for key in tensor_keys:
        restored_tensor = restored.state[restored_parameter][key]
        assert restored_tensor.dtype == torch.float32
        torch.testing.assert_close(restored_tensor, original_state[key], atol=0, rtol=0)
        assert restored_tensor.data_ptr() != original_state[key].data_ptr()
    gradient = torch.tensor([.013, -.017, .023], dtype=torch.bfloat16)
    parameter.grad = gradient.clone()
    restored_parameter.grad = gradient.clone()
    optimizer.step()
    restored.step()
    torch.testing.assert_close(parameter, restored_parameter, atol=0, rtol=0)
    for key in tensor_keys:
        torch.testing.assert_close(optimizer.state[parameter][key], restored.state[restored_parameter][key], atol=0, rtol=0)
    assert optimizer.state[parameter]["step"] == restored.state[restored_parameter]["step"]


def test_independent_copy_and_refresh_do_not_share_parameters_or_buffers():
    class Buffered(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(3, 4)
            self.register_buffer("history", torch.arange(3.))
    student = Buffered().train()
    frozen = runtime.independent_copy(student)
    assert not frozen.training
    assert all(not p.requires_grad for p in frozen.parameters())
    for original, detached in zip(student.parameters(), frozen.parameters()):
        assert original.data_ptr() != detached.data_ptr()
    assert student.history.data_ptr() != frozen.history.data_ptr()
    initial = copy.deepcopy(frozen.state_dict())
    with torch.no_grad():
        student.layer.weight.add_(1)
        student.history.add_(1)
    for key, value in frozen.state_dict().items():
        torch.testing.assert_close(value, initial[key])
    runtime.refresh_behavior(frozen, student)
    for key, value in frozen.state_dict().items():
        torch.testing.assert_close(value, student.state_dict()[key])
        assert value.data_ptr() != student.state_dict()[key].data_ptr()


def test_fp32_generation_head_ignores_external_autocast():
    head = torch.nn.Linear(3, 9).bfloat16()
    head._mpc_weight = head.weight.detach().float()
    head._mpc_bias = head.bias.detach().float()
    hidden = torch.randn(2, 3).bfloat16()
    expected = F.linear(hidden.float(), head._mpc_weight, head._mpc_bias)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = runtime._fp32_head_forward(head, hidden)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_target_chunks_ignore_external_autocast_and_cover_full_vocabulary():
    torch.manual_seed(53)
    sources = [
        (1., torch.randn(7, 3).bfloat16(), torch.randn(11, 3).bfloat16(), None),
        (-.3, torch.randn(7, 4).bfloat16(), torch.randn(11, 4).bfloat16(), torch.randn(11).bfloat16()),
    ]
    expected_logits = sum(c * F.linear(h.float(), w.float(), None if b is None else b.float()) for c, h, w, b in sources)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        chunks = list(runtime.target_chunks(sources, chunk=3))
    assert [(start, stop) for start, stop, _ in chunks] == [(0, 3), (3, 6), (6, 7)]
    actual = torch.cat([probabilities for _, _, probabilities in chunks])
    assert actual.shape == (7, 11) and actual.dtype == torch.float32 and not actual.requires_grad
    torch.testing.assert_close(actual, expected_logits.softmax(-1), atol=1e-7, rtol=1e-7)


class FakeTokenizer:
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = 0
    def decode(self, tokens, skip_special_tokens=True):
        return "CODE" + ("\n" + chr(96)*3)*2 if 6 in tokens else "CODE"


def test_completion_stop_does_not_mistake_leading_newline_opening_fence_for_closing():
    class FenceTokenizer:
        eos_token_id = 2
        def decode(self, tokens, **kwargs):
            symbols = {8: "\n", 6: chr(96)*3, 3: "manufactoria\ncode\n"}
            return "".join(symbols.get(int(token), "") for token in tokens)
    fence = lambda ids, scores: ids[:, -1] == 6
    with patch.object(runtime, "StopStringCriteria", return_value=fence):
        stopper = runtime.CompletionStop(FenceTokenizer(), prompt_length=2)
    prefix = torch.tensor([[1, 7]])
    opening = torch.cat([prefix, torch.tensor([[8, 6]])], 1)
    assert not bool(stopper(opening, None).item()), "Opening fence is not a stopping event"
    closing = torch.cat([opening, torch.tensor([[3, 6]])], 1)
    assert bool(stopper(closing, None).item())
    assert stopper.lengths.tolist() == [4]


class FakeLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 9).bfloat16()
        self.actual_scores = []
    def get_output_embeddings(self):
        return self.head
    def generate(self, *, input_ids, attention_mask, generation_config, logits_processor, stopping_criteria):
        assert generation_config.temperature == 1. and generation_config.top_p == 1.
        assert generation_config.top_k == 0 and generation_config.repetition_penalty == 1.
        assert generation_config.do_sample
        count = generation_config.num_return_sequences
        ids = input_ids.repeat_interleave(count, dim=0)
        alive = torch.ones(count, dtype=torch.bool)
        # Row 0 ends in natural EOS; row 1 ends in closing fence; row 2 EOS at cap.
        emitted = torch.tensor([[3, 2, 0], [4, 6, 0], [5, 3, 2]])
        for step in range(min(generation_config.max_new_tokens, emitted.shape[1])):
            hidden = torch.arange(count * 3, dtype=torch.bfloat16).reshape(count, 3) / 20 + step / 10
            scores = self.head(hidden)
            for processor in logits_processor:
                scores = processor(ids, scores)
            self.actual_scores.append(scores.detach().clone().float())
            chosen = emitted[:, step].clone()
            chosen[~alive] = generation_config.pad_token_id
            ids = torch.cat([ids, chosen[:, None]], dim=1)
            for stop in stopping_criteria:
                alive &= ~stop(ids, scores)
            if not bool(alive.any()):
                break
        return ids


def _generation_patches():
    return [
        patch.object(runtime, "bounded_prompt", return_value="prompt"),
        patch.object(runtime, "encode_prompt", return_value={
            "input_ids": torch.tensor([[1, 7]]), "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }),
        patch.object(runtime, "StopStringCriteria", return_value=lambda ids, scores: ids[:, -1] == 6),
        patch.object(runtime, "score_completion_details", return_value={"reward": 0., "tier": "invalid"}),
    ]


def test_generate_fake_cpu_preserves_behavior_logs_stops_masks_and_model_state():
    from contextlib import ExitStack
    lm = FakeLM().train()
    wrapper = SimpleNamespace(causal_lm=lm)
    original_forward = lm.head.forward
    row = {"id": "tiny", "problem_family": "test", "ground_truth": {}}
    with ExitStack() as stack:
        for context in _generation_patches():
            stack.enter_context(context)
        records = runtime.generate(wrapper, FakeTokenizer(), row, count=3, max_new_tokens=3, seed=2027)
    assert [record["completion_ids"] for record in records] == [[3, 2], [4, 6], [5, 3, 2]]
    assert lm.training and lm.head.forward == original_forward
    assert not hasattr(lm.head, "_mpc_weight") and not hasattr(lm.head, "_mpc_bias")
    for index, record in enumerate(records):
        expected = [lm.actual_scores[step][index].log_softmax(-1)[token].item() for step, token in enumerate(record["completion_ids"])]
        torch.testing.assert_close(torch.tensor(record["old_log_probs"]), torch.tensor(expected), atol=0, rtol=0)
        ids, attention, mask = runtime.sequence_inputs(record, FakeTokenizer(), torch.device("cpu"))
        assert int(mask.sum()) == len(record["completion_ids"])
        assert ids[:, 1:][mask].tolist() == record["completion_ids"]
        assert not record["length_cap"], "Natural stop at exact limit is not truncation"


def test_generate_restores_head_and_training_mode_if_stop_constructor_fails():
    from contextlib import ExitStack
    lm = FakeLM().train()
    wrapper = SimpleNamespace(causal_lm=lm)
    original_forward = lm.head.forward
    row = {"id": "tiny", "problem_family": "test", "ground_truth": {}}
    with ExitStack() as stack:
        for context in _generation_patches()[:2]:
            stack.enter_context(context)
        stack.enter_context(patch.object(runtime, "StopStringCriteria", side_effect=RuntimeError("fake tokenizer failure")))
        with pytest.raises(RuntimeError, match="fake tokenizer"):
            runtime.generate(wrapper, FakeTokenizer(), row, count=3, max_new_tokens=3, seed=2027)
    assert lm.training and lm.head.forward == original_forward
    assert not hasattr(lm.head, "_mpc_weight") and not hasattr(lm.head, "_mpc_bias")


def test_generate_restores_all_mutations_if_generation_config_fails():
    from contextlib import ExitStack
    lm = FakeLM().train()
    wrapper = SimpleNamespace(causal_lm=lm)
    original_forward = lm.head.forward
    row = {"id": "tiny", "problem_family": "test", "ground_truth": {}}
    with ExitStack() as stack:
        for context in _generation_patches():
            stack.enter_context(context)
        stack.enter_context(patch.object(runtime, "GenerationConfig", side_effect=RuntimeError("fake config failure")))
        with pytest.raises(RuntimeError, match="fake config failure"):
            runtime.generate(wrapper, FakeTokenizer(), row, count=3, max_new_tokens=3, seed=2027)
    assert lm.training and lm.head.forward == original_forward
    assert not hasattr(lm.head, "_mpc_weight") and not hasattr(lm.head, "_mpc_bias")


def test_generate_restores_partial_head_setup_and_preserves_original_error():
    from contextlib import ExitStack
    class ExplodingHead(torch.nn.Linear):
        def __setattr__(self, name, value):
            if name == "_mpc_bias":
                raise RuntimeError("fake head allocation failure")
            super().__setattr__(name, value)
    lm = FakeLM().train()
    lm.head = ExplodingHead(3, 9).bfloat16()
    wrapper = SimpleNamespace(causal_lm=lm)
    original_forward = lm.head.forward
    row = {"id": "tiny", "problem_family": "test", "ground_truth": {}}
    with ExitStack() as stack:
        for context in _generation_patches():
            stack.enter_context(context)
        with pytest.raises(RuntimeError, match="fake head allocation failure"):
            runtime.generate(wrapper, FakeTokenizer(), row, count=3, max_new_tokens=3, seed=2027)
    assert lm.training and lm.head.forward == original_forward
    assert not hasattr(lm.head, "_mpc_weight") and not hasattr(lm.head, "_mpc_bias")


def test_real_transformers_categorical_sampling_old_logp_matches_replayed_policy():
    from contextlib import ExitStack
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(42)
    config = GPT2Config(vocab_size=9, n_positions=32, n_embd=8, n_layer=1, n_head=2,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
        resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)
    lm = GPT2LMHeadModel(config).eval()
    wrapper = SimpleNamespace(causal_lm=lm)
    row = {"id": "tiny", "problem_family": "test", "ground_truth": {}}
    rng_before = torch.get_rng_state().clone()
    with ExitStack() as stack:
        for context in _generation_patches()[:2]:
            stack.enter_context(context)
        stack.enter_context(patch.object(runtime, "StopStringCriteria",
            return_value=lambda ids, scores: torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)))
        stack.enter_context(patch.object(runtime, "score_completion_details", return_value={"reward": 0., "tier": "invalid"}))
        records = runtime.generate(wrapper, FakeTokenizer(), row, count=3, max_new_tokens=5, seed=2027)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, atol=0, rtol=0)
    assert len(records) == 3
    for record in records:
        ids = torch.tensor([record["prompt_ids"] + record["completion_ids"]])
        with torch.no_grad():
            logits = lm(ids).logits[:, :-1].float()
            tokens = ids[:, 1:]
            logp = logits.log_softmax(-1).gather(-1, tokens[..., None]).squeeze(-1)
        expected = logp[0, len(record["prompt_ids"]) - 1:]
        torch.testing.assert_close(torch.tensor(record["old_log_probs"]), expected, atol=2e-6, rtol=2e-6)
        if 2 in record["completion_ids"]:
            assert record["completion_ids"][-1] == 2
