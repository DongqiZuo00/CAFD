import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import (GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM,
                          Ministral3Config, Ministral3ForCausalLM)

from cafd.target_path_t1 import (CachedModel, canonical_terms, combined_step,
    left_pad, generate_batch, GenerationFailure, run_policy, verify_record)


def tiny(seed, kind="llama"):
    torch.manual_seed(seed)
    if kind == "gpt2":
        model = GPT2LMHeadModel(GPT2Config(vocab_size=17,n_embd=16,n_layer=2,n_head=2,n_positions=64))
    elif kind == "ministral3":
        model = Ministral3ForCausalLM(Ministral3Config(vocab_size=17,hidden_size=16,intermediate_size=32,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,head_dim=8,
            max_position_embeddings=64,sliding_window=None,pad_token_id=0,eos_token_id=16))
    else:
        model = LlamaForCausalLM(LlamaConfig(vocab_size=17,hidden_size=16,intermediate_size=32,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64))
    return CachedModel(model)


@pytest.mark.parametrize("kind",["llama","gpt2","ministral3"])
def test_separate_kv_leftpad_logits_equal_fullprefix_and_unpadded(kind):
    models = {"s":tiny(1,kind),"t":tiny(2,kind),"base":tiny(3,kind)}
    terms = {"s":1.,"t":1.,"base":-1.}
    prompts = [[1,3,4],[1,5],[1,6,7,8,9]]
    ids, attention = left_pad(prompts,0,"cpu")
    caches, sequence = {}, ids.clone()
    for step in range(4):
        cached = combined_step(models,terms,ids,attention,caches)
        reference = combined_step(models,terms,sequence,attention,{})
        torch.testing.assert_close(cached,reference,atol=1e-6,rtol=1e-5)
        for j,prompt in enumerate(prompts):
            unpadded = torch.tensor([prompt])
            expected = combined_step(models,terms,unpadded,torch.ones_like(unpadded),{})
            torch.testing.assert_close(cached[j],expected[0],atol=1e-6,rtol=1e-5)
        assert len({id(cache) for cache in caches.values()}) == 3
        next_ids = torch.tensor([[3+step],[4+step],[5+step]])
        for j in range(len(prompts)):
            prompts[j].append(int(next_ids[j,0]))
        sequence = torch.cat((sequence,next_ids),1)
        attention = torch.cat((attention,torch.ones((3,1),dtype=torch.long)),1)
        ids = next_ids


def test_q0_deduplicates_exactly_to_s0():
    terms = canonical_terms([("s",1),("t0",1),("t0",-1)])
    assert terms == {"s":1.}
    with pytest.raises(ValueError):
        canonical_terms([("s",float("nan"))])


class Tokenizer:
    eos_token_id=16
    pad_token_id=0
    name_or_path="tiny"
    def __len__(self): return 17
    def decode(self,tokens,**kwargs): return ",".join(map(str,tokens))


def never_stop(ids,scores):
    return torch.zeros(len(ids),dtype=torch.bool)


def test_per_sequence_seed_is_independent_of_other_sequences_and_batch_order():
    models = {"s":tiny(4)}
    prompts,seeds = [[1,3,4],[1,8]], [2027,99]
    together,cost = generate_batch(models,{"s":1},Tokenizer(),prompts,seeds,
        device="cpu",max_new_tokens=8,stopper=never_stop)
    for j in range(2):
        single,_ = generate_batch(models,{"s":1},Tokenizer(),[prompts[j]],[seeds[j]],
            device="cpu",max_new_tokens=8,stopper=never_stop)
        assert single[0]["completion_ids"] == together[j]["completion_ids"]
    assert cost["generated_tokens"] == 16
    assert cost["model_forward_calls"] == 8
    assert cost["transformer_token_positions"] == 6+7*2


def test_external_stop_never_injects_eos_and_inactive_rows_masked():
    model = tiny(4)
    original = model.forward
    attentions = []
    def capture(ids,attention,cache=None):
        attentions.append(attention.clone())
        return original(ids,attention,cache)
    model.forward=capture
    calls=[]
    def stop(ids,scores):
        calls.append(1)
        return torch.tensor([True,len(calls)>=3])
    traces,cost = generate_batch({"s":model},{"s":1},Tokenizer(),[[1,3],[1,2]],[1,2],
        device="cpu",max_new_tokens=8,sample=False,stopper=stop)
    assert [len(t["completion_ids"]) for t in traces] == [1,3]
    assert traces[0]["stop_reason"] in ("closing_fence","eos")
    assert attentions[1][0,-1] == 0 and attentions[1][1,-1] == 1
    assert cost["active_output_head_positions"] == 4
    assert cost["output_head_positions"] == 6
    assert cost["generated_tokens"] == 4


def test_failure_retains_every_partial_output():
    class Broken:
        def forward(self,*args): raise RuntimeError("test failure")
    with pytest.raises(GenerationFailure) as caught:
        generate_batch({"s":Broken()},{"s":1},Tokenizer(),[[1],[1,2]],[1,2],
            device="cpu",max_new_tokens=8,stopper=never_stop)
    traces,cost = caught.value.partial
    assert len(traces) == 2
    assert all(t["error"] and t["stop_reason"] == "generation_error" for t in traces)


def test_failed_verification_not_success_and_keeps_trace():
    row = {"ground_truth":[]}
    trace = dict(completion_ids=[3],stop_reason="generation_error",error="OOM",selected_log_probs=[-1.])
    with patch("cafd.verifier.score_completion_details",return_value=dict(reward=1.,tier="full_pass")), \
         patch("cafd.verifier.extract_contract_program",return_value="program"), \
         patch("cafd.verifier.verify_program",return_value=dict(all_passed=True)):
        result = verify_record(row,trace,Tokenizer())
    assert not result["full_pass"] and result["reward"] == 0
    assert result["completion_ids"] == [3] and result["verifier"]["all_passed"]


def test_policy_resume_limit_and_failed_denominator(tmp_path):
    (tmp_path/"cafd").mkdir()
    (tmp_path/"cafd/verifier.py").write_text("frozen verifier")
    pool = SimpleNamespace(device=torch.device("cpu"),tokenizer=Tokenizer(),load_history=[],models={},
        select=lambda terms: (_ for _ in ()).throw(RuntimeError("missing checkpoint")))
    rows = [dict(id=str(j),problem_family="family",ground_truth=[],prompt="x") for j in range(3)]
    seeds = {str(j):[j,j+100] for j in range(3)}
    decoding = dict(count=2,batch_size=2,max_new_tokens=8)
    with patch("cafd.prompting.prompt_token_ids",return_value=[1,2]), \
         patch("cafd.data.bounded_prompt",return_value="x"):
        first = run_policy(tmp_path,tmp_path/"run","S0",{"s":1},rows,decoding,seeds,"cpu",
                           pool=pool,max_new_records=2)
        assert first["completed_records"] == 2 and first["expected_records"] == 6
        second = run_policy(tmp_path,tmp_path/"run","S0",{"s":1},rows,decoding,seeds,"cpu",pool=pool)
        assert second["completed_records"] == 6 and second["resumed_records"] == 2
        third = run_policy(tmp_path,tmp_path/"run","S0",{"s":1},rows,decoding,seeds,"cpu",pool=pool)
        assert third["resumed_records"] == 6
        with pytest.raises(RuntimeError,match="identity mismatch"):
            run_policy(tmp_path,tmp_path/"run","S0",{"s":1},rows,dict(decoding,max_new_tokens=9),seeds,"cpu",pool=pool)
    persisted = [json.loads(line) for line in (tmp_path/"run/policies/S0/rollouts.jsonl").read_text().splitlines()]
    assert len(persisted) == 6
    assert all(r["status"] == "error" and not r["full_pass"] for r in persisted)
