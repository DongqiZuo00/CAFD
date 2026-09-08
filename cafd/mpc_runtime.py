"""MPC-only sampling/scoring; no legacy trainer monkeypatches."""
import copy
import gc
import types
import torch
import torch.nn.functional as F
from transformers import GenerationConfig, LogitsProcessor, StoppingCriteria, StopStringCriteria
from .data import bounded_prompt
from .prompting import encode_prompt, PROMPT_RENDERER_MISTRAL3_INSTRUCT
from .training_common import HiddenCausalLM, make_sequence
from .mistral_runtime import load_model
from .verifier import score_completion_details

class BehaviorRecorder(LogitsProcessor):
    """Actual categorical log probabilities with one B x V buffer."""
    def __init__(self):
        self.previous, self.selected = None, []
    def consume(self, ids):
        if self.previous is not None:
            self.selected.append(self.previous.gather(1, ids[:, -1:]).squeeze(1).detach().cpu())
            self.previous = None
    def __call__(self, input_ids, scores):
        self.consume(input_ids)
        self.previous = scores.detach().float().log_softmax(-1)
        return scores
    def finish(self, sequences):
        self.consume(sequences)
        return torch.stack(self.selected, dim=1).tolist()

class CompletionStop(StoppingCriteria):
    """External closing-fence stopping; never inject an EOS."""
    def __init__(self, tokenizer, prompt_length):
        self.prompt_length, self.eos = prompt_length, tokenizer.eos_token_id
        self.tokenizer = tokenizer
        self.fence = StopStringCriteria(tokenizer, ["\n" + chr(96)*3])
        self.lengths = None
    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[:, self.prompt_length:]
        if self.lengths is None:
            self.lengths = torch.zeros(len(input_ids), dtype=torch.long, device=input_ids.device)
        matched = self.fence(generated, scores)
        for index in matched.nonzero(as_tuple=False).flatten().tolist():
            text = self.tokenizer.decode(generated[index].tolist(), skip_special_tokens=True)
            matched[index] = text.count(chr(96)*3) >= 2
        if self.eos is not None:
            matched = matched | (generated[:, -1] == self.eos)
        self.lengths[matched & (self.lengths == 0)] = generated.shape[1]
        return matched

def _fp32_head_forward(head, hidden):
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return F.linear(hidden.float(), head._mpc_weight, head._mpc_bias)

@torch.no_grad()
def generate(model, tokenizer, row, *, count, max_new_tokens, seed, sample=True, max_prompt_tokens=4096):
    lm = model.causal_lm
    encoded = encode_prompt(tokenizer, bounded_prompt(row), renderer=PROMPT_RENDERER_MISTRAL3_INSTRUCT, return_tensors="pt")
    if encoded["input_ids"].shape[1] > max_prompt_tokens:
        raise RuntimeError("prompt cap exceeded; refusing silent truncation")
    ids = encoded["input_ids"].to(next(lm.parameters()).device)
    attention = encoded["attention_mask"].to(ids.device)
    recorder, stopper = BehaviorRecorder(), CompletionStop(tokenizer, ids.shape[1])
    was_training = lm.training
    head = lm.get_output_embeddings()
    original_forward = head.forward
    config = GenerationConfig(do_sample=sample, temperature=1., top_p=1., top_k=0,
        repetition_penalty=1., num_beams=1, num_return_sequences=count,
        max_new_tokens=max_new_tokens, use_cache=True, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    try:
        lm.eval()
        head._mpc_weight = head.weight.detach().float()
        head._mpc_bias = None if head.bias is None else head.bias.detach().float()
        head.forward = types.MethodType(_fp32_head_forward, head)
        with torch.random.fork_rng(devices=[ids.device.index or 0] if ids.is_cuda else []):
            torch.manual_seed(seed)
            if ids.is_cuda:
                torch.cuda.manual_seed_all(seed)
            sequences = lm.generate(input_ids=ids, attention_mask=attention, generation_config=config,
                                    logits_processor=[recorder], stopping_criteria=[stopper])
        old, records = recorder.finish(sequences), []
        for j,tokens in enumerate(sequences[:, ids.shape[1]:].tolist()):
            length = len(tokens)
            if stopper.lengths is not None and int(stopper.lengths[j]) > 0:
                length = int(stopper.lengths[j])
            tokens = tokens[:length]
            if tokenizer.eos_token_id in tokens:
                tokens = tokens[:tokens.index(tokenizer.eos_token_id)+1]
            details = score_completion_details(tokenizer.decode(tokens, skip_special_tokens=True), row["ground_truth"], "hierarchical")
            records.append(dict(row_id=row["id"], family=row["problem_family"], prompt_ids=ids[0].tolist(),
                completion_ids=tokens, old_log_probs=old[j][:len(tokens)], reward=details["reward"],
                full_pass=details["reward"] == 1., tier=details["tier"],
                length_cap=len(tokens)==max_new_tokens and (stopper.lengths is None or int(stopper.lengths[j])==0),
                source="current_student", seed=seed))
        return records
    finally:
        head.forward = original_forward
        for name in ("_mpc_weight", "_mpc_bias"):
            if hasattr(head, name):
                delattr(head, name)
        lm.train(was_training)

def load_hidden(root,path,device,*,trainable=False):
    return HiddenCausalLM(load_model(str(path),"",cache_dir=root/".cache/huggingface/hub",device=device,trainable=trainable))

def independent_copy(model):
    result = copy.deepcopy(model).eval().requires_grad_(False)
    if any(a.data_ptr()==b.data_ptr() for a,b in zip(model.parameters(),result.parameters())):
        raise RuntimeError("frozen copy shares Student storage")
    return result

def refresh_behavior(behavior,student):
    behavior.load_state_dict(student.state_dict(),strict=True)
    behavior.eval().requires_grad_(False)

class TeacherPool:
    def __init__(self,root,device):
        self.root,self.device,self.models,self.loads = root,device,{},0
    def select(self,paths):
        paths = set(paths)
        for path in list(self.models):
            if path not in paths:
                del self.models[path]
        gc.collect()
        torch.cuda.empty_cache()
        for path in sorted(paths):
            if path not in self.models:
                self.models[path] = load_hidden(self.root,path,self.device)
                self.loads += 1
        return self.models

def sequence_inputs(record,tokenizer,device):
    ids,attention,_,mask = make_sequence(record["prompt_ids"],record["completion_ids"],tokenizer,device)
    return ids,attention,mask

@torch.no_grad()
def frozen_sources(record,tokenizer,reference,teachers,coefficients,device):
    ids,attention,mask = sequence_inputs(record,tokenizer,device)
    sources = []
    for coefficient,model in [(1.,reference)]+[(c,teachers[p]) for p,c in coefficients.items()]:
        hidden = model(ids,attention)[:, :-1][mask]
        head = model.lm_head
        sources.append((coefficient,hidden.detach(),head.weight.detach(),None if head.bias is None else head.bias.detach()))
    return sources

@torch.no_grad()
def target_chunks(sources,chunk=64):
    count = sources[0][1].shape[0]
    for start in range(0,count,chunk):
        end,logits = min(count,start+chunk),None
        for coefficient,hidden,weight,bias in sources:
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                term = F.linear(hidden[start:end].float(),weight.float(),None if bias is None else bias.float())
            logits = coefficient*term if logits is None else logits+coefficient*term
        yield start,end,logits.softmax(-1)

def restore_optimizer_exact(optimizer,saved):
    """Restore original FP32 masters after generic optimizer BF16 casts."""
    optimizer.load_state_dict(saved)
    for old,live in zip(saved["param_groups"],optimizer.param_groups):
        for index,parameter in zip(old["params"],live["params"]):
            if index in saved["state"]:
                optimizer.state[parameter] = {k:v.to(parameter.device,dtype=torch.float32).clone()
                    if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in saved["state"][index].items()}
    optimizer.assert_fp32_states()
