#%%

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Ignores a warning, unsure if this option is right
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache" # Cache transformers

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("load_ext", "line_profiler")    
    ipython.run_line_magic("autoreload", "2")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoTokenizer
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
assert os.path.exists("/workspace/sae/weights") # For weight saving

from typing import Union, Literal, List, Dict, Tuple, Optional, Iterable, Callable, Any, Sequence, Set, Deque, DefaultDict, Iterator, Counter, FrozenSet, OrderedDict
import transformer_lens
from circuitsvis.tokens import colored_tokens
from math import ceil
import numpy as np
import pandas as pd
from random import randint
from jaxtyping import Float, Int, Bool
from datasets import load_dataset
from transformer_lens import utils
import subprocess
import ast
import time
from IPython.display import display, HTML
import gc
from copy import deepcopy
import argparse
from tqdm.notebook import tqdm
from time import ctime
from dataclasses import dataclass
import einops
import wandb

from sae.model import SAE
from sae.plotly_utils import hist, scatter
from sae.utils import loss_fn, train_step, get_batch_tokens, get_activations, get_neel_model, get_cfg

#%%


#%%

cfg = get_cfg()

if ipython is None:
    # Parse all arguments
    parser = argparse.ArgumentParser()
    for k, v in cfg.items():
        parser.add_argument(f"--{k}", type=type(v) if type(v)!=list else str, default=v) # Sigh lists so cursed here
    args = parser.parse_args()
    for k, v in vars(args).items():
        if k in cfg and type(cfg[k]) == list and type(v) != list:
            cfg[k] = ast.literal_eval(v)
        else:
            cfg[k] = v
    if cfg["testing"]:
        print("!!! !!! Are you sure you are correctly launching this CLI run??? It has testing enabled")

else:
    cfg = {
        "testing": True,
        **cfg, # Manual notebook overrides here
        # "wandb_mode_online_override": True,  # Offline wandb
    }

if cfg["testing"]:
    assert cfg["activation_training_order"] == "shuffled", cfg["activation_training_order"]
    cfg["buffer_size"] = 2**17

if (cfg["activation_training_order"] == "ordered") != (cfg["buffer_size"] is None):
    raise ValueError(f'We train activations shuffled iff we have a buffer')

sae_batch_size = cfg["batch_size"] * cfg["seq_len"]

if cfg["buffer_size"] is not None:
    assert sae_batch_size <= cfg["buffer_size"], f"SAE batch size {sae_batch_size} is larger than buffer size {cfg['buffer_size']}"

if (cfg["resample_mode"] == "anthropic") != (cfg["anthropic_resample_batches"] is not None):
    raise ValueError(f'We resample anthropically iff we have anthropic_resample_batches')

if cfg["resample_mode"] == "anthropic":
    assert cfg["anthropic_resample_batches"] % cfg["batch_size"] == 0, f"anthropic_resample_batches {cfg['anthropic_resample_batches']} must be divisible by batch_size {cfg['batch_size']}"

    assert cfg["anthropic_resample_batches"] >= cfg["d_sae"], f"anthropic_resample_batches {cfg['anthropic_resample_batches']} must be greater than d_sae {cfg['d_sae']}"

assert cfg["sched_type"].lower() in ["none", "cosine_warmup"], f"Unknown sched_type {cfg['sched_type']}"

def resample_at_step_idx(step_idx) -> bool:
    return step_idx in cfg["resample_sae_neurons_at"] or (cfg["resample_sae_neurons_every"] is not None and step_idx % cfg["resample_sae_neurons_every"] == 0)

lm = transformer_lens.HookedTransformer.from_pretrained("gelu-1l").to(cfg["dtype"])
lm.tokenizer = AutoTokenizer.from_pretrained("ArthurConmy/alternative-neel-tokenizer")
assert lm.cfg.d_mlp == cfg["d_in"], f"lm.cfg.d_mlp {lm.cfg.d_mlp} != cfg['d_in'] {cfg['d_in']}"

# %%

torch.random.manual_seed(cfg["seed"])
dummy_sae = SAE(cfg) # For now, just keep same dimension

#%%

# Get some C4 or other data from HF
raw_all_data = iter(load_dataset(cfg["dataset"], *(cfg["dataset_args"] or []), **(cfg["dataset_kwargs"] or {})))

#%%

wandb_notes = ""

try:
    with open(__file__, "r") as f:
        wandb_notes += f.read()
except Exception as e:
    print("Couldn't read file", __file__, "due to", str(e), "so not adding notes")

run_name = f'LR-{cfg["lr"]}-LAMBDA-{cfg["l1_lambda"]}-DSAW-{cfg["d_sae"]}{ctime().replace(" ", "_").replace(":", "-") + "_" + str(randint(1, 100))}'

if True:
    wandb.init(
        project="sae",
        group=cfg["wandb_group"],
        job_type="train",
        config=cfg,
        name=run_name,
        mode="offline" if cfg["testing"] or cfg["wandb_mode_online_override"] else "online",
        notes=wandb_notes,
    )   

#%%

sae = deepcopy(dummy_sae) # Reinitialize `sae`
opt = torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
sched = None if cfg["sched_type"].lower()=="none" else torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["sched_epochs"], eta_min=cfg["lr"]*cfg["sched_lr_factor"])
if sched is not None: 
    for _ in range(cfg["sched_warmup_epochs"]):
        sched.step()

#%%

start_time = time.time()
save_weights_failed = False

# def my_profiling():
if True: # Usually we don't want to profile, so `if True` is better as it keeps things in scope
    if cfg["activation_training_order"] == "shuffled":
        buffer: Float[torch.Tensor, "sae_batch d_in"] = torch.FloatTensor(size=(0, cfg["d_in"])).to(cfg["dtype"]).to(cfg["buffer_device"])
        # Hopefully ~800 million elements doesn't blow up CPU

    running_frequency_counter = torch.zeros(size=(cfg["d_sae"],)).long().cpu()

    # Check that our Anthropic resampling is even possible
    if cfg["resample_mode"] == "anthropic":        
        resample_indices = [idx for idx in range(10_000_000) if resample_at_step_idx(idx)]
        resample_index_gaps = [resample_indices[i+1] - resample_indices[i] for i in range(len(resample_indices)-1)]
        assert all([resample_index_gap >= cfg["anthropic_resample_last"] for resample_index_gap in resample_index_gaps]), f"Anthropic resampling is not possible with {cfg['anthropic_resample_last']}"
        # Hmm this may be annoying when we want to resample quickly at the start, but not then later

    last_test_every = -1
    prelosses = []
    
    step_iterator = range(
        # 2,
        ceil(cfg["num_tokens"] / sae_batch_size),
    )
    if cfg["testing"]:
        step_iterator = tqdm(step_iterator, desc="Step")

    for step_idx in step_iterator:
        # Note that we use Step 0 to make a "test set" 

        # First handle getting tokens
        if cfg["activation_training_order"] == "ordered" or step_idx == 0:
            batch_tokens = get_batch_tokens(
                lm=lm,
                dataset=raw_all_data,
                seq_len=cfg["seq_len"],
                batch_size=cfg["batch_size"] if step_idx > 0 else cfg["test_set_batch_size"],
            )

            refill_buffer = False

        else:
            assert cfg["activation_training_order"] == "shuffled", cfg["activation_training_order"]
            
            refill_buffer = buffer.shape[0] <= sae_batch_size or resample_at_step_idx(step_idx)

            if refill_buffer: # Use whole buffer for our new resampling technique
                # If we have less than half the buffer size, 
                # Generate enough new tokens to fill the buffer
                batch_tokens = get_batch_tokens(
                    lm=lm,
                    dataset=raw_all_data,
                    seq_len=cfg["seq_len"],
                    batch_size=ceil((cfg["buffer_size"] - buffer.shape[0]) // cfg["seq_len"]),
                    use_tqdm = cfg["buffer_size"] > 2**20,
                )

        # Then handle getting activations (possibly moving to correct device)
        if cfg["activation_training_order"] == "ordered" or step_idx == 0:
            mlp_post_acts = get_activations(
                lm=lm,
                cfg=cfg,
                batch_tokens=batch_tokens,
                act_name=cfg["act_name"],
            )

        else:
            if refill_buffer:
                print("Refilling buffer; time elapsed", time.time()-start_time, "...")
                if cfg["testing"] and step_idx > 100:
                    raise Exception("We have finished iterating")
                    # return

                # We need to refill the buffer
                # We do a similar thing when Anthropic resampling
                refill_iterator = range(0, batch_tokens.shape[0], cfg["batch_size"])
                total_size = len(refill_iterator) * cfg["batch_size"] * cfg["seq_len"] + buffer.shape[0]
                if cfg["testing"]:
                    refill_iterator = tqdm(refill_iterator, desc="Refill")

                # Initialize empty tensor buffer of the maximum required size
                new_buffer = torch.zeros(total_size, *buffer.shape[1:], dtype=buffer.dtype, device=cfg["buffer_device"])

                # Fill existing buffer
                new_buffer[:buffer.shape[0]] = buffer

                # Insert activations directly into pre-allocated buffer
                insert_idx = buffer.shape[0]
                for refill_batch_idx_start in refill_iterator:
                    refill_batch_tokens = batch_tokens[refill_batch_idx_start:refill_batch_idx_start + cfg["batch_size"]]
                    new_buffer[insert_idx:insert_idx + refill_batch_tokens.numel()] = get_activations(
                        lm=lm,
                        cfg=cfg,
                        batch_tokens=refill_batch_tokens,
                        act_name=cfg["act_name"],
                    ).to(cfg["buffer_device"])
                    insert_idx += refill_batch_tokens.numel()

                # Truncate unused space in new_buffer
                buffer = new_buffer[:insert_idx]

                # Shuffle buffer
                buffer = buffer[torch.randperm(buffer.shape[0])]
                print("... done.")

            # Pop off the end of the buffer
            mlp_post_acts = buffer[-sae_batch_size:].to(cfg["device"])
            buffer = buffer[:-sae_batch_size]

            if cfg["testing"]:
                print("After: ", buffer.shape, mlp_post_acts.shape)

        if step_idx == 0:
            test_tokens = batch_tokens
            test_set = mlp_post_acts

            # Find the loss normally, and when zero ablating activation
            for factor in [1.0, 0.0]:
                logits = lm.run_with_hooks(
                    test_tokens,
                    fwd_hooks=[(cfg["act_name"], lambda activation, hook: factor*activation)],
                )
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                correct_logprobs = logprobs[torch.arange(logprobs.shape[0])[:, None], torch.arange(logprobs.shape[1]-1)[None], test_tokens[:, 1:]]
                loss = -correct_logprobs.mean()
                wandb.log({"prelosses": loss.item()})
                prelosses.append(loss.item())

            continue

        metrics = train_step(sae=sae, opt=opt, sched=sched, step_idx=step_idx, cfg=cfg, mini_batch=mlp_post_acts, test_data=(test_set if step_idx%cfg["test_every"]==0 else None))
        running_frequency_counter += metrics["did_fire_logged"]

        if step_idx%cfg["test_every"]==0:

            gc.collect()
            torch.cuda.empty_cache()

            # Figure out the loss on the test prompts
            metrics["test_loss_with_sae"] = sae.get_test_loss(lm=lm, test_tokens=test_tokens).item()
            metrics["loss_recovered"] = (prelosses[1]-metrics["test_loss_with_sae"])/(prelosses[1]-prelosses[0])

            gc.collect()
            torch.cuda.empty_cache()


        if cfg["save_state_dict_every"](step_idx) or save_weights_failed:
            # Remove file
            # Kinda sketch...
            if cfg["delete_cache"]:
                try:
                    # These seem to be working badly; use https://docs.wandb.ai/ref/cli/wandb-artifact/wandb-artifact-cache/wandb-artifact-cache-cleanup ???
                    subprocess.run("rm -rf /root/.cache/wandb/artifacts/**", shell=True) # Using delete_cache=True for just one process fixes this right?
                    subprocess.run("rm -rf /workspace/sae/weights/**.pt", shell=True)
                    subprocess.run("rm -rf /root/.local/share/wandb/artifacts/staging/**", shell=True)
                except Exception as e:
                    print("Couldn't cache clear: " + str(e))

            # Then save the state dict to weights/
            try:
                fname = f'/workspace/sae/weights/{run_name}.pt'
                torch.save(sae.state_dict(), fname)
                
                # Log the last weights to wandb
                # Save as wandb artifact
                artifact = wandb.Artifact(
                    name=f"weights_{run_name}",
                    type="weights",
                    description="Weights for SAE",
                )
                artifact.add_file(fname)
                wandb.log_artifact(artifact)
                artifact.wait()
            except Exception as e:
                print("!!! warning, could not save weights" + str(e))
            else:
                save_weights_failed=False
        
        if resample_at_step_idx(step_idx):
            # And figure out how frequently all the features fire
            # And then actually resample those neurons

            # Now compute dead neurons
            current_frequency_counter = running_frequency_counter.to(cfg["dtype"]) / (sae_batch_size * (step_idx - last_test_every)) # TODO implement exactly the Anthropic thing

            if cfg["resample_mode"] == "anthropic":
                # Get things that didn't fire
                indices = (running_frequency_counter == 0).nonzero(as_tuple=False)[:, 0]
            elif cfg["resample_mode"] == "reinit":        
                # Get indices of neurons with frequency < reset_cutoff
                indices = (current_frequency_counter < cfg["resample_sae_neurons_cutoff"]).nonzero(as_tuple=False)[:, 0]
            else:
                raise ValueError(f"Unknown resample_mode {cfg['resample_mode']}")

            last_test_every = step_idx
            # Reset the counter
            running_frequency_counter = torch.zeros_like(running_frequency_counter)

            fig = hist(
                [torch.max(current_frequency_counter, torch.FloatTensor([1e-10]).to(current_frequency_counter.dtype)).log10().tolist()], # Make things -10 if they never appear
                # Show proportion on y axis
                histnorm="percent",
                title = "Histogram of SAE Neuron Firing Frequency (Proportions of all Neurons)",
                xaxis_title = "Log10(Frequency)",
                yaxis_title = "Percent (changed!) of Neurons",
                return_fig = True,
                # Do not show legend
                showlegend = False,
            )
            metrics["frequency_histogram"] = fig
        
            if indices.numel() > 0:
                # If there are any neurons we should resample, do it
                sae.resample_neurons(indices=indices, sched=sched, dataset=raw_all_data, opt=opt, sae=sae, lm=lm, metrics=metrics)

            metrics["num_neurons_resampled"] = indices.numel()

        if resample_at_step_idx(step_idx + cfg["anthropic_resample_last"]):
            running_frequency_counter = torch.zeros_like(running_frequency_counter)

        metrics["step_idx"] = step_idx
        wandb.log(metrics if len(metrics) != 9 or step_idx % 20 == 0 or step_idx < 2000 else {})
        if step_idx == 7 and len(metrics) != 9: # Something random
            print("*"*100)
            print(f"NOOOOOOOOOOOOOOOOOOOOOOOOOOO! You messed this up, len(metrics) changed to {len(metrics)}")
            print("*"*100)
            wandb.log({"L": 1})

#%%

neel_sae = deepcopy(sae)
neel_sae.load_from_neels(1)

# %%
    
# Load in my 300 L0 / 80% Loss Recovered
sae.load_from_my_wandb(
    run_id = "794ulknc",
    # index_from_back_override=3, # 1 did not work
    index_from_front_override=2,
)

#%%

# Test how important all the neurons are ...
# Note: this is likely a pretty high variance estimate

if ipython is not None:
    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            losses = torch.FloatTensor(size=(0,)).to(lm.cfg.device)
            my_cache = {}
            for neuron_idx_ablated in tqdm([None] + list(range(cfg["d_sae"])), desc="Neuron"):
                def ablation_hook(activation, hook, type: Literal["zero", "mean"] = "mean"):
                    if neuron_idx_ablated is None:
                        return activation

                    else:
                        ablated_activation = activation

                        if len(ablated_activation.shape)==2:
                            ablated_activation[:, neuron_idx_ablated] -= (ablated_activation[:, neuron_idx_ablated].mean(dim=(0,), keepdim=True) if type=="mean" else ablated_activation[:, neuron_idx_ablated])

                        elif len(ablated_activation.shape)==3:    
                            ablated_activation[:, :, neuron_idx_ablated] -= (ablated_activation[:, :, neuron_idx_ablated].mean(dim=(0, 1), keepdim=True) if type=="mean" else ablated_activation[:, :, neuron_idx_ablated])

                        else:
                            raise ValueError(f"Unexpected shape {ablated_activation.shape}")

                        return ablated_activation

                logits = lm.run_with_hooks(
                    test_tokens,
                    fwd_hooks=[(cfg["act_name"], lambda activation, hook: sae.run_with_hooks(activation, fwd_hooks=[("hook_hidden_post", ablation_hook)])[0])],
                )

                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                correct_logprobs = logprobs[torch.arange(logprobs.shape[0])[:, None], torch.arange(logprobs.shape[1]-1)[None], test_tokens[:, 1:]]
                loss = -correct_logprobs.mean()
                if neuron_idx_ablated is None:
                    normal_loss = loss.item()
                else:
                    losses = torch.cat((losses, torch.FloatTensor([loss.item()]).to(lm.cfg.device)), dim=0)
                    if neuron_idx_ablated % 100 == 99:
                        print(losses.topk(10))

#%%

# All adapted from Neel's colab

SPACE = "·"
NEWLINE="↩"
TAB = "→"

def process_token(s: Any):
    if isinstance(s, torch.Tensor):
        s = s.item()
    if isinstance(s, np.int64):
        s = s.item()
    if isinstance(s, int):
        s = lm.to_string(s)
    s = s.replace(" ", SPACE)
    s = s.replace("\n", NEWLINE+"\n")
    s = s.replace("\t", TAB)
    return s

def process_tokens(l):
    if isinstance(l, str):
        l = lm.to_str_tokens(l)
    elif isinstance(l, torch.Tensor) and len(l.shape)>1:
        l = l.squeeze(0)
    return [process_token(s) for s in l]

def list_flatten(nested_list):
    return [x for y in nested_list for x in y]

def make_token_df(tokens, len_prefix=5, len_suffix=1):
    str_tokens = [process_tokens(lm.to_str_tokens(t)) for t in tokens]
    unique_token = [[f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens]

    context = []
    batch = []
    pos = []
    label = []
    for b in range(tokens.shape[0]):
        # context.append([])
        # batch.append([])
        # pos.append([])
        # label.append([])
        for p in range(tokens.shape[1]):
            prefix = "".join(str_tokens[b][max(0, p-len_prefix):p])
            if p==tokens.shape[1]-1:
                suffix = ""
            else:
                suffix = "".join(str_tokens[b][p+1:min(tokens.shape[1]-1, p+1+len_suffix)])
            current = str_tokens[b][p]
            context.append(f"{prefix}|{current}|{suffix}")
            batch.append(b)
            pos.append(p)
            label.append(f"{b}/{p}")
    # print(len(batch), len(pos), len(context), len(label))
    return pd.DataFrame(dict(
        str_tokens=list_flatten(str_tokens),
        unique_token=list_flatten(unique_token),
        context=context,
        batch=batch,
        pos=pos,
        label=label,
    ))

# %%

hidden_acts = {}

def hidden_act_hook(activation, hook):
    hidden_acts[hook.name] = activation.cpu()
    return activation

with torch.no_grad():
    with torch.autocast("cuda", torch.bfloat16):
        lm.reset_hooks()
        sae.reset_hooks()
        logits = lm.run_with_hooks(
            batch_tokens[:500],
            fwd_hooks=[(cfg["act_name"], lambda activation, hook: sae.run_with_hooks(activation, fwd_hooks=[("hook_hidden_post", hidden_act_hook)])[0])],
        )

MODE = "neel" # "old"

if ipython is not None:
    # Now actually study the top neuron
    topk=4
    top_neuron_idx = losses.topk(topk+1)[1][-1].item()

    for top_neuron_idx in [top_neuron_idx]: # range(-100, -200, -12): # = 7 #,  2395, 14983, 14163,
        if MODE == "neel":
            prop=hidden_acts["hook_hidden_post"][:, :, top_neuron_idx].gt(0).float().mean().item()
            if prop < 0.000000001: continue 
            print("Neuron", top_neuron_idx, "fired on avg", hidden_acts["hook_hidden_post"][:, :, top_neuron_idx].mean().item(), "and proportion", prop)
            token_df = make_token_df(batch_tokens[:500])
            token_df["feature"] = utils.to_numpy(hidden_acts["hook_hidden_post"][:, :, top_neuron_idx].flatten())
            token_df = token_df.sort_values("feature", ascending=False).head(20).style.background_gradient("coolwarm")
            display(token_df)

        elif MODE == "old":
            with torch.no_grad():
                # Visualize where this damn thing fires on some RANODMLY sampled sequences
                reshaped_test_acts = einops.rearrange(test_set, "(batch seq) d_in -> batch seq d_in", batch=test_tokens.shape[0])

                pre_firing_amount = sae.run_with_cache( # TODO if this ever blows up GPU, make custom hook that just saves relevant neuron
                    reshaped_test_acts,
                    names_filter = "hook_hidden_pre",
                )[1]["hook_hidden_pre"]
                assert len(pre_firing_amount.shape) == 3, pre_firing_amount.shape
                sae_neuron_firing = pre_firing_amount[:, :, top_neuron_idx]
                
                for seq_idx in range(10):
                    display(
                            colored_tokens(
                                lm.to_str_tokens(
                                    test_tokens[seq_idx],
                                ),
                                sae_neuron_firing[seq_idx],
                            )
                    )
                print("Neuron", top_neuron_idx, "fired", sae_neuron_firing.sum().item(), "times")
                print(sae_neuron_firing.mean()) # >0 a lot???

#%%

feature_id = 2395
token_df = make_token_df(test_tokens)
token_df["feature"] = utils.to_numpy(hidden_acts["hook_hidden_post"][:, :, feature_id].flatten())
token_df.sort_values("feature", ascending=False).head(20).style.background_gradient("coolwarm")

# %%
