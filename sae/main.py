#%%

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Ignores a warning, unsure if this option is right
os.environ["ACCELERATE_DISABLE_RICH"] = "1"

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("load_ext", "line_profiler")    
    ipython.run_line_magic("autoreload", "2")

from typing import Union, Literal, List, Dict, Tuple, Optional, Iterable, Callable, Any, Sequence, Set, Deque, DefaultDict, Iterator, Counter, FrozenSet, OrderedDict
import torch
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
import transformer_lens
from circuitsvis.tokens import colored_tokens
from math import ceil
from random import randint
from jaxtyping import Float, Int, Bool
from datasets import load_dataset
import subprocess
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
from sae.utils import loss_fn, train_step, get_batch_tokens, get_activations

#%%

lm = transformer_lens.HookedTransformer.from_pretrained("gelu-1l")

#%%

# # Neel's config
# default_cfg = {
#     "seed": 49,
#     "batch_size": 4096,
#     "buffer_mult": 384,
#     "lr": 1e-4,
#     "num_tokens": int(2e9),
#     "l1_coeff": 3e-4,
#     "beta1": 0.9,
#     "beta2": 0.99,
#     "dict_mult": 8,
#     "seq_len": 128,
#     "d_mlp": 2048,
#     "enc_dtype":"fp32",
#     "remove_rare_dir": False,
# }
# cfg = arg_parse_update_cfg(default_cfg)
# cfg["model_batch_size"] = cfg["batch_size"] // cfg["seq_len"] * 16
# cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"]
# cfg["buffer_batches"] = cfg["buffer_size"] // cfg["seq_len"]
# pprint.pprint(cfg)

_default_cfg: Dict[str, Any] = { # TODO remove Any
    "seed": 1,  # RNG seed for reproducibility. Lol I think `1` is a better SAE?
    "batch_size": 32,  # Number of samples we pass through THE LM
    "seq_len": 128,  # Length of each input sequence for the model
    "d_in": lm.cfg.d_mlp,  # Input dimension for the encoder model
    "d_sae": 16384,  # Dimensionality for the sparse autoencoder (SAE)
    "lr": 7 * 1e-5,  # This is because Neel uses L2, and I think we should use mean squared error
    "l1_lambda": 3.6 * 1e-4, # I would have thought this needs be divided by d_in but maybe it's just tiny?!
    "dataset": "c4",  # Name of the dataset to use
    "dataset_args": ["en"],  # Any additional arguments for the dataset
    "dataset_kwargs": {"split": "train", "streaming": True}, 
    # Keyword arguments for dataset. Highly recommend streaming for massive datasets!
    "beta1": 0.9,  # Adam beta1
    "beta2": 0.99,  # Adam beta2
    "act_name": "blocks.0.mlp.hook_post",
    "num_tokens": int(2e12), # Number of tokens to train on 
    "wandb_mode": "online", 
    "test_set_batch_size": 20, # 20 Sequences
    "test_every": 100,
    "save_state_dict_every": lambda step: step%10_000 == 1, # So still saves immediately
    "wandb_group": None,
    "resample_mode": "reinit", # Either "reinit" or "Anthropic"
    "resample_sae_neurons_every": lambda step: step%10_000 == 0 or step in [500, 2000], # Neel uses 30_000 but we want to move a bit faster. Plus doing lots of resamples early seems great. NOTE: the [500, 2000] seems crucial for a sudden jump in performance, I don't know why!
    "resample_sae_neurons_cutoff": 1e-5, # Maybe resample fewer later...
    "resample_sae_neurons_batches_covered": 10, # How many batches to cover before resampling
    "dtype": torch.float32, 
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "activation_training_order": "shuffled", # Do we shuffle all MLP activations across all batch and sequence elements (Neel uses a buffer for this), using `"shuffled"`? Or do we order them (`"ordered"`)
    "buffer_size": 2**21, # Size of the buffer
    "buffer_device": "cuda:0", # Size of the buffer
    "testing": False,
}

if ipython is None:
    # Parse all arguments
    parser = argparse.ArgumentParser()
    for k, v in _default_cfg.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    cfg = {**_default_cfg}
    for k, v in vars(args).items():
        cfg[k] = v
    if cfg["testing"]:
        print("!!! !!! Are you sure you are correctly launching this CLI run??? It has testing enabled")

else:
    cfg = {
        **_default_cfg,
        # "wandb_mode": "offline",  # Offline wandb
    }

if (cfg["activation_training_order"] == "ordered") != (cfg["buffer_size"] is None):
    raise ValueError(f'We train activations shuffled iff we have a buffer')

sae_batch_size = cfg["batch_size"] * cfg["seq_len"]

if cfg["testing"]:
    cfg["buffer_size"] = 2**17

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

run_name = ctime().replace(" ", "_").replace(":", "-") + "_" + str(randint(1, 100))

wandb.init(
    project="sae",
    group=cfg["wandb_group"],
    job_type="train",
    config=cfg,
    name=run_name,
    mode=cfg["wandb_mode"],
    notes=wandb_notes,
)   

#%%

sae = deepcopy(dummy_sae) # Reinitialize `sae`
opt = torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))

#%%

start_time = time.time()

# def my_profiling():
if True: # Usually we don't want to profile, so `if True` is better as it keeps things in scope

    if cfg["activation_training_order"] == "shuffled":
        buffer: Float[torch.Tensor, "sae_batch d_in"] = torch.FloatTensor(size=(0, cfg["d_in"])).float().to(cfg["buffer_device"])
        # Hopefully ~800 million elements doesn't blow up CPU

    running_frequency_counter = torch.zeros(size=(cfg["d_sae"],)).long().cpu()
    last_test_every = -1
    prelosses = []
    
    step_iterator = range(
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

        else:
            assert cfg["activation_training_order"] == "shuffled", cfg["activation_training_order"]

            if buffer.shape[0] < cfg["buffer_size"] // 2:
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
            if buffer.shape[0] < cfg["buffer_size"] // 2:
                print(time.time()-start_time)
                if cfg["testing"] and step_idx > 100:
                    raise Exception("We have finished iterating")
                    # return

                # We need to refill the buffer
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

        metrics = train_step(sae=sae, opt=opt, cfg=cfg, mini_batch=mlp_post_acts, test_data=(test_set if step_idx%cfg["test_every"]==0 else None))
        running_frequency_counter += metrics["did_fire_logged"]

        if step_idx%cfg["test_every"]==0:
            # Figure out the loss on the test prompts
            metrics["test_loss_with_sae"] = sae.get_test_loss(lm=lm, test_tokens=test_tokens).item()
            metrics["loss_recovered"] = (prelosses[1]-metrics["test_loss_with_sae"])/(prelosses[1]-prelosses[0])
        
        if cfg["save_state_dict_every"](step_idx):
            # First save the state dict to weights/
            fname = os.path.expanduser(f'~/sae/weights/{run_name}.pt')
            torch.save(sae.state_dict(), fname)
            
            # Clear out, first?
            subprocess.run("rm -rf /root/.cache/wandb/artifacts/**", shell=True) # TODO still seems to break later syncing, sad, fix this

            # Log the last weights to wandb
            # Save as wandb artifact
            artifact = wandb.Artifact(
                name=f"weights_{run_name}",
                type="weights",
                description="Weights for SAE",
            )
            artifact.add_file(fname)
            wandb.log_artifact(artifact)

        if cfg["resample_sae_neurons_every"](step_idx):
            # And figure out how frequently all the features fire
            # And then actually resample those neurons

            # Now compute dead neurons
            current_frequency_counter = running_frequency_counter.float() / (sae_batch_size * (step_idx - last_test_every))
            last_test_every = step_idx

            fig = hist(
                [torch.max(current_frequency_counter, torch.FloatTensor([1e-10])).log10().tolist()], # Make things -10 if they never appear
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
        
            # Reset the counter
            running_frequency_counter = torch.zeros_like(running_frequency_counter)

            # Get indices of neurons with frequency < reset_cutoff
            indices = (current_frequency_counter < cfg["resample_sae_neurons_cutoff"]).nonzero(as_tuple=False)[:, 0]
            metrics["num_neurons_resampled"] = indices.numel()

            resample_sae_loss_increases = torch.FloatTensor(size=(0,)).float()
            resample_mlp_post_acts = torch.FloatTensor(size=(0, cfg["d_in"])).float().cpu() # Hopefully ~800 million elements doesn't blow up CPU 

            # Sample tons more data and save the directions here

            if cfg["resample_mode"] == "anthropic":
                # TODO this is a bit of a mess
                for resample_batch_idx in range(cfg["resample_sae_neurons_batches_covered"]):
                    resample_batch_tokens = get_batch_tokens(
                        lm=lm,
                        dataset=raw_all_data,
                        seq_len=cfg["seq_len"],
                        batch_size=cfg["batch_size"],
                    )
                    sae_loss = sae.get_test_loss(lm=lm, test_tokens=resample_batch_tokens)

                    with torch.no_grad():
                        # TODO fold this into the forward pass in get_test_loss, we don't need two forward passes
                        with torch.autocast("cuda", torch.bfloat16):    
                            model_logits, activation_cache = lm.run_with_cache(
                                resample_batch_tokens,
                                names_filter=cfg["act_name"],
                            )
                            mlp_post_acts = activation_cache[cfg["act_name"]].reshape(-1, cfg["d_in"])
                            model_logprobs = torch.nn.functional.log_softmax(model_logits, dim=-1)
                            model_loss = -model_logprobs[torch.arange(model_logprobs.shape[0])[:, None], torch.arange(model_logprobs.shape[1]-1)[None], resample_batch_tokens[:, 1:]].flatten()

                            resample_sae_loss_increases = torch.cat((resample_sae_loss_increases, (sae_loss - model_loss).cpu()), dim=0)
                            resample_mlp_post_acts = torch.cat((resample_mlp_post_acts, mlp_post_acts.cpu()), dim=0)

            if indices.numel() > 0:
                # If there are any neurons we should resample, do this
                sae.resample_neurons(indices, opt, resample_sae_loss_increases, resample_mlp_post_acts, mode=cfg["resample_mode"])

        metrics["step_idx"] = step_idx
        wandb.log(metrics)

#%%

# Warning: random crap 

# Save the state dict to weights/
torch.save(sae.state_dict(), os.path.expanduser("~/sae/weights/last_weights.pt"))
# Log the last weights to wandb
wandb.save(os.path.expanduser("~/sae/weights/last_weights.pt"))
wandb.finish()

# %%

# Test how important all the neurons are ...
if ipython is not None:
    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            losses = torch.FloatTensor(size=(0,)).to(lm.cfg.device)

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

# %%

if ipython is not None:
    # Now actually study the top neuron
    topk=5
    top_neuron_idx = losses.topk(topk)[1][topk-1].item()

    with torch.no_grad():
        # Visualize where this damn thing fires

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

# %%

if ipython is not None:
    # Save the state dict to weights/
    torch.save(sae.state_dict(), os.path.expanduser("~/sae/weights/last_weights.pt"))
    # Log the last weights to wandb
    wandb.save(os.path.expanduser("~/sae/weights/last_weights.pt"))

#%%

# Load back artifact # TODO test why this seems so broken...
run_id = "4fl2qtec"
run = wandb.Api().run(f"sae/{run_id}")
# artifact = run.use_artifact(artifact_name)

# %%

l = list(run.logged_artifacts())
latest_artifact = l[-2]
latest_artifact.download(root=os.path.expanduser("~/sae/weights/wandb"))

#%%

