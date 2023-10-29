#%%

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

from sae.model import SAE
# from sae.buffer import Buffer
from typing import Union, Literal, List, Dict, Tuple, Optional
import torch
torch.set_float32_matmul_precision('high') # When training if float32, torch.compile asked us to add this

import transformer_lens
from circuitsvis.tokens import colored_tokens
from math import ceil
from random import randint
from sae.plotly_utils import hist, scatter
from datasets import load_dataset
from IPython.display import display, HTML
from copy import deepcopy
import argparse
from tqdm.notebook import tqdm
from time import ctime
from dataclasses import dataclass
import einops
import wandb

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

_default_cfg = {
    "seed": 42,  # RNG seed for reproducibility
    "batch_size": 4096,  # Number of samples in each mini-batch
    "seq_len": 128,  # Length of each input sequence for the model
    "d_in": lm.cfg.d_mlp,  # Input dimension for the encoder model
    "d_sae": 16384,  # Dimensionality for the sparse autoencoder (SAE)
    "lr": 2e-4,  # This is because Neel uses L2, and I think we should use mean squared error
    "l1_lambda": 4e-4, # I would have thought this needs be divided by d_in but maybe it's just tiny?!
    "dataset": "c4",  # Name of the dataset to use
    "dataset_args": ["en"],  # Any additional arguments for the dataset
    "dataset_kwargs": {"split": "train", "streaming": True}, 
    # Keyword arguments for dataset. Highly recommend streaming for massive datasets!
    "beta1": 0.9,  # Adam beta1
    "beta2": 0.99,  # Adam beta2
    "buffer_size": 20_000,  # Size of the buffer
    "act_name": "blocks.0.mlp.hook_post",
    "num_tokens": int(2e12), # Number of tokens to train on 
    "wandb_mode": "online", 
    "test_set_size": 128 * 20, # 20 Sequences
    "test_every": 100, #
    "wandb_group": None,
    "reset_sae_neurons_every": 3_000, # Neel uses 30_000 but we want to move a bit faster
    "reset_sae_neurons_cutoff": 1e-6,
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

else:
    cfg = {
        **_default_cfg,
        # "wandb_mode": "offline",  # Offline wandb
    }

assert cfg["batch_size"] % cfg["seq_len"] == 0, (cfg["batch_size"], cfg["seq_len"])

# %%

torch.random.manual_seed(cfg["seed"])
dummy_sae = SAE(d_in=cfg["d_in"], d_sae=cfg["d_sae"]) # For now, just keep same dimension

#%%

# Get some C4 (validation data?) and then see what it looks like

raw_all_data = iter(load_dataset(cfg["dataset"], *(cfg["dataset_args"] or []), **(cfg["dataset_kwargs"] or {})))
# raw_iterable = [data_elem["text"] for data_elem in raw_all_data]

# buffer = Buffer(
#     cfg = cfg,
#     iterable_dataset = raw_iterable,
# )

# for i in range(1000):
#     s=raw_all_data[i]["text"]
#     tokens=lm.to_tokens(s, truncate=False)
#     print(i, tokens.shape)

# %%

sae = deepcopy(dummy_sae)
opt = torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))

#%%

def loss_fn(
    sae_reconstruction,
    sae_hiddens,
    ground_truth,
    l1_lambda,
):
    # Note L1 is summed, L2 is meaned here. Should be the most appropriate when comparing losses across different LM size
    reconstruction_loss = torch.pow((sae_reconstruction-ground_truth), 2).mean(dim=(0, 1)) 
    reconstruction_l1 = torch.abs(sae_reconstruction).sum(dim=1).mean(dim=(0,))
    avg_num_firing = (sae_hiddens > 0).float().sum(dim=1).mean(dim=(0,)) # On a given token, how many neurons fired?
    did_fire = (sae_hiddens > 0).long().sum(dim=0).cpu() # How many times did each neuron fire?

    l1_loss = l1_lambda * reconstruction_l1
    loss = reconstruction_loss + l1_loss
    
    # if loss > 1e3:
    #     assert False

    return {
        "loss": loss,
        "loss_logged": loss.item(),
        "reconstruction_loss_logged": reconstruction_loss.item(),
        "l1_loss_logged": l1_loss.item(),
        "reconstruction_l1_logged": reconstruction_l1.item(),
        "avg_num_firing_logged": avg_num_firing.item(),
        "did_fire_logged": did_fire,
    }

def train_step(
    sae, 
    mini_batch,
    test_data=None,
):
    opt.zero_grad(True)
        
    # TODO add feature, normalizing W_out columns (or rows??) to 1
    # old_wout = model.W_out.clone().detach() 
    # TODO then benchmark how much slower this is
    
    metrics = loss_fn(*sae(mini_batch), ground_truth=mini_batch, l1_lambda=cfg["l1_lambda"])

    metrics["loss"].backward()
    opt.step()

    if test_data is not None:
        test_loss_metrics = loss_fn(*sae(test_data), ground_truth=test_data, l1_lambda=cfg["l1_lambda"])
        test_loss_metrics = {f"test_{k}": v for k, v in test_loss_metrics.items()}
        metrics = {**metrics, **test_loss_metrics}

    return metrics

    # TODO actually apply the normalizing feature here
    # TODO ie remove the gradient jump in the direction of `old_wout`

# TODO torch.compile gives lots of cached recompiles. Fix these

#%%

wandb_notes = ""

try: 
    with open(__file__, "r") as f:
        wandb_notes += f.read()
except Exception as e:
    print("Couldn't read file", __file__, "due to", str(e), "so not adding notes")

run_name = ctime().replace(" ", "_").replace(":", "-") + "_" + randint(1, 100)

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

def get_batch_tokens(
    lm, 
    dataset,
):
    """Warning: uses `cfg` globals and probably more. For profiling, mostly"""

    batch_tokens = torch.LongTensor(size=(0, cfg["seq_len"])).to(lm.cfg.device)
    current_batch = []
    current_length = 0

    while (
        (step_idx > 0 and batch_tokens.numel() < cfg["batch_size"]) 
        or
        (step_idx == 0 and cfg["test_set_size"] > batch_tokens.numel())
    ):
        s = next(dataset)["text"]
        tokens = lm.to_tokens(s, truncate=False, move_to_device=True)
        token_len = tokens.shape[1]

        while token_len > 0:
            # Space left in the current batch
            space_left = cfg["seq_len"] - current_length

            # If the current tokens fit entirely into the remaining space
            if token_len <= space_left:
                current_batch.append(tokens[:, :token_len])
                current_length += token_len
                break
            else:
                # Take as much as will fit
                current_batch.append(tokens[:, :space_left])
                # Remove used part
                tokens = tokens[:, space_left:]
                token_len -= space_left
                current_length = cfg["seq_len"]

            # If a batch is full, concatenate and move to next batch
            if current_length == cfg["seq_len"]:
                full_batch = torch.cat(current_batch, dim=1)
                batch_tokens = torch.cat((batch_tokens, full_batch), dim=0)
                current_batch = []
                current_length = 0
        
        # Ensure we didn't accidentally add too many tokens
        batch_tokens = batch_tokens[:cfg["batch_size"]//cfg["seq_len"]]

    return batch_tokens

#%%

# def my_profiling():
if True: # Usually we don't want to profile, so `if True` is better as it keeps things in scope
    running_frequency_counter = torch.zeros(size=(cfg["d_sae"],)).long().cpu()

    for step_idx in range(
        ceil(cfg["num_tokens"] / (cfg["batch_size"])),
    ):
        # We use Step 0 to make a "test set" 
        # 
        # Firstly get the activations
        # Slowly construct the batch from the iterable dataset
        # TODO implement something like Neel's buffer that will be able to randomize activations much better

        batch_tokens = get_batch_tokens(
            lm=lm,
            dataset=raw_all_data,
        )

        with torch.no_grad():
            with torch.autocast("cuda", torch.bfloat16):    
                mlp_post_acts = lm.run_with_cache(
                    batch_tokens,
                    names_filter=cfg["act_name"],
                )[1][cfg["act_name"]].reshape(-1, cfg["d_in"])

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

            continue

        metrics = train_step(sae, mini_batch=mlp_post_acts, test_data=(test_set if step_idx%cfg["test_every"]==0 else None))
        running_frequency_counter += metrics["did_fire_logged"]

        if step_idx%cfg["test_every"]==0:
            # Also figure out the loss on the test prompts
            # And figure out how frequently all the features fire
            # And then actually resample those neurons

            with torch.no_grad():
                with torch.autocast("cuda", torch.bfloat16):
                    logits = lm.run_with_hooks(
                        test_tokens,
                        fwd_hooks=[(cfg["act_name"], lambda activation, hook: einops.rearrange(sae.forward(einops.rearrange(activation, "batch seq d_in -> (batch seq) d_in"))[0], "(batch seq) d_in -> batch seq d_in", batch=test_tokens.shape[0]))],
                    )
                    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                    correct_logprobs = logprobs[torch.arange(logprobs.shape[0])[:, None], torch.arange(logprobs.shape[1]-1)[None], test_tokens[:, 1:]]
                    test_loss = -correct_logprobs.mean()
                    metrics["test_loss_with_sae"] = test_loss.item()

        if step_idx%cfg["reset_sae_neurons_every"]==0:
            # Save the state dict to weights/
            fname = os.path.expanduser(f'~/sae/weights/{run_name}.pt')
            torch.save(sae.state_dict(), fname)
            # Log the last weights to wandb
            wandb.save(fname) # TODO check whether this saves the whole history

            current_frequency_counter = running_frequency_counter.float() / (cfg["batch_size"] * cfg["test_every"])
            fig = hist(
                [torch.max(current_frequency_counter, torch.FloatTensor([1e-10])).log10().tolist()], # Make things -10 if they never appear
                # Show proportion on y axis
                histnorm="probability",
                title = "Histogram of SAE Neuron Firing Frequency (Proportions of all Neurons)",
                xaxis_title = "Log10(Frequency)",
                yaxis_title = "Proportion of Neurons",
                return_fig = True,
                # Do not show legend
                showlegend = False,
            )
            metrics["frequency_histogram"] = fig
        
            # Reset
            running_frequency_counter = torch.zeros_like(running_frequency_counter)

            # Get indices of neurons with frequency < reset_cutoff
            indices = (current_frequency_counter < cfg["reset_sae_neurons_cutoff"]).nonzero(as_tuple=False)[:, 0]
            metrics["num_neurons_resampled"] = indices.numel()

            if indices.numel() > 0:
                sae.resample_neurons(indices)            

        wandb.log(metrics)

#%%

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

                    # activation = einops.rearrange(sae.run_with_hooks(einops.rearrange(activation, "batch seq d_in -> (batch seq) d_in"))[0], "(batch seq) d_in -> batch seq d_in", batch=test_tokens.shape[0]) 
                    # TODO too tired, finish tomorrow, was going to look at whether any neurons are actually useful

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
    topk=4
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
        print(sae_neuron_firing.mean()) # >0 a lot???

# %%

if ipython is not None:
    # Save the state dict to weights/
    torch.save(sae.state_dict(), os.path.expanduser("~/sae/weights/last_weights.pt"))
    # Log the last weights to wandb
    wandb.save(os.path.expanduser("~/sae/weights/last_weights.pt"))

# %%
