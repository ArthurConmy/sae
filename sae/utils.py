from typing import Union, Literal, List, Dict, Tuple, Optional, Iterable, Callable, Any, Sequence, Set, Deque, DefaultDict, Iterator, Counter, FrozenSet, OrderedDict
import torch
import transformer_lens
from circuitsvis.tokens import colored_tokens
from math import ceil
from random import randint
from jaxtyping import Float, Int, Bool
from sae.plotly_utils import hist, scatter
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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
from pprint import pprint
from torch.utils.data import Dataset
from transformer_lens import utils

def hoyer_square(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Hoyer-Square sparsity measure."""
    eps = torch.finfo(z.dtype).eps

    numer = z.norm(p=1, dim=dim)
    denom = z.norm(p=2, dim=dim)
    return numer.div(denom + eps).square()

@torch.autocast("cuda", torch.bfloat16)
def loss_fn(
    sae_reconstruction,
    sae_hiddens,
    ground_truth,
    cfg,
):
    # Note L1 is summed, L2 is meaned here. Should be the most appropriate when comparing losses across different LM size
    if cfg["l2_loss_form"] == "normalize":
        reconstruction_loss = (torch.pow((sae_reconstruction-ground_truth), 2) / (ground_truth**2).sum(dim=-1, keepdim=True).sqrt()).mean()
    else:
        reconstruction_loss = torch.pow((sae_reconstruction-ground_truth), 2).mean(dim=(0, 1)) 

    if cfg["l1_loss_form"] == "l1":
        sparsity = torch.abs(sae_hiddens).sum(dim=1).mean(dim=(0,)) # 3.4 gradient steps per second. Can get to at least 12.3 : ) 
    elif cfg["l1_loss_form"] == "hoyer":
        sparsity = hoyer_square(sae_hiddens).mean(dim=(0,))
    else:
        raise ValueError(f"Unknown l1_loss_form {cfg['l1_loss_form']}")

    avg_num_firing = (sae_hiddens > 0).to(cfg["dtype"]).sum(dim=1).mean(dim=(0,)) # On a given token, how many neurons fired?
    did_fire = (sae_hiddens > 0).long().sum(dim=0).cpu() # How many times did each neuron fire?

    sparsity_loss = cfg["l1_lambda"] * sparsity
    loss = reconstruction_loss + sparsity_loss
    
    return {
        "loss": loss,
        "loss_logged": loss.item(),
        "reconstruction_loss_logged": reconstruction_loss.item(),
        "sparsity": sparsity.item(),
        "sparsity_loss": sparsity_loss.item(),
        "avg_num_firing_logged": avg_num_firing.item(),
        "did_fire_logged": did_fire,
    }

def train_step(
    sae, 
    opt,
    cfg,
    mini_batch,
    step_idx,
    sched=None, 
    test_data=None,
):
    opt.zero_grad()
    # TODO benchmark whether grad adjustment is expensive here
    
    metrics = loss_fn(*sae(mini_batch), ground_truth=mini_batch, cfg=cfg)
    metrics["loss"].backward()

    # Update grads so that they remove the parallel component
    # (d_sae, d_in) shape
    with torch.no_grad():
        parallel_component = einops.einsum(
            sae.W_dec.grad,
            sae.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        sae.W_dec.grad -= einops.einsum(
            parallel_component,
            sae.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )

    opt.step()
    if sched is not None and (not cfg["sched_finish"] or step_idx < cfg["sched_epochs"] or sched.get_last_lr()[0]+1e-9 < cfg["lr"]): # By default, return low LRs after resampling to normal conditions again
        sched.step()

    metrics["lr"] = opt.param_groups[0]["lr"]
    opt.zero_grad()

    # We move off the sphere! (Note also Adam has momentum-like components) So, we still renormalize to have unit norm 
    with torch.no_grad():
        # Renormalize W_dec to have unit norm
        norms = torch.norm(sae.W_dec.data, dim=1, keepdim=True)

        # # ... These are indeed really close to 1, almost all the time... with norm low
        # metrics = { # Mostly to sanity check that these aren't wildly different to 1!
        #     **metrics,
        #     "W_dec_norms_mean": norms.mean().item(),
        #     "W_dec_norms_std": norms.std().item(),
        #     "W_dec_norms_min": norms.min().item(),
        #     "W_dec_norms_max": norms.max().item(),
        # }
        sae.W_dec.data /= (norms + 1e-6) # Eps for numerical stability I hope

    if test_data is not None:
        test_loss_metrics = loss_fn(*sae(test_data), ground_truth=test_data, cfg=cfg)
        test_loss_metrics = {f"test_{k}": v for k, v in test_loss_metrics.items()}
        metrics = {**metrics, **test_loss_metrics}

    return metrics

    # TODO actually apply the normalizing feature here
    # TODO ie remove the gradient jump in the direction of `old_wout`

# TODO torch.compile gives lots of cached recompiles. Fix these. Is this because we're subclassing HookedRootModule???

def get_batch_tokens(
    lm, 
    dataset: Iterator[Dict], # Dict should contain "text": "This is my batch element..."
    batch_size: int,
    seq_len: int,
    device: Optional[torch.device] = None, # Defaults to `lm.cfg.device`
    use_tqdm: bool = False,
):
    batch_tokens = torch.LongTensor(size=(0, seq_len)).to(device or lm.cfg.device)
    current_batch = []
    current_length = 0

    if use_tqdm:
        pbar = tqdm(total=batch_size, desc="Filling batches")
    
    while batch_tokens.shape[0] < batch_size:
        s = next(dataset)["text"]
        tokens = lm.to_tokens(s, truncate=False, move_to_device=True).squeeze(0)
        assert len(tokens.shape) == 1, f"tokens.shape should be 1D but was {tokens.shape}"
        token_len = tokens.shape[0]
    
        while token_len > 0:
            # Space left in the current batch
            space_left = seq_len - current_length
    
            # If the current tokens fit entirely into the remaining space
            if token_len <= space_left:
                current_batch.append(tokens[:token_len])
                current_length += token_len
                break
    
            else:
                # Take as much as will fit
                current_batch.append(tokens[:space_left])
    
                # Remove used part, add BOS
                tokens = tokens[space_left:]
                tokens = torch.cat((torch.LongTensor([lm.tokenizer.bos_token_id]).to(tokens.device), tokens), dim=0)
    
                token_len -= space_left
                token_len += 1
                current_length = seq_len
    
            # If a batch is full, concatenate and move to next batch
            if current_length == seq_len:
                full_batch = torch.cat(current_batch, dim=0)
                batch_tokens = torch.cat((batch_tokens, full_batch.unsqueeze(0)), dim=0)
                current_batch = []
                current_length = 0

        if use_tqdm:    
            pbar.n = batch_tokens.shape[0]
            pbar.refresh()
    
    return batch_tokens[:batch_size]

@torch.no_grad()
@torch.autocast("cuda", torch.bfloat16)
def get_activations(
    lm,
    batch_tokens, 
    cfg,
    act_name,
    reshape=True,    
):
    activations = lm.run_with_cache(
        batch_tokens,
        names_filter=act_name,
    )[1][act_name]
    
    if reshape:
        return activations.reshape(-1, cfg["d_in"])
    else:
        return activations

def get_neel_model(version = 1):
    """
    Loads the saved autoencoder from HuggingFace.

    Version is expected to be an int, or "run1" or "run2"

    version 25 is the final checkpoint of the first autoencoder run,
    version 47 is the final checkpoint of the second autoencoder run.
    """

    if version==1:
        version = 25
    elif version==2:
        version = 47
    else:
        raise ValueError("Version must be 1 or 2")

    cfg = utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}_cfg.json")
    pprint("Loading this checkpoint:")
    pprint(str(cfg))
    state_dict = utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True)
    return cfg, state_dict

def get_cfg(**kwargs) -> Dict[str, Any]: # TODO remove Any
    cur_dict = {
        "seed": 1, 
        "batch_size": 4,  # Number of samples we pass through THE LM 
        "seq_len": 1024,  # Length of each input sequence for the model
        "d_in": 768,  # Input dimension for the encoder model
        "d_sae": 16384 * 8,  # Dimensionality for the sparse autoencoder (SAE)
        "lr": 0.0012,  # This is low because Neel uses L2, and I think we should use mean squared error
        "l1_lambda": 0.00016,
        "dataset": "Skylion007/openwebtext",  # Name of the dataset to use
        "dataset_args": [],  # Any additional arguments for the dataset
        "dataset_kwargs": {"split": "train", "streaming": True}, 
        # Keyword arguments for dataset. Highly recommend streaming for massive datasets!
        "beta1": 0.9,  # Adam beta1
        "beta2": 0.999,  # Adam beta2
        "act_name": "blocks.8.hook_mlp_out",
        "num_tokens": int(2e12), # Number of tokens to train on 
        "test_set_batch_size": 1,
        "test_set_num_batches": 300,
        "wandb_mode_online_override": False, # Even if in testing, wandb online anyways
        "test_every": 500,
        "save_state_dict_every": lambda step: step%19000 == 1, # Disabled currently; used Mod 1 so this still saves immediately. Plus doesn't interfere with resampling (very often)
        "wandb_group": None,
        "resample_condition": "freq", # Choose nofire or freq
        "resample_sae_neurons_cutoff": lambda step: (1e-6 if step < 25_000 else 1e-7), # Maybe resample fewer later... only used if resample_condition == "nofire"
        "resample_mode": "anthropic", # Either "reinit" or "Anthropic"
        "anthropic_resample_batches": 200_000, # 32_000 // 100, # How many batches to go through when doing Anthropic reinit. Should be >=d_sae so there are always enough. Plus 
        "resample_sae_neurons_every": 205042759847598434752987523487239,
        "resample_sae_neurons_at": [10_000, 20_000] + torch.arange(50_000, 125_000, 25_000).tolist(),
        "dtype": torch.float32, 
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "activation_training_order": "shuffled", # Do we shuffle all MLP activations across all batch and sequence elements (Neel uses a buffer for this), using `"shuffled"`? Or do we order them (`"ordered"`)
        "buffer_size": 2**19, # Size of the buffer
        "buffer_device": "cuda:0", # Size of the buffer
        "testing": False,
        "delete_cache": False, # TODO make this parsed better, likely is just a string
        "sched_type": "cosine_warmup", # "cosine_annealing", # Mark as None if not using 
        "sched_epochs": 50*20, # Think that's right???
        "sched_lr_factor": 0.1, # This seems to help a little. But not THAT much, so tone down
        "sched_warmup_epochs": 50*20,
        "sched_finish": True,
        "resample_factor": 0.2, # 3.4 steps per second
        "bias_resample_factor": 0.0,
        "log_everything": False,
        "anthropic_resample_last": 7500,
        "l1_loss_form": "l1",
        "l2_loss_form": "normalize",
    }

    for k, v in kwargs.items():
        if k not in cur_dict:
            raise ValueError(f"Unknown config key {k}")
        cur_dict[k] = v

    return cur_dict
