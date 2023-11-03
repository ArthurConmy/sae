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
from typing import Union, Literal, List, Dict, Tuple, Optional, Iterable, Callable, Any, Sequence, Set, Deque, DefaultDict, Iterator, Counter, FrozenSet, OrderedDict
from torch.utils.data import Dataset
from transformer_lens import utils

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
    opt,
    cfg,
    mini_batch,
    sched=None, 
    test_data=None,
):
    opt.zero_grad()
    # TODO benchmark whether grad adjustment is expensive here
    
    metrics = loss_fn(*sae(mini_batch), ground_truth=mini_batch, l1_lambda=cfg["l1_lambda"])
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
    if sched is not None:
        sched.step()
        metrics["lr"] = opt.param_groups[0]["lr"]
    opt.zero_grad()

    # We move off the sphere! (Note also Adam has momentum-like components) So, we still renormalize to have unit norm 
    with torch.no_grad():
        # Renormalize W_dec to have unit norm
        norms = torch.norm(sae.W_dec.data, dim=1, keepdim=True)
        metrics = { # Mostly to sanity check that these aren't wildly different to 1!
            **metrics,
            "W_dec_norms_mean": norms.mean().item(),
            "W_dec_norms_std": norms.std().item(),
            "W_dec_norms_min": norms.min().item(),
            "W_dec_norms_max": norms.max().item(),
        }
        sae.W_dec.data /= (norms + 1e-6) # Eps for numerical stability I hope

    if test_data is not None:
        test_loss_metrics = loss_fn(*sae(test_data), ground_truth=test_data, l1_lambda=cfg["l1_lambda"])
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
    
        pbar.n = batch_tokens.shape[0]
        pbar.refresh()
    
    return batch_tokens

@torch.autocast("cuda", torch.bfloat16)
@torch.no_grad()
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
        "batch_size": 32,  # Number of samples we pass through THE LM
        "seq_len": 128,  # Length of each input sequence for the model
        "d_in": None,  # Input dimension for the encoder model
        "d_sae": 16384,  # Dimensionality for the sparse autoencoder (SAE)
        "lr": 6.5 * 1e-5,  # This is low because Neel uses L2, and I think we should use mean squared error
        "l1_lambda": 3.6 * 1e-4,
        "dataset": "c4",  # Name of the dataset to use
        "dataset_args": ["en"],  # Any additional arguments for the dataset
        "dataset_kwargs": {"split": "train", "streaming": True}, 
        # Keyword arguments for dataset. Highly recommend streaming for massive datasets!
        "beta1": 0.9,  # Adam beta1
        "beta2": 0.99,  # Adam beta2
        "act_name": "blocks.0.mlp.hook_post",
        "num_tokens": int(2e12), # Number of tokens to train on 
        "wandb_mode": "online", 
        "test_set_batch_size": 100, # 20 Sequences
        "test_every": 100,
        "save_state_dict_every": lambda step: step%37123 == 1, # So still saves immediately. Plus doesn't interfere with resampling (very often)
        "wandb_group": None,
        "resample_mode": "reinit", # Either "reinit" or "Anthropic"
        "resample_sae_neurons_every": 30_000, # Neel uses 30_000 but we want to move a bit faster. Plus doing lots of resamples early seems great. NOTE: the [500, 2000] seems crucial for a sudden jump in performance, I don't know why!
        "resample_sae_neurons_at": [],
        "resample_sae_neurons_cutoff": 5.5 * 1e-6, # Maybe resample fewer later...
        "resample_sae_neurons_batches_covered": 10, # How many batches to cover before resampling
        "dtype": torch.float32, 
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "activation_training_order": "shuffled", # Do we shuffle all MLP activations across all batch and sequence elements (Neel uses a buffer for this), using `"shuffled"`? Or do we order them (`"ordered"`)
        "buffer_size": 2**19, # Size of the buffer
        "buffer_device": "cuda:0", # Size of the buffer
        "testing": False,
        "delete_cache": False, # TODO make this parsed better, likely is just a string
    }

    for k, v in kwargs.items():
        if k not in cur_dict:
            raise ValueError(f"Unknown config key {k}")
        cur_dict[k] = v

    return cur_dict
