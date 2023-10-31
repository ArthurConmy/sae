from sae.model import SAE
# from sae.buffer import Buffer
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
    test_data=None,
):
    opt.zero_grad(True)
    # TODO benchmark whether grad adjustment is expensive here
    
    metrics = loss_fn(*sae(mini_batch), ground_truth=mini_batch, l1_lambda=cfg["l1_lambda"])
    metrics["loss"].backward()

    # Update grads so that they remove the parallel component
    # (d_sae, d_in) shape
    with torch.no_grad():
        parallel_component = einops.einsum(
            sae.W_out.grad,
            sae.W_out.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        sae.W_out.grad -= einops.einsum(
            parallel_component,
            sae.W_out.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )

    opt.step()
    opt.zero_grad()

    # We move off the sphere! (Note also Adam has momentum-like components) So, we still renormalize to have unit norm 
    with torch.no_grad():
        # Renormalize W_out to have unit norm
        norms = torch.norm(sae.W_out.data, dim=1, keepdim=True)
        metrics = { # Mostly to sanity check that these aren't wildly different to 1!
            **metrics,
            "W_out_norms_mean": norms.mean().item(),
            "W_out_norms_std": norms.std().item(),
            "W_out_norms_min": norms.min().item(),
            "W_out_norms_max": norms.max().item(),
        }
        sae.W_out.data /= (norms + 1e-6) # Eps for numerical stability I hope

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
