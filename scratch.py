#%%

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

from sae.model import SAE
# from sae.buffer import Buffer
from typing import Union, Literal, List, Dict, Tuple, Optional
import torch
torch.set_float32_matmul_precision('high') # When training if float32, torch.compile asked us to add this

import transformer_lens
from math import ceil
from datasets import load_dataset
from copy import deepcopy
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
    "lr": 1e-4,  # This is because Neel uses L2, and I think we should use mean squared error
    "l1_lambda": 3e-3, # I would have thought this needs be divided by d_in but maybe it's just tiny?!
    "dataset": "c4",  # Name of the dataset to use
    "dataset_args": ["en"],  # Any additional arguments for the dataset
    "dataset_kwargs": {"split": "train", "streaming": True}, # Keyword arguments for dataset. Highly recommend streaming for massive datasets!
    "beta1": 0.9,  # Adam beta1
    "beta2": 0.99,  # Adam beta2
    "buffer_size": 20_000,  # Size of the buffer
    "act_name": "blocks.0.mlp.hook_post",
    "num_tokens": int(2e9),  # Number of tokens to train on 
    "wandb_mode": "online", 
    "test_set_size": 100, # 
    "test_every": 100, #
}

# Copy with new args
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

sae = deepcopy(dummy_sae).to(lm.cfg.device)
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
    l1_loss = l1_lambda * torch.abs(sae_reconstruction).sum(dim=1).mean(dim=(0,))
    loss = reconstruction_loss + l1_loss
    
    # if loss > 1e3:
    #     assert False

    return {
        "loss": loss,
        "loss_logged": loss.item(),
        "reconstruction_loss_logged": reconstruction_loss.item(),
        "l1_loss_logged": l1_loss.item(),
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

# Offline wandb
wandb.init(
    project="sae",
    group="c410k",
    job_type="train",
    config=cfg,
    name=ctime().replace(" ", "_").replace(":", "-"),
    mode=cfg["wandb_mode"],
)

#%%

# current_all_data=deepcopy(raw_all_data) # So restarts actually restart

for step_idx in range(
    ceil(cfg["num_tokens"] / (cfg["batch_size"])),
):
    # We use Step 0 to make a "test set" 
    # 
    # Firstly get the activations
    # Slowly construct the batch from the iterable dataset
    # TODO implement something like Neel's buffer that will be able to randomize activations much better

    batch_tokens = torch.LongTensor(size=(0, cfg["seq_len"])).to(lm.cfg.device)
    current_batch = []
    current_length = 0

    while (
        (step_idx > 0 and batch_tokens.numel() < cfg["batch_size"]) 
        or
        (step_idx == 0 and cfg["test_set_size"] > batch_tokens.numel())
    ):
        s = next(raw_all_data)["text"]
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

    metrics = train_step(sae, mini_batch=mlp_post_acts, test_data=(test_set if step_idx%cfg["test_every"]==0 else None))

    if step_idx%cfg["test_every"]==0:
        # Also figure out the loss on the test prompts
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

    wandb.log(metrics)

    # TODO add some test feature

wandb.finish()

# %%

# Test how important all the neurons are ...

with torch.no_grad():
    with torch.autocast("cuda", torch.bfloat16):
        losses = torch.FloatTensor(size=(0,)).to(lm.cfg.device)

        for neuron_idx_ablated in tqdm([None] + list(range(cfg["d_sae"])), desc="Neuron"):
            def my_zero_hook(activation, hook):

                # activation = einops.rearrange(sae.run_with_hooks(einops.rearrange(activation, "batch seq d_in -> (batch seq) d_in"))[0], "(batch seq) d_in -> batch seq d_in", batch=test_tokens.shape[0]) # TODO too tired, finish tomorrow, was going to look at whether any neurons are actually useful

                if neuron_idx_ablated is None:
                    return activation
                else:
                    ablated_activation = activation
                    assert len(ablated_activation.shape)==3, ablated_activation.shape
                    ablated_activation[:, :, neuron_idx_ablated] = 0 # Zero ablated
                    return ablated_activation

            logits = lm.run_with_hooks(
                test_tokens,
                fwd_hooks=[(cfg["act_name"], my_zero_hook)], 
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
