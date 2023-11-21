#%%

import IPython
ipython = IPython.get_ipython()
if ipython is not None:
    ipython.magic("%load_ext autoreload")
    ipython.magic("%autoreload 2")

import plotly.express as px
import plotly.graph_objects as go
import wandb
import torch
from sae.utils import loss_fn, train_step, get_batch_tokens, get_activations, get_neel_model, get_cfg


# Initialize the wandb API
api = wandb.Api()

# Define the project and the substring to search for
project_name = "sae"
search_substrings = ["Nov__9_02-56", "Nov__9_03"] # These were from the mini sweep

# Get all runs in the specified project
project_runs = api.runs(f"ArthurConmy/{project_name}")

#%%

# Filter runs by name containing the specified substring
filtered_runs = [run for run in project_runs if any([search_substring in run.name for search_substring in search_substrings])]

# Now you have a list of runs that have "Nov__9" in their name
for run in filtered_runs:
    print(f"Run ID: {run.id}, Name: {run.name}")

#%%

import transformer_lens
import sae # my library
from sae import SAE

first_run = filtered_runs[0]

cfg = first_run.config # Yeeeeah
if isinstance(cfg["dtype"], str):
    cfg["dtype"] = eval(cfg["dtype"])

import torch
lm = transformer_lens.HookedTransformer.from_pretrained("gelu-1l") 

# .to(first_run.config["dtype"]) # Lol be careful

#%%

from pathlib import Path
curpath = Path(__file__)
assert str(curpath).endswith("sae/wandb_to_hf.py"), f"{str(curpath)=} sus"

# %%

run_name = "2rqzhpl3"
import transformer_lens.utils as tl_utils
cfg = tl_utils.download_file_from_hf("ArthurConmy/sae-replication", f"{run_name}.json")

my_sae = SAE(cfg = {**cfg, "d_sae": int(cfg["d_sae"]), "d_in": int(cfg["d_in"]), "dtype": eval(cfg["dtype"]), "device": "cuda:0"})
# "d_sae": int(cfg["d_sae"]), "d_in": int(cfg["d_in"], dtype = 

#%%

# Load in weights
pt_path = curpath.parent.parent.parent / "sae-replication" / (f"{run_name}.pt")

#%%

import huggingface_hub
my_file = huggingface_hub.hf_hub_download("ArthurConmy/sae-replication", f"{run_name}.pt")
with open(my_file, "rb") as f:
    state_dict = torch.load(f)
my_sae.load_state_dict(state_dict)

#%%

my_sae.to("cuda:0")

#%%

from datasets import load_dataset
ds = iter(load_dataset(cfg["dataset"], *(eval(cfg["dataset_args"]) or []), **(eval(cfg["dataset_kwargs"]) or {})))

#%%

toks = batch_tokens = get_batch_tokens(
    lm=lm,
    dataset=ds,
    seq_len=128,
    batch_size=100,
) # TODO test indetically to main.py

# %%

MODE = "zero"
runnin_loss_sum = 0.0

min_abs_denominators = []
loss_recovereds = []

for batch_idx in range(100):
    if batch_idx != 0:
        test_tokens = get_batch_tokens(
            lm=lm,
            dataset = ds,
            batch_size=60,
            seq_len=128,
        )
    else:
        test_tokens = toks

    def abl_hook(mlp_acts, hook):
        if MODE == "zero": 
            return 0.0*mlp_acts 

        else:
            assert MODE == "mean"
            mlp_acts[:] = torch.mean(mlp_acts, dim = (0, 1), keepdim=True) # mean over batch and sequence, so each MLP hidden feature is a single number
            return mlp_acts

    loss_func = torch.nn.CrossEntropyLoss(reduction="none")
    ablation_logits = lm.run_with_hooks(
        test_tokens,
        fwd_hooks=[(cfg["act_name"], abl_hook)],
    )
    ablation_loss = loss_func(ablation_logits[:, :-1, :].flatten(0, 1), test_tokens[:, 1:].flatten(0, 1))

    unflattened_ablation_loss = ablation_loss.reshape(test_tokens.shape[0], test_tokens.shape[1] - 1)

    unflattened_sae_loss = my_sae.get_test_loss(
        lm, test_tokens, return_mode="all"
    )
    
    sae_loss = unflattened_sae_loss.flatten()

    # Get loss normally
    lm_logits = lm(test_tokens)
    lm_loss = loss_func(lm_logits[:, :-1, :].flatten(0, 1), test_tokens[:, 1:].flatten(0, 1))
    unflatten_lm_loss = lm_loss.reshape(test_tokens.shape[0], test_tokens.shape[1] - 1)


    # Calculate the loss recovered
    loss_recovered = (ablation_loss - sae_loss) / (ablation_loss - lm_loss)

    stable_loss_recovered = (ablation_loss.mean() - sae_loss.mean()) / (ablation_loss.mean() - lm_loss.mean())

    unflatten_loss_recovered = loss_recovered.reshape(test_tokens.shape[0], test_tokens.shape[1] - 1)

    # Sort the test_losses by true losses
    # sorted_test_losses = sorted(enumerate(sae_loss.tolist()), key = lambda x: lm_loss[x[0]])

    arr1inds = lm_loss.argsort().tolist()
    sorted_lm_loss = lm_loss[arr1inds[::-1]]
    sorted_sae_loss = sae_loss[arr1inds[::-1]]
    sorted_ablation_loss = ablation_loss[arr1inds[::-1]]

    if False:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x = list(range(len(sorted_lm_loss))),
            y = sorted_ablation_loss.detach().cpu().numpy(),
            mode='lines',
        ))
        fig.add_trace(
        go.Scatter(
            x=list(range(len(sorted_lm_loss))),
            y=sorted_sae_loss.detach().cpu().numpy(),
            mode='lines',
        ))

        fig.add_trace(go.Scatter(
            x = list(range(len(sorted_lm_loss))),
            y = sorted_lm_loss.detach().cpu().numpy(),
            mode='lines',
        ))

        fig.show()

    # Sort the loss recovered

    sorted_loss_recovered = sorted(loss_recovered)
    middle_of_loss_recovered = torch.tensor(sorted_loss_recovered[len(sorted_loss_recovered)//200: 199*(len(sorted_loss_recovered)//200)])

    topk_denoms = (-(ablation_loss - lm_loss).abs()).topk(10)

    print(
        f"Loss recovered: {middle_of_loss_recovered.mean():.2f} +- {middle_of_loss_recovered.std():.2f}",
        "Full loss recovered:", round(loss_recovered.mean().item(), 4), round(loss_recovered.std().item(), 4),
        "Stable loss recovered:", round(stable_loss_recovered.mean().item(), 4), round(stable_loss_recovered.std().item(), 4),
        "Numerator",
        (ablation_loss - sae_loss).topk(10), 
        "Denominator",
        topk_denoms,
        "\n\n\n\n\n\n"
    ) # already found that numerator averaged and denominator averaged are much more stable

    # Get the minimum denominator
    min_abs_denominators.append(topk_denoms.values.abs().min().item())
    loss_recovereds.append(loss_recovered.mean().item())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = min_abs_denominators,
        y = loss_recovereds,
        mode='lines',
    ))

    runnin_loss_sum += loss_recovered.mean().item()
    print(f"Running loss sum: {runnin_loss_sum/(batch_idx+1):.2f}")
    break



#%%

# What do the bad points look like?
# Use unflattened_loss_recovered with argsort

sorted_loss_recovered_indices = (-loss_recovered).argsort()

batch_indices = sorted_loss_recovered_indices // (test_tokens.shape[1] - 1)
seq_indices = sorted_loss_recovered_indices % (test_tokens.shape[1] - 1)


#%%

get_batch_tokens(

)

#%%

for idx in range(10):
    b, s = batch_indices[idx], seq_indices[idx]
    print(
        "Loss recovered:", round(unflatten_loss_recovered[b, s].item(), 4), b, s,
    )
    paddinglen=3
    toks = lm.to_str_tokens(test_tokens[b, s-paddinglen:s+paddinglen+1])
    toks[paddinglen] = "***" + toks[paddinglen] + "***"
    print(toks)

# %%

#TODO: test whether the 83% or whatever caused this