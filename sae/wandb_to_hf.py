#%%

import IPython
ipython = IPython.get_ipython()
if ipython is not None:
    ipython.magic("%load_ext autoreload")
    ipython.magic("%autoreload 2")

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

import torch; lm = transformer_lens.HookedTransformer.from_pretrained("gelu-1l") 
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

# %%

for _ in range(100):
    test_tokens = get_batch_tokens(
        lm=lm,
        dataset = ds,
        batch_size=30,
        seq_len=256,
    )


    loss_func = torch.nn.CrossEntropyLoss(reduction="none")
    factor = 0.0
    zero_ablation_logits = lm.run_with_hooks(
        test_tokens,
        fwd_hooks=[(cfg["act_name"], lambda activation, hook: factor*activation)],
    )
    zero_ablation_loss = loss_func(zero_ablation_logits[:, :-1, :].flatten(0, 1), test_tokens[:, 1:].flatten(0, 1))


    sae_loss = my_sae.get_test_loss(
        lm, test_tokens, return_mode="all"
    ).flatten()


    # Get loss normally
    lm_logits = lm(test_tokens)
    lm_loss = loss_func(lm_logits[:, :-1, :].flatten(0, 1), test_tokens[:, 1:].flatten(0, 1))


    # Calculate the loss recovered
    loss_recovered = (zero_ablation_loss - sae_loss) / (zero_ablation_loss - lm_loss)


    # Sort the test_losses by true losses
    # sorted_test_losses = sorted(enumerate(sae_loss.tolist()), key = lambda x: lm_loss[x[0]])

    arr1inds = lm_loss.argsort().tolist()
    sorted_lm_loss = lm_loss[arr1inds[::-1]]
    sorted_sae_loss = sae_loss[arr1inds[::-1]]
    sorted_zero_ablation_loss = zero_ablation_loss[arr1inds[::-1]]

    if False:
        
        import plotly.express as px

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x = list(range(len(sorted_lm_loss))),
            y = sorted_zero_ablation_loss.detach().cpu().numpy(),
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


    # sort the loss recovered

    sorted_loss_recovered = sorted(loss_recovered)
    middle_of_loss_recovered = torch.tensor(sorted_loss_recovered[len(sorted_loss_recovered)//4: 3*(len(sorted_loss_recovered)//4)])

    print(
        f"Loss recovered: {middle_of_loss_recovered.mean():.2f} +- {middle_of_loss_recovered.std():.2f}",
        "Full loss recovered:", round(loss_recovered.mean().item(), 4), round(loss_recovered.std().item(), 4),
    )

#%%
