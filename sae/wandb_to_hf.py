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

#%%

import json
import os

for run in filtered_runs:
    cfg = run.config # Yeeeeah
    print(run.id)

    json_path = curpath.parent.parent.parent / "sae-replication" / (f"{run.id}.json")

    if not os.path.exists(json_path):
        # Save the config as a JSON, make all values strings
        with open(json_path, "w") as f:
            json.dump({k: str(v) for k, v in cfg.items()}, f)

    if isinstance(cfg["dtype"], str):
        cfg["dtype"] = eval(cfg["dtype"])

    if not torch.cuda.is_available(): 
        cfg["device"] = "cpu"

    my_sae=SAE(cfg)

    pt_path = curpath.parent.parent.parent / "sae-replication" / (f"{run.id}.pt")
    if not os.path.exists(pt_path):
        my_sae.load_from_my_wandb(
            run_id=run.id,
            # Nones mean it loads from back
        )
        torch.save(my_sae.state_dict(), pt_path)

    pickle_path = curpath.parent.parent.parent / "sae-replication" / (f"{run.id}.pkl")

    if not os.path.exists(pickle_path):
        
        if not os.path.exists(pt_path):
            my_sae.load_state_dict(torch.load(pt_path))
        
        # Also pickle dump the model
        import pickle
        import numpy as np
        numpy_state_dict = {k: np.array(v.float()) for k, v in my_sae.state_dict().items()}
        with open(pickle_path, "wb") as f:
            pickle.dump(numpy_state_dict, f)

#%%

assert False

# torch.save(my_sae.state_dict(), "hello.pt")

# !apt-get install git-lfs

# from huggingface_hub import HfFolder, Repository
# import torch

# # Authenticate with Hugging Face (you'll need to get your token from the Hugging Face website)
# hf_token = "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"  # Replace with your actual token
# HfFolder.save_token(hf_token)

# # Assuming you have your model in a variable `model`

# # Fine for now
# # model_path = "model.pt"  # The path where you want to save the model
# # torch.save(my_sae.state_dict(), model_path)  # Save model state dict

# # Define your repository name and local folder to clone to
# repo_name = "sae-replication"  # Replace with your repository name
# local_repo_path = "sae-replication"  # Local path to clone the repository to
# username = "ArthurConmy"  # Replace with your Hugging Face username

# # Clone the repository from Hugging Face
# repo = Repository(local_repo_path, clone_from=f"{username}/{repo_name}")

# # Copy the model file to the cloned repository directory
# !cp {model_path} {local_repo_path}

# # Using the Repository class to push to the hub
# repo.git_add(auto_lfs_track=True)  # Auto track large files with git-lfs
# repo.git_commit("Add new model")  # Commit with a message
# repo.git_push()  # Push to the hub

# print(f"Model pushed to your Hugging Face repository: {username}/{repo_name}")

# !huggingface-cli login

# !git clone

# %%

run_name = "2rqzhpl3"

# Load JSON
import json
json_path = curpath.parent.parent.parent / "sae-replication" / (f"{run_name}.json")

with open(json_path, "r") as f:
    cfg = json.load(f)

my_sae = SAE(cfg = {**cfg, "d_sae": int(cfg["d_sae"]), "d_in": int(cfg["d_in"]), "dtype": eval(cfg["dtype"]), "device": "cpu"})
# "d_sae": int(cfg["d_sae"]), "d_in": int(cfg["d_in"], dtype = 

#%%

# Load in weights
pt_path = curpath.parent.parent.parent / "sae-replication" / (f"{run_name}.pt")
my_sae.load_state_dict(torch.load(pt_path))

# %%

logits, cache = my_sae.run_with_cache(torch.ones(1, my_sae.d_in, dtype = my_sae.dtype))

# %%

cache.keys()

#%%

ds = iter(load_dataset(cfg["dataset"], *(eval(cfg["dataset_args"]) or []), **(eval(cfg["dataset_kwargs"]) or {})))

# %%

from datasets import load_dataset
# LOL EVEN WORSE
test_tokens = get_batch_tokens(
    lm=lm,
    dataset = next(ds),
    batch_size=1,
    seq_len=256,
)

#%%

test_loss = my_sae.get_test_loss(
    lm, test_tokens, return_mode="all"
)

#%%