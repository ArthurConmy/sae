#%%

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

from sae.model import SAE
from typing import Union, Literal
import torch
import transformer_lens
from datasets import load_dataset
from copy import deepcopy
from tqdm.notebook import tqdm

#%%

lm = transformer_lens.HookedTransformer.from_pretrained("gelu-1l")

#%%

D_SAE = lm.cfg.d_model
dummy_sae = SAE(d_in=lm.cfg.d_mlp, d_sae=D_SAE) # For now, just keep same dimension

#%%

# Get some C4 (validation data?) and then see what it looks like

#%%

hf_dataset_name = "stas/c4-en-10k"
c4_tenk = load_dataset(hf_dataset_name, split="train")

# %%

for i in range(10):
    s=c4_tenk[i]["text"]
    tokens=lm.to_tokens(s, truncate=False)
    print(i, tokens.shape)

# %%

# <BOS> ...  <BOS> ... works I think!
BOS = lm.tokenizer.bos_token_id

# %%

SEQ_LEN = lm.cfg.n_ctx
BATCH_SIZE = 150
batch_tokens = torch.LongTensor(size=(0, SEQ_LEN)).to(lm.cfg.device)
current_batch = []
current_length = 0
CUTOFF: Union[Literal["just_batch", "all"], int] = BATCH_SIZE * 10

print("Tokenizing...")
for i in tqdm(range(10_000)):

    if CUTOFF == "just_batch" and batch_tokens.shape[0] >= BATCH_SIZE:
        break

    if isinstance(CUTOFF, int) and batch_tokens.shape[0] >= CUTOFF:
        break

    s = c4_tenk[i]["text"]
    tokens = lm.to_tokens(s, truncate=False, move_to_device="cpu")
    token_len = tokens.shape[1]

    while token_len > 0:
        # Space left in the current batch
        space_left = SEQ_LEN - current_length

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
            current_length = SEQ_LEN

        # If a batch is full, concatenate and move to next batch
        if current_length == SEQ_LEN:
            full_batch = torch.cat(current_batch, dim=1)
            batch_tokens = torch.cat((batch_tokens, full_batch), dim=0)
            current_batch = []
            current_length = 0

print("... finished!")

CUTOFF_LAST = True

if not CUTOFF_LAST:
    # Handle the last batch if it's not empty
    if current_length > 0:
        last_batch = torch.cat(current_batch, dim=1)
        padded_last_batch = torch.nn.functional.pad(last_batch, (0, SEQ_LEN - current_length), value=lm.tokenizer.pad_token_id)
        batch_tokens = torch.cat((batch_tokens, padded_last_batch), dim=0)



# batch_tokens is now a tensor where each slice along dim=0 has shape [1, SEQ_LEN]

# %%

def loss_fn(
    sae_reconstruction,
    sae_hiddens,
    ground_truth,
    l1_lambda,
):
    # Note L1 is summed, L2 is meaned here. Should be the most appropriate when comparing losses across different LM size
    return torch.pow((sae_reconstruction-ground_truth), 2).mean(dim=(0, 1, 2)) + l1_lambda * torch.abs(sae_reconstruction).sum(dim=2).mean(dim=(0, 1))

LR = 1e-3
L1_LAMBDA = 1e-3

def train_step(
    model, 
    opt,
    mini_batch,
):
    opt.zero_grad(True)
        
    # TODO add feature, normalizing W_out columns (or rows??) to 1
    # old_wout = model.W_out.clone().detach() 
    # TODO then benchmark how much slower this is
    
    loss = loss_fn(*model(mini_batch), ground_truth=mini_batch, l1_lambda=L1_LAMBDA)
    saved_loss = loss.item()

    loss.backward()
    opt.step()

    return saved_loss

    # TODO actually apply the normalizing feature here
    # TODO ie remove the gradient jump in the direction of `old_wout`

# comp_train_step = torch.compile(train_step)
# TODO make compilable!

#%%

sae = deepcopy(dummy_sae).to(lm.cfg.device)

BATCH_SIZE = 100
assert batch_tokens.shape[0]%BATCH_SIZE==0, (batch_tokens.shape, BATCH_SIZE)
num_steps = batch_tokens.shape[0]//BATCH_SIZE
opt = torch.optim.Adam(sae.parameters(), lr=LR)

for step_num in tqdm(range(num_steps)):
    
    # Firstly get the activations
    mlp_post_name = "blocks.0.mlp.hook_post"

    mlp_post_acts = lm.run_with_cache(
        batch_tokens[step_num*BATCH_SIZE:(step_num+1)*BATCH_SIZE],
        names_filter=mlp_post_name,
        # device="cpu",
    )[1][mlp_post_name]

    # mini_batch = data[step_num*BATCH_SIZE:(step_num+1)*BATCH_SIZE].to(lm.cfg.device)
    saved_loss = train_step(sae, opt, mini_batch=mlp_post_acts)
    print(step_num, saved_loss)

# %%

