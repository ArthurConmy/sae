#%%

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
from sae.model import SAE
import torch
import transformer_lens
from datasets import load_dataset
from tqdm.notebook import tqdm

#%%

lm = transformer_lens.HookedTransformer.from_pretrained("gelu-1l")

#%%

sae = SAE(d_in=lm.cfg.d_mlp, d_sae=lm.cfg.d_model) # For now, just keep same dimension

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

SEQ_LEN = 512
batch_tokens = torch.LongTensor(size=(0, SEQ_LEN)).to(lm.cfg.device)
current_batch = []
current_length = 0

for i in tqdm(range(10_000)):
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

CUTOFF_LAST = True

if not CUTOFF_LAST:
    # Handle the last batch if it's not empty
    if current_length > 0:
        last_batch = torch.cat(current_batch, dim=1)
        padded_last_batch = torch.nn.functional.pad(last_batch, (0, SEQ_LEN - current_length))
        batch_tokens = torch.cat((batch_tokens, padded_last_batch), dim=0)

# batch_tokens is now a tensor where each slice along dim=0 has shape [1, SEQ_LEN]

# %%

