"""
Currently not used
"""

import torch
import einops
from tqdm import tqdm
from typing import Iterable, Union, Literal, List, Tuple, Optional, Dict

class Buffer():
    """
    Borrowed from Neel's repo, working with HuggingFace datasets.

    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty. 
    """
    def __init__(
        self,
        cfg,
        iterable_dataset: Iterable[str],
    ):
        self.buffer = torch.zeros((cfg["buffer_size"], cfg["act_size"]), dtype=torch.bfloat16, requires_grad=False).to(cfg["device"])
        self.cfg = cfg
        self.token_pointer = 0
        self.first = True
        self.iterable_dataset = iterable_dataset
        self.refresh()
    
    @torch.no_grad()
    def refresh(
        self, 
        model,
    ):
        # Prepare tokens from raw strings
        tokens = [next(self.iterable_dataset)["text"] for _ in range(self.cfg["model_batch_size"])]

        for i in tqdm(range(10_000)):

            if CUTOFF == "just_batch" and batch_tokens.shape[0] >= cfg["batch_size"]:
                break

            if isinstance(CUTOFF, int) and batch_tokens.shape[0] >= CUTOFF:
                break

            s = raw_all_data[i]["text"]
            tokens = lm.to_tokens(s, truncate=False, move_to_device=False) # Why is this so slow ???
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

        print("... finished!")

        CUTOFF_LAST = True

        if not CUTOFF_LAST:
            # Handle the last batch if it's not empty
            if current_length > 0:
                last_batch = torch.cat(current_batch, dim=1)
                padded_last_batch = torch.nn.functional.pad(last_batch, (0, cfg["seq_len"] - current_length), value=lm.tokenizer.pad_token_id)
                batch_tokens = torch.cat((batch_tokens, padded_last_batch), dim=0)

        FILTER_TO_MOD_SIZE = True

        if FILTER_TO_MOD_SIZE:
            batch_tokens = batch_tokens[:batch_tokens.shape[0] - (batch_tokens.shape[0] % cfg["batch_size"])]

        # batch_tokens is now a tensor where each slice along dim=0 has shape [1, SEQ_LEN]

        self.pointer = 0
        with torch.autocast("cuda", torch.bfloat16):
            if self.first:
                num_batches =           num_batches = self.cfg["buffer_batches"] // 2
            self.first = False
            for _ in range(0, num_batches, self.cfg["model_batch_size"]):

                
                _, cache = model.run_with_cache(tokens, stop_at_layer=self.cfg["layer"]+1, names_filter=self.cfg["act_name"])
                acts = cache[self.cfg["act_name"]].reshape(-1, self.cfg["act_size"])
                
                # print(tokens.shape, acts.shape, self.pointer, self.token_pointer)
                self.buffer[self.pointer: self.pointer+acts.shape[0]] = acts
                self.pointer += acts.shape[0]
                self.token_pointer += self.cfg["model_batch_size"]
                # if self.token_pointer > all_tokens.shape[0] - self.cfg["model_batch_size"]:
                #     self.token_pointer = 0

        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).to(self.cfg["device"])]

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer:self.pointer+self.cfg["batch_size"]]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0]//2 - self.cfg["batch_size"]:
            # print("Refreshing the buffer!")
            self.refresh()
        return out