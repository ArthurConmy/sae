import torch.nn as nn
import torch
import einops
from transformer_lens.hook_points import HookedRootModule, HookPoint

class SAE(HookedRootModule):
    def __init__(
        self,
        d_in,
        d_sae,
        dtype=torch.float32,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.dtype = dtype

        self.W_in = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_in, d_sae, dtype=dtype)))
        self.b_in = nn.Parameter(torch.zeros(d_sae, dtype=dtype))
        
        self.W_out = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_sae, d_in, dtype=dtype)))
        self.b_out = nn.Parameter(torch.zeros(d_in, dtype=dtype))

        self.hook_sae_in = HookPoint()
        self.hook_hidden_pre = HookPoint()
        self.hook_hidden_post = HookPoint()
        self.hook_sae_out = HookPoint()

    def forward(self, x):
        sae_in = self.hook_sae_in(x)
        hidden_pre = self.hook_hidden_pre(
            einops.einsum(
                sae_in,
                self.W_in,
                "batch d_in, d_in d_sae -> batch d_sae",
            )
            + self.b_in
        )
        hidden_post = self.hook_hidden_post(torch.nn.functional.relu(hidden_pre))
        sae_out = self.hook_sae_out(
            einops.einsum(
                hidden_post,
                self.W_out,
                "batch d_sae -> batch d_in",
            )
            + self.b_out
        )
        return sae_out