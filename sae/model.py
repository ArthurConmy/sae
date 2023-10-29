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
        device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.dtype = dtype
        self.device = device

        self.W_in = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_in, d_sae, dtype=dtype, device=device)))
        self.b_in = nn.Parameter(torch.zeros(d_sae, dtype=dtype, device=device))
        
        self.W_out = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_sae, d_in, dtype=dtype, device=device)))
        self.b_out = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

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
                "batch d_sae, d_sae d_in -> batch d_in",
            )
            + self.b_out
        )
        return sae_out, hidden_post

    @torch.no_grad()
    def resample_neurons(self, indices):
        if len(indices.shape) != 1 and indices.shape[0] > 0:
            raise ValueError(f"indices must be a non-empty 1D tensor but was {indices.shape}")
        
        new_W_in = torch.nn.init.kaiming_uniform_(torch.empty(self.d_in, indices.shape[0], dtype=self.dtype, device=self.device))
        new_W_out = torch.nn.init.kaiming_uniform_(torch.empty(indices.shape[0], self.d_in, dtype=self.dtype, device=self.device))
        new_b_in = torch.zeros(indices.shape[0], dtype=self.dtype, device=self.device)

        self.W_in.data[:, indices] = new_W_in
        self.W_out.data[indices, :] = new_W_out
        self.b_in.data[indices] = new_b_in
