import torch.nn as nn
import torch
import einops
from transformer_lens.hook_points import HookedRootModule, HookPoint
from torch.distributions.multinomial import Multinomial

class SAE(HookedRootModule):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg["d_in"]
        self.d_sae = cfg["d_sae"]
        self.dtype = cfg["dtype"]
        self.device = cfg["device"]

        # NOTE: ensure that we initialise the weights in the order W_in, b_in, W_out, b_out as editing Adam states requires this
        self.W_in = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.d_in, self.d_sae, dtype=self.dtype, device=self.device)))
        self.b_in = nn.Parameter(torch.zeros(self.d_sae, dtype=self.dtype, device=self.device))
        
        self.W_out = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.d_sae, self.d_in, dtype=self.dtype, device=self.device)))

        with torch.no_grad():
            # Anthropic normalize this to have unit columns
            self.W_out /= torch.norm(self.W_out, dim=1, keepdim=True)

        self.b_out = nn.Parameter(torch.zeros(self.d_in, dtype=self.dtype, device=self.device))

        self.hook_sae_in = HookPoint()
        self.hook_hidden_pre = HookPoint()
        self.hook_hidden_post = HookPoint()
        self.hook_sae_out = HookPoint()

        self.setup() # Required for `HookedRootModule`s

    def forward(self, x):
        sae_in = self.hook_sae_in(x - self.b_out) # Remove encoder bias as per Anthropic

        hidden_pre = self.hook_hidden_pre(
            einops.einsum(
                sae_in,
                self.W_in,
                "... d_in, d_in d_sae -> ... d_sae",
            )
            + self.b_in
        )
        hidden_post = self.hook_hidden_post(torch.nn.functional.relu(hidden_pre))

        sae_out = self.hook_sae_out(
            einops.einsum(
                hidden_post,
                self.W_out,
                "... d_sae, d_sae d_in -> ... d_in",
            )
            + self.b_out
        )

        return sae_out, hidden_post

    @torch.no_grad()
    def get_test_loss(self, lm, test_tokens):
        with torch.autocast("cuda", torch.bfloat16):
            logits = lm.run_with_hooks(
                test_tokens,
                fwd_hooks=[(self.cfg["act_name"], lambda activation, hook: einops.rearrange(self.forward(einops.rearrange(activation, "batch seq d_in -> (batch seq) d_in"))[0], "(batch seq) d_in -> batch seq d_in", batch=test_tokens.shape[0]))],
            )
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            correct_logprobs = logprobs[torch.arange(logprobs.shape[0])[:, None], torch.arange(logprobs.shape[1]-1)[None], test_tokens[:, 1:]]
            test_loss = -correct_logprobs.mean()
        
        return test_loss

    @torch.no_grad()
    def resample_neurons(self, indices, opt, resample_sae_loss_increases, resample_mlp_post_acts):
        """Do Anthropic-style resampling"""

        if len(indices.shape) != 1 and indices.shape[0] > 0:
            raise ValueError(f"indices must be a non-empty 1D tensor but was {indices.shape}")

        if False:        
            new_W_in = torch.nn.init.kaiming_uniform_(torch.empty(self.d_in, indices.shape[0], dtype=self.dtype, device=self.device))
            new_b_in = torch.zeros(indices.shape[0], dtype=self.dtype, device=self.device)
            new_W_out = torch.nn.init.kaiming_uniform_(torch.empty(indices.shape[0], self.d_in, dtype=self.dtype, device=self.device))

            self.W_in.data[:, indices] = new_W_in
            self.b_in.data[indices] = new_b_in
            
            self.W_out.data[indices, :] = new_W_out
            self.W_out /= torch.norm(self.W_out, dim=1, keepdim=True)
            return # Rip no Adam reset

        # Sample len(indices) number of indices from resample_sae_loss_increases proportio
        unnormalized_probability_distribution = torch.nn.functional.relu(resample_sae_loss_increases)
        probability_distribution = Multinomial(probs=unnormalized_probability_distribution / unnormalized_probability_distribution.sum())
        
        # Sample len(indices) number of indices without replacement
        samples = probability_distribution.sample((indices.shape[0],))
        resampled_neuron_outs = torch.nonzero(samples, as_tuple=False).squeeze(1)

        # Update the weights

        # Reset all the Adam parameters
        