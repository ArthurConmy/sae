import torch.nn as nn
import torch
import einops
from transformer_lens.hook_points import HookedRootModule, HookPoint
from torch.distributions.multinomial import Categorical

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

        # NOTE: if using resampling neurons method, you must ensure that we initialise the weights in the order W_in, b_in, W_out, b_out as editing Adam states requires this
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

        return sae_out, hidden_post # TODO setup return_mode: "sae_out", "hidden_post", "both"

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
    def resample_neurons(self, indices, opt, resample_sae_loss_increases, resample_mlp_post_acts, mode="anthropic"):
        """Do Anthropic-style resampling.
        
        TODO document and implement better. This is really messy!!! :("""

        if len(indices.shape) != 1 or indices.shape[0] == 0:
            raise ValueError(f"indices must be a non-empty 1D tensor but was {indices.shape}")

        if mode=="reinit":
            new_W_in = torch.nn.init.kaiming_uniform_(torch.empty(self.d_in, indices.shape[0], dtype=self.dtype, device=self.device))
            new_b_in = torch.zeros(indices.shape[0], dtype=self.dtype, device=self.device)
            new_W_out = torch.nn.init.kaiming_uniform_(torch.empty(indices.shape[0], self.d_in, dtype=self.dtype, device=self.device))

            self.W_in.data[:, indices] = new_W_in
            self.b_in.data[indices] = new_b_in
            
            self.W_out.data[indices, :] = new_W_out
            self.W_out /= torch.norm(self.W_out, dim=1, keepdim=True)

        else:
            # Sample len(indices) number of indices from resample_sae_loss_increases proportio
            unnormalized_probability_distribution = torch.nn.functional.relu(resample_sae_loss_increases)
            probability_distribution = Categorical(probs=unnormalized_probability_distribution / unnormalized_probability_distribution.sum())
            
            # Sample len(indices) number of indices
            # TODO check we aren't sampling the same point several times, that sounds bad
            samples = probability_distribution.sample((indices.shape[0],))
            resampled_neuron_outs = resample_mlp_post_acts[samples]

            # Replace W_out with normalized versions of these
            self.W_out.data[indices, :] = (resampled_neuron_outs / torch.norm(resampled_neuron_outs, dim=1, keepdim=True)).to(self.dtype).to(self.device)

            # Set biases to zero
            self.b_in.data[indices] = 0.0

            # Set W_in equal to W_out.T in these indices, first
            self.W_in.data[:, indices] = self.W_out.data[indices, :].T

            # Then, change norms to be equal to 0.2 times the average norm of all the other columns, if other columns exist
            if indices.shape[0] < self.d_sae:
                sum_of_all_norms = torch.norm(self.W_in.data, dim=0).sum()
                sum_of_all_norms -= len(indices)
                average_norm = sum_of_all_norms / (self.d_sae - len(indices))
                self.W_in.data[:, indices] *= 0.2 * average_norm

            else:
                self.W_in.data[:, indices] *= 0.2

        # Reset all the Adam parameters
        for dict_idx, (k, v) in enumerate(opt.state.items()):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    assert k.data.shape == (self.d_in, self.d_sae)
                    v[v_key][:, indices] = 0.0
                elif dict_idx == 1:
                    assert k.data.shape == (self.d_sae,)
                    v[v_key][indices] = 0.0
                elif dict_idx == 2:
                    assert k.data.shape == (self.d_sae, self.d_in)
                    v[v_key][indices, :] = 0.0
                elif dict_idx == 3:
                    assert k.data.shape == (self.d_in,)
                else:
                    raise ValueError(f"Unexpected dict_idx {dict_idx}")

        # Check that the opt is really updated
        for dict_idx, (k, v) in enumerate(opt.state.items()):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    if k.data.shape != (self.d_in, self.d_sae):
                        print("Warning: it does not seem as if resetting the Adam parameters worked, there are shapes mismatches")
                    if v[v_key][:, indices].abs().max().item() > 1e-6:
                        print("Warning: it does not seem as if resetting the Adam parameters worked")