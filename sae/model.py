from typing import Literal, Optional, Union, Tuple, List, Dict, Iterable, Any, Callable, Sequence
from tqdm.auto import tqdm
import wandb
from pathlib import Path
from sae.utils import get_neel_model, get_cfg, get_batch_tokens
import torch.nn as nn
import torch
import einops
from transformer_lens.hook_points import HookedRootModule, HookPoint
from torch.distributions.multinomial import Categorical

ENTITY = "ArthurConmy"
PROJECT = "sae"

class SAE(HookedRootModule):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg["d_in"]
        if not isinstance(self.d_in, int):
            raise ValueError(
                f"d_in must be an int but was {self.d_in=}; {type(self.d_in)=}"
            )
        self.d_sae = cfg["d_sae"]
        self.dtype = cfg["dtype"]
        self.device = cfg["device"]

        # NOTE: if using resampling neurons method, you must ensure that we initialise the weights in the order W_enc, b_enc, W_dec, b_dec
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_in, self.d_sae, dtype=self.dtype, device=self.device)
            )   
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.d_sae, dtype=self.dtype, device=self.device)
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_sae, self.d_in, dtype=self.dtype, device=self.device)
            )
        )

        with torch.no_grad():
            # Anthropic normalize this to have unit columns
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

        self.b_dec = nn.Parameter(
            torch.zeros(self.d_in, dtype=self.dtype, device=self.device)
        )

        self.hook_sae_in = HookPoint()
        self.hook_hidden_pre = HookPoint()
        self.hook_hidden_post = HookPoint()
        self.hook_sae_out = HookPoint()

        self.setup()  # Required for `HookedRootModule`s

    def forward(
        self, 
        x,
        return_mode: Literal["sae_out", "hidden_post", "both"]="both",
    ):
        sae_in = self.hook_sae_in(
            x - self.b_dec
        )  # Remove encoder bias as per Anthropic

        hidden_pre = self.hook_hidden_pre(
            einops.einsum(
                sae_in,
                self.W_enc,
                "... d_in, d_in d_sae -> ... d_sae",
            )
            + self.b_enc
        )
        hidden_post = self.hook_hidden_post(torch.nn.functional.relu(hidden_pre))

        sae_out = self.hook_sae_out(
            einops.einsum(
                hidden_post,
                self.W_dec,
                "... d_sae, d_sae d_in -> ... d_in",
            )
            + self.b_dec
        )

        if return_mode == "sae_out":
            return sae_out
        elif return_mode == "hidden_post":
            return hidden_post
        elif return_mode == "both":
            return sae_out, hidden_post
        else:
            raise ValueError(f"Unexpected {return_mode=}")

    @torch.no_grad()
    def get_test_loss(self, lm, test_tokens, cfg, hook=None, return_mode: Literal["mean", "all"]="mean"):
        # with torch.autocast("cuda", torch.bfloat16):
        
        all_correct_logprobs = None
        for start_index in range(0, test_tokens.shape[0], cfg["test_set_batch_size"]):
            logits = lm.run_with_hooks(
                test_tokens[start_index : start_index + cfg["test_set_batch_size"]],
                fwd_hooks=[
                    (
                        self.cfg["act_name"],
                        hook or (lambda activation, hook: self.forward(activation, return_mode="sae_out")),
                    )
                ],
            )
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            correct_logprobs = logprobs[
                torch.arange(logprobs.shape[0])[:, None],
                torch.arange(logprobs.shape[1] - 1)[None],
                test_tokens[start_index : start_index + cfg["test_set_batch_size"]][:, 1:],
            ]
            if all_correct_logprobs is None:
                all_correct_logprobs = correct_logprobs.cpu()
            else:
                all_correct_logprobs = torch.cat(
                    [all_correct_logprobs, correct_logprobs.cpu()], dim=0
                )

        if return_mode == "mean":
            return -all_correct_logprobs.mean()
        elif return_mode == "all":
            return -all_correct_logprobs
        else:
            raise ValueError(f"Unexpected {return_mode=}")

    def reinit_neurons(
        self,
        indices,
        opt,
    ):
        new_W_enc = torch.nn.init.kaiming_uniform_(
            torch.empty(
                self.d_in, indices.shape[0], dtype=self.dtype, device=self.device
            )
        )
        new_W_enc /= torch.norm(new_W_enc, dim=0, keepdim=True)
        self.W_enc.data[:, indices] = new_W_enc
        if indices.shape[0] < self.d_sae:
            sum_of_all_norms = torch.norm(self.W_enc.data, dim=0).sum()
            sum_of_all_norms -= len(indices)
            average_norm = sum_of_all_norms / (self.d_sae - len(indices))
            # metrics["resample_norm_thats_hopefully_less_or_around_one"] = average_norm.item()
            self.W_enc.data[:, indices] *= self.cfg["resample_factor"] * average_norm
        else:
            # Whatever, norm 1 times resample factor seems fiiiiine
            self.W_enc.data[:, indices] *= self.cfg["resample_factor"]
        new_W_enc *= self.cfg["resample_factor"]
        new_b_enc = torch.zeros(
            indices.shape[0], dtype=self.dtype, device=self.device
        )
        self.b_enc.data[indices] = new_b_enc
        self.W_dec.data[indices, :] = self.W_enc.data[:, indices].T # Clone em!
        self.W_dec.data[indices, :] /= torch.norm(self.W_dec.data[indices, :], dim=1, keepdim=True)
        self.W_dec /= torch.norm(self.W_dec, dim=1, keepdim=True)

    def anthropic_resample(
        self,
        indices, 
        opt,
        lm,
        dataset,
        metrics,
    ):
        anthropic_iterator = range(0, self.cfg["anthropic_resample_batches"], self.cfg["batch_size"])
        total_size = len(anthropic_iterator) * self.cfg["batch_size"] * self.cfg["seq_len"]
        anthropic_iterator = tqdm(anthropic_iterator, desc="Anthropic loss calculating")
        global_loss_increases = torch.zeros((self.cfg["anthropic_resample_batches"],), dtype=self.dtype, device=self.device)
        global_input_activations = torch.zeros((self.cfg["anthropic_resample_batches"], self.d_in), dtype=self.dtype, device=self.device)

        for refill_batch_idx_start in anthropic_iterator:
            refill_batch_tokens = get_batch_tokens(
                lm=lm,
                dataset=dataset,
                batch_size=self.cfg["batch_size"],
                seq_len=self.cfg["seq_len"],
            )
            # Do a forwards pass, including calculating loss increase

            sae_loss = self.get_test_loss(
                lm=lm,
                test_tokens=refill_batch_tokens,
                return_mode="all",
            )

            cache=[]
            def caching_and_replace_activation(activation, hook):
                cache.append(activation)
                return self.forward(activation, return_mode="sae_out")

            normal_loss, normal_activations_cache = lm.run_with_cache(
                refill_batch_tokens,
                names_filter=self.cfg["act_name"],
                return_type = "loss",
                loss_per_token = True,
            )
            normal_activations = normal_activations_cache[self.cfg["act_name"]]

            changes_in_loss = sae_loss - normal_loss
            changes_in_loss_dist = Categorical(
                torch.nn.functional.relu(changes_in_loss) / torch.nn.functional.relu(changes_in_loss).sum(dim=1, keepdim=True)
            )
            samples = changes_in_loss_dist.sample()
            assert samples.shape == (self.cfg["batch_size"],), f"{samples.shape=}; {self.cfg['batch_size']=}"
            
            global_loss_increases[
                refill_batch_idx_start: refill_batch_idx_start + self.cfg["batch_size"]
            ] = changes_in_loss[torch.arange(self.cfg["batch_size"]), samples]
            global_input_activations[
                refill_batch_idx_start: refill_batch_idx_start + self.cfg["batch_size"]
            ] = normal_activations[torch.arange(self.cfg["batch_size"]), samples]

        sample_indices = torch.multinomial(
            global_loss_increases / global_loss_increases.sum(),
            len(indices), 
            replacement=False,
        )

        # Replace W_dec with normalized versions of these
        self.W_dec.data[indices, :] = (
            (
                global_input_activations[sample_indices]
                / torch.norm(global_input_activations[sample_indices], dim=1, keepdim=True)
            )
            .to(self.dtype)
            .to(self.device)
        )


        # Set W_enc equal to W_dec.T in these indices, first
        self.W_enc.data[:, indices] = self.W_dec.data[indices, :].T

        # Then, change norms to be equal to a factor (0.2 in Anthropic) times the average norm of all the other columns, if other columns exist
        if indices.shape[0] < self.d_sae:
            sum_of_all_norms = torch.norm(self.W_enc.data, dim=0).sum()
            sum_of_all_norms -= len(indices)
            average_norm = sum_of_all_norms / (self.d_sae - len(indices))
            metrics["resample_norm_thats_hopefully_less_or_around_one"] = average_norm.item()
            self.W_enc.data[:, indices] *= self.cfg["resample_factor"] * average_norm

            # Set biases to resampledvalue
            relevant_biases = self.b_enc.data[indices].mean()
            self.b_enc.data[indices] = relevant_biases * self.cfg["bias_resample_factor"]

        else:
            self.W_enc.data[:, indices] *= self.cfg["resample_factor"]
    
            self.b_enc.data[indices] = - 5.0

    @torch.no_grad()
    def resample_neurons(
        self,
        indices,
        opt,
        sched=None,
        dataset=None,
        sae=None,
        lm=None,
        metrics=None,
    ):
        """Do Resampling"""

        if len(indices.shape) != 1 or indices.shape[0] == 0:
            raise ValueError(
                f"indices must be a non-empty 1D tensor but was {indices.shape}"
            )

        if self.cfg["resample_mode"] == "reinit":
            self.reinit_neurons(indices=indices, opt=opt)
        elif self.cfg["resample_mode"] == "anthropic":
            # Anthropic resampling
            self.anthropic_resample(indices=indices, opt=opt, lm=lm, dataset=dataset, metrics=metrics)
        else:
            raise ValueError(f"Unexpected {self.cfg['resample_mode']=}")

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
                        print(
                            "Warning: it does not seem as if resetting the Adam parameters worked, there are shapes mismatches"
                        )
                    if v[v_key][:, indices].abs().max().item() > 1e-6:
                        print(
                            "Warning: it does not seem as if resetting the Adam parameters worked"
                        )

        if sched is not None:
            # Keep on stepping till we're cfg["lr"] * cfg["sched_lr_factor"]

            max_iters = 10**7
            while sched.get_last_lr()[0] > self.cfg["lr"] * self.cfg["sched_lr_factor"] + 1e-9:
                sched.step()
                max_iters -= 1
                if max_iters == 0:
                    raise ValueError("Too many iterations -- sched is messed up")

    def load_from_neels(self, version: int = 1):
        neel_cfg, state_dict = get_neel_model(version)
        cfg = get_cfg(
            d_in=neel_cfg["d_mlp"],
            d_sae=neel_cfg["d_mlp"] * neel_cfg["dict_mult"],
            dtype={"fp32": torch.float32}[neel_cfg["enc_dtype"]],
        )
        self.load_state_dict(state_dict=state_dict, **({"map_location": torch.device('cpu')} if not torch.cuda.is_available() else {}))


    def get_config(
        self,
        run_id: str,
    ):
        api = wandb.Api()
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        return run.config    

    def load_from_my_wandb(
        self, 
        run_id: str, 
        index_from_back_override = None,
        index_from_front_override = None,
    ):
        if index_from_back_override is not None and index_from_front_override is not None:
            raise ValueError(f"Can't have both {index_from_back_override=} and {index_from_front_override=}")

        api = wandb.Api()
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

        list_logged_artifacts = list(run.logged_artifacts())
        len_logged_arifacts = len(list_logged_artifacts)

        # Kinda cursed three way condition in one line
        index_iterator = range(len_logged_arifacts-1, -1, -1) if index_from_back_override is None and index_from_front_override is None else [(int(index_from_front_override is not None)*2 - 1) * (index_from_back_override or index_from_front_override)]

        for index in index_iterator:
            try:
                logged_artifact = list_logged_artifacts[index] 
                logged_artifact_dir = Path(logged_artifact.download())
                dir_fnames = [p.name for p in Path(logged_artifact_dir).iterdir()]
                assert len(dir_fnames) == 1
                fname = dir_fnames[0]
                state_dict = torch.load(logged_artifact_dir / fname, **({"map_location": torch.device('cpu')} if not torch.cuda.is_available() else {}))
                if "W_in" in state_dict:
                    # old format of state dict 
                    state_dict = {
                        "W_enc": state_dict["W_in"],
                        "b_enc": state_dict["b_in"],
                        "W_dec": state_dict["W_out"],
                        "b_dec": state_dict["b_out"],
                    }
                self.load_state_dict(state_dict=state_dict)
            except Exception as e:
                print(f"Tried to load the {len_logged_arifacts-index}th from last, failed due to {e}")
            else:
                break

        try:
            self.load_state_dict(state_dict=state_dict) #, **({"map_location": torch.device('cpu')} if not torch.cuda.is_available() else {}))
        except Exception as e:
            print(f"Couldn't load because `{e}`; may want to check for config mismatch {run.cfg=}; {self.cfg=}")