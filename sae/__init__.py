"""Special stuff for torch.compile (For later...)"""

import torch
torch.set_float32_matmul_precision('high') # When training if float32, torch.compile asked us to add this

import einops
from einops._torch_specific import allow_ops_in_compiled_graph # requires einops>=0.6.1
allow_ops_in_compiled_graph()