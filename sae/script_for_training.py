import torch
import subprocess
import os
from itertools import product
import numpy as np
import multiprocessing

used = set()

def run_script(threshold, gpu_id, keywords):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    args = ["python", "/workspace/sae/sae/main.py"]
    for key, value in keywords.items():
        args.append(f"--{key}={value}")
    print(" ".join(args))
    subprocess.run(args, env=env)

if __name__ == '__main__':

    num_gpus = 6 # Number of GPUs available
    num_jobs_per_gpu = 1  # Number of jobs per GPU

    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
    jobs = []
    keyword_list = []

    for width in [16384*8, 16384*4]: # [2048, 16384*8, 16384]:
        for l1_lambda in (torch.FloatTensor([8, 12, 16]) / 10_000).tolist(): # [0.0013, 0.001]:
            keyword_list.append({"d_sae": width, "lr": 0.0012, "l1_lambda": l1_lambda})

    # for width in [16384, 16384//8]: # [2048, 16384*8, 16384]:
    #     for l1_lambda in (torch.FloatTensor([8, 16]) / 10_000).tolist(): # [0.0013, 0.001]:
    #         keyword_list.append({"d_sae": width, "lr": 0.0012, "l1_lambda": l1_lambda})

    keyword_list[-1]["delete_cache"]=True

    # keyword_list.append(
    #     {
    #         "d_sae": 16384*16,
    #         "lr": 8e-4,
    #         "l1_lambda": 12 / 10_000,
    #     },
    # )

    # keyword_list = [
    #     {},
    #     {},
    #     {} 
        # {"d_sae": 16384, "lr": 8e-4, "l1_lambda": 12 / 10_000},
        # {"d_sae": 16384, "lr": 12 * 1e-4, "l1_lambda": 12 / 10_000}, 
        # {"d_sae": 16384*16, "lr": 8e-4, "l1_lambda": 12 / 10_000},
        # {"d_sae": 16384*16, "lr": 12*1e-4, "l1_lambda": 12 / 10_000}, # Two super promising runs
        # {"d_sae": 16384*16, "lr": 12*1e-4, "l1_lambda": 16 / 10_000}, # Two super promising runs - another in a similar vein, scaling like Anthropic did
        # {"d_sae": 16384, "lr": 8 * 1e-4, "l1_lambda": 8 / 10_000}, # Also stick these on GPU 0-1
        # {"d_sae": 16384, "lr": 12 * 1e-4, "l1_lambda": 8 / 10_000},
    # ]

# if width > 100_000:
    # keyword_list[-1]["buffer_size"] = 2**16

    for threshold_idx, keywords in enumerate(keyword_list):
        gpu_id = (threshold_idx // num_jobs_per_gpu) % num_gpus
        jobs.append(pool.apply_async(run_script, (keywords, gpu_id, keywords)))
    
    for job in jobs:
        print(job, "\n")
        job.get()

