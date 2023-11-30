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
    jobs = []
    keyword_list = []
    
    num_gpus = 8 # Number of GPUs available
    num_jobs_per_gpu = 1  # Number of jobs per GPU

    for width in [131072, 65536]:
        keyword_list.append({
            "d_sae": width,
            "lr": 0.0012,
            "l1_lambda": 16 / 10_000,
        })

    for l1_lambda in [0.0008, 0.0012, 0.0016]:
        keyword_list.append({
            "d_sae": 131072,
            "lr": 0.0012,
            "l1_lambda": l1_lambda,
            # non-Anthropic resampling
            "resample_mode": "reinit",
        })
        keyword_list.append({
            "d_sae": 131072,
            "lr": 0.0012,
            "l1_lambda": l1_lambda,
            "activation_training_order": "ordered",
        })

    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
    keyword_list[-1]["delete_cache"]=True

    assert len(keyword_list) <= num_gpus * num_jobs_per_gpu, "Too many jobs for the number of GPUs available"
    for threshold_idx, keywords in enumerate(keyword_list):
        gpu_id = threshold_idx % num_gpus
        jobs.append(pool.apply_async(run_script, (keywords, gpu_id, keywords)))
    
    for job in jobs:
        print(job, "\n")
        job.get()

