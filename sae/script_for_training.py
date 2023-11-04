import subprocess
import os
from itertools import product
import numpy as np
import multiprocessing

used = set()

def run_script(threshold, gpu_id, keywords):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(1)
    args = ["python", os.path.expanduser("~/sae/sae/main.py")]
    for key, value in keywords.items():
        args.append(f"--{key}={value}")
    print(" ".join(args))
    subprocess.run(args, env=env)

if __name__ == '__main__':

    num_gpus = 2 # Number of GPUs available
    num_jobs_per_gpu = 1  # Number of jobs per GPU

    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
    jobs = []
    keyword_list = [
        {"lr": 0.001, "l1_lambda": 0.00028},
#        {"lr": 0.001, "l1_lambda": 0.00028},
 #       {"lr": 0.001, "l1_lambda": 0.0002},
  #      {"lr": 0.002, "l1_lambda": 0.00024},
   #     {"lr": 0.002, "l1_lambda": 0.00020},
    #    {"lr": 0.002, "l1_lambda": 0.00022},        
    ]

    # keyword_list = [{"resample_sae_neurons_cutoff": 1e-6}, {"resample_sae_neurons_cutoff": 1e-6, "l1_lambda": 7.2 * 1e-4}, {"l1_lambda": 7.2 * 1e-4}, {}, {"l1_lambda": 7.2 * 1e-4, "resample_sae_neurons_at": []}, {"resample_sae_neurons_every": 10**20, "resample_sae_neurons_at": [25000, 50000, 75000], "resample_sae_neurons_cutoff": 1e-6}]
#    assert len(keyword_list)==6

    for threshold_idx, keywords in enumerate(keyword_list):
        gpu_id = (threshold_idx // num_jobs_per_gpu) % num_gpus
        jobs.append(pool.apply_async(run_script, (keywords, gpu_id, keywords)))
    
    for job in jobs:
        print(job, "\n")
        job.get()

