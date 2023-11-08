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

    num_gpus = 6 # Number of GPUs available
    num_jobs_per_gpu = 1  # Number of jobs per GPU

    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
    jobs = []
    keyword_list = []

    for width in [2048, 16384, 16384*8]:
        for lr in [5.5 * 1e-5, 1e-5]:
            keyword_list.append({"d_sae": width, "lr": lr})
            if width > 100_000:
                keyword_list[-1]["buffer_size"] = 2**16

    for threshold_idx, keywords in enumerate(keyword_list):
        gpu_id = (threshold_idx // num_jobs_per_gpu) % num_gpus
        jobs.append(pool.apply_async(run_script, (keywords, gpu_id, keywords)))
    
    for job in jobs:
        print(job, "\n")
        job.get()

