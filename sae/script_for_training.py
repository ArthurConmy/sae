import subprocess
import os
from itertools import product
import numpy as np
import multiprocessing

used = set()

def run_script(threshold, gpu_id, keywords):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    args = ["python", os.path.expanduser("~/sae/sae/main.py")]
    for key, value in keywords.items():
        args.append(f"--{key}={value}")
    print(" ".join(args))
    subprocess.run(args, env=env)

if __name__ == '__main__':

    num_gpus = 6 # Number of GPUs available
    num_jobs_per_gpu = 2  # Number of jobs per GPU

    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
    jobs = []
    keyword_list = []

    for width in [2048, 16384*8, 16384]:
        for lr in [5.5 * 1e-5, 3e-5]:
            for l1_lambda in [0.0012, 0.001]:
                keyword_list.append({"d_sae": width, "lr": lr, "l1_lambda": l1_lambda})
                # if width > 100_000:
                    # keyword_list[-1]["buffer_size"] = 2**16

    for threshold_idx, keywords in enumerate(keyword_list):
        gpu_id = (threshold_idx // num_jobs_per_gpu) % num_gpus
        jobs.append(pool.apply_async(run_script, (keywords, gpu_id, keywords)))
    
    for job in jobs:
        print(job, "\n")
        job.get()

