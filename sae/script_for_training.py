import subprocess
import os
from itertools import product
import numpy as np
import multiprocessing

used = set()

def run_script(threshold, gpu_id, kwargs):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    args = ["python", os.path.expanduser("~/sae/sae/script_for_training.py")]
    args.extend([f"--lr={kwargs['lr']}", f"--l1_lambda={kwargs['l1_lambda']}", f"--seed={kwargs['seed']}"])
    print(" ".join(args))
    subprocess.run(args, env=env)

if __name__ == '__main__':

    num_gpus = 8  # Number of GPUs available
    num_jobs_per_gpu = 1  # Number of jobs per GPU

    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
    jobs = []

    for it in range(3, int(1e6)):
        curspace = list(product([0.7 * 1e-4, 0.75 * 1e-4, 0.8 * 1e-4], [3.75 * 1e-4, 4 * 1e-4], [1]))[:1]
        
        for threshold_idx, threshold in enumerate(curspace):
            if threshold in used:
                continue
            used.add(threshold)

            gpu_id = (threshold_idx // num_jobs_per_gpu) % num_gpus
            jobs.append(pool.apply_async(run_script, (threshold, gpu_id, {"lr": threshold[0], "l1_lambda": threshold[1], "seed": threshold[2]})))
        
        if curspace:
            break

    for job in jobs:
        job.get()
