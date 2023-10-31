import os
import subprocess
from itertools import product
import numpy as np 
import multiprocessing
from math import gcd

used = set()

def run_script(threshold, gpu_id, kwargs): # Note fake kwargs
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(["python", os.path.expanduser("~/sae/sae/script_for_training.py")] + [f"--lr={kwargs['lr']}", f"--l1_lambda={kwargs['l1_lambda']}", f"--seed={kwargs['seed']}"], env=env)
    # subprocess.run(["python", os.path.expanduser("~/sae/scratch.py")] + [f"--seed={kwargs['seed']}"], env=env)

if __name__ == '__main__':

    num_gpus = 7 # specify the number of GPUs available
    num_jobs_per_gpu = 1 # specify the number of jobs per GPU

    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
    jobs = []

    for it in range(3, int(1e6)):        
        curspace = list(product([0.7 * 1e-4, 0.75 * 1e-4, 0.8 * 1e-4], (1e-4 * np.array([3.5, 4, 4.5])).tolist(), [1]))

        if not isinstance(curspace, list):
            curspace = curspace[1:-1]

        for threshold_idx, threshold in list(enumerate(curspace)):
            if threshold in used:
                continue
            used.add(threshold)

            gpu_id = (threshold_idx // num_jobs_per_gpu) % num_gpus
            jobs.append(pool.apply_async(run_script, (threshold, gpu_id, {"lr": threshold[0], "l1_lambda": threshold[1], "seed": threshold[2]})))

        if isinstance(curspace, list):
            break

    # wait for all jobs to finish
    for job in jobs:
        job.get()

    # clean up
    pool.close()
    pool.join()
