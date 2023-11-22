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
    # LR-0.0012-LAMBDA-0.0015999999595806003-DSAW-2048Thu_Nov__9_02-56-22_2023_48
    # LR-0.0012-LAMBDA-0.0015999999595806003-DSAW-16384Thu_Nov__9_02-56-22_2023_51
    # LR-0.0012-LAMBDA-0.0007999999797903001-DSAW-2048Thu_Nov__9_02-56-22_2023_78
    # LR-0.0012-LAMBDA-0.0007999999797903001-DSAW-16384Thu_Nov__9_02-56-22_2023_84
    # for width in [16384*8, 16384*4]: # [2048, 16384*8, 16384]:
    #     for l1_lambda in (torch.FloatTensor([8, 12, 16]) / 10_000).tolist(): # [0.0013, 0.001]:
    #         keyword_list.append({"d_sae": width, "lr": 0.0012, "l1_lambda": l1_lambda})

    jobs = []
    keyword_list = []
    
    num_gpus = 8 # Number of GPUs available
    num_jobs_per_gpu = 2  # Number of jobs per GPU

    # # Add the smaller width things at the start and at the end
    # for l1_lambda in (torch.FloatTensor([8, 16]) / 10_000).tolist(): # [0.0013, 0.001]:
    #     for width in [16384]: # [2048, 16384*8, 16384]:
    #         # Make this basically the same, but with the 0.2 for resampled bois
    #         keyword_list.append({"d_sae": width, "lr": 0.0012, "l1_lambda": l1_lambda, "resample_factor": 0.2, "sched_type": "cosine_warmup", "buffer_size": 2**19})

    for l1_lambda in [0.0008, 0.0012]:
        for width in [2048, 131072, 65536, 100000, 1024]:
            keyword_list.append({"d_sae": width, "lr": 0.0012, "l1_lambda": l1_lambda, "sched_type": "none", "buffer_size": 2**19, "sched_type": "cosine_warmup"})

    for l1_lambda in (torch.FloatTensor([8, 16]) / 10_000).tolist(): # [0.0013, 0.001]:
        for width in [16384//8]: # [2048, 16384*8, 16384]:
            keyword_list.append({"d_sae": width, "lr": 0.0012, "l1_lambda": l1_lambda, "resample_factor": 0.2, "buffer_size": 2**19, "sched_type": "cosine_warmup"})


    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
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

    assert len(keyword_list) <= num_gpus * num_jobs_per_gpu, "Too many jobs for the number of GPUs available"
    for threshold_idx, keywords in enumerate(keyword_list):
        gpu_id = threshold_idx % num_gpus
        jobs.append(pool.apply_async(run_script, (keywords, gpu_id, keywords)))
    
    for job in jobs:
        print(job, "\n")
        job.get()

