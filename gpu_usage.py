import torch
from main import *
import os
import time
import psutil
import pynvml
import datetime
import csv
from copy import deepcopy
"""
Perform a GPU usage analysis for the training script based on the number of particles in the input
"""


LOGFILE = "performance_log.csv"


def write_to_csv(row, logfile=LOGFILE):
    file_exists = os.path.isfile(logfile)
    # Open the CSV file in append mode
    with open(logfile, 'a', newline='') as csvfile:
        # Use the keys of the row as header fields
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        # If the file does not exist, write the header
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def measure_performance(func):
    def wrapper(*args, **kwargs):
        # Initialize NVML for GPU monitoring
        pynvml.nvmlInit()
        # Change the index if you want to monitor a GPU other than the first one.
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_util_before = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        gpu_mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used

        process = psutil.Process(os.getpid())
        cpu_mem_before = process.memory_info().rss
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        gpu_util_after = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        gpu_mem_after = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        pynvml.nvmlShutdown()

        cpu_mem_after = process.memory_info().rss
        
        cpu_mem_change = (cpu_mem_after - cpu_mem_before) / (1024 * 1024)  # in MB
        gpu_mem_change = (gpu_mem_after - gpu_mem_before) / (1024 * 1024)
        
        row = {
            "timestamp": datetime.datetime.now().isoformat(),
            "execution_time_sec": round(elapsed_time, 4),
            "cpu_mem_change_MB": round(cpu_mem_change, 4),
            "gpu_util_before": gpu_util_before,
            "gpu_util_after": gpu_util_after,
            "gpu_mem_change_MB": round(gpu_mem_change, 4)
        }
        row.update({'epoch': args[0].epochs,
                    'model': args[0].model,
                    'n_balls': args[0].n_balls,
                    'exp_name': args[0].exp_name})
        return result
    return wrapper


@measure_performance
def check_performance(args):
    main(args)

if __name__ == "__main__":
    models = ['egno', 'segno']
    n_balls_list = [3, 5, 20, 50]
    base_args = get_args()
    base_args.epochs = 10
    base_args.outf = Path('performance_results')

    for model in models:
        for n_balls in n_balls_list:
            args = deepcopy(base_args)
            args.model = model
            args.n_balls = n_balls
            args.exp_name = f'{model}_n_balls_{n_balls}'

            print(f'Running {model} with {n_balls} balls...')
            main(args)  # Assuming main function handles the training and logging