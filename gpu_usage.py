import torch
from main import get_args
from pathlib import Path
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import psutil
import pynvml
import datetime
from copy import deepcopy
import wandb
import pandas as pd
"""
Perform a GPU usage analysis for the training script based on the number of particles in the input
"""


LOGFILE = "performance_log.csv"


def write_to_csv(row:dict, logfile=LOGFILE):
    file_exists = os.path.isfile(logfile)
    new_row = pd.DataFrame([row])
    if not file_exists:
        new_row.to_csv(logfile, index=False)
    else:
        new_row.to_csv(logfile, mode='a', header=False, index=False)

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
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            print(f"Error during function execution: {e}")
            result = None
        
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
        write_to_csv(row, LOGFILE)
        return result
    return wrapper


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)[args.model.upper()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    loss_mse = nn.MSELoss()
    loss_mse_no_red = nn.MSELoss(reduction='none')
    if args.model == 'segno':
        from SEGNO.nbody.models.model import SEGNO
        from SEGNO.nbody.dataset_nbody import NBodyDataset #from nbody.dataset_nbody import NBodyDataset
        from SEGNO.nbody.train_nbody import run_epoch

        nbody_name = config['other_params']['nbody_name']

        dataset_train = NBodyDataset(args.data_dir, partition='train', dataset_name=nbody_name, dataset=args.dataset,
                                    max_samples=args.max_samples, n_balls=args.n_balls)
        loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

        params = config['model_params'] | dict(varDT=args.varDT, device=device)
        params['n_inputs'] = args.num_inputs
        # if args.num_inputs > 1:
        #     # All dynamical node_features for each input + the static one
        #     params['in_node_nf'] = (params['in_node_nf'] - 1) * args.num_inputs + 1
        model = SEGNO(**params)
        criterion = [loss_mse,loss_mse_no_red]
    else:
        from EGNO.simulation.dataset_simple import NBodyDynamicsDataset as SimulationDataset
        from EGNO.model.egno import EGNO
        from EGNO.main_simulation_simple_no import run_epoch

        args.varDT = True if args.varDT and args.num_inputs>1 else False

        dataset_train = SimulationDataset(data_dir=args.data_dir, partition='train', max_samples=args.max_samples, dataset=args.dataset, n_balls=args.n_balls, 
                                          num_timesteps=args.num_timesteps,num_inputs=args.num_inputs, varDT=args.varDT) #, num_inputs=args.num_inputs
        loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
        
        params = config['model_params'] | dict(num_timesteps=args.num_timesteps, num_inputs=args.num_inputs, varDT=args.varDT, device=device)
        model = EGNO(**params)
        criterion = loss_mse_no_red
        print(args.rollout,args.num_inputs,args.varDT, args.n_balls)

    optimizer = optim.Adam(model.parameters(), lr=float(config['training_params']['lr']), weight_decay=float(config['training_params']['weight_decay']))

    train_loss = run_epoch(model, optimizer, criterion, 0, loader_train, args)
    return train_loss



@measure_performance
def check_performance(args):
    main(args)


if __name__ == "__main__":
    models = ['egno', 'segno']
    n_balls_list = [5, 10, 20, 50]
    base_args = get_args()
    base_args.epochs = 3
    base_args.outf = Path('performance_results')
    wandb.init(project='gpu_usage_analysis', mode='disabled')  

    for model in models:
        for n_balls in n_balls_list:
            args = deepcopy(base_args)
            args.model = model
            args.n_balls = n_balls
            args.exp_name = f'{model}_n_balls_{n_balls}'

            print(f'Running {model} with {n_balls} balls...')
            check_performance(args)
