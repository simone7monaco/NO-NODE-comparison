import argparse
from pathlib import Path
import pickle
import yaml
import random
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from EGNO.utils import EarlyStopping
import json
import wandb

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError(f'Invalid boolean value: {value}')
    
def get_args():
    parser = argparse.ArgumentParser(description='Main module for SEGNO and EGNO')
    parser.add_argument('model', type=str, choices=['segno', 'egno'],
                        help='Model to use: segno or egno')
    parser.add_argument('--exp_name', type=str, default='exp_2', help='Experiment name')
    parser.add_argument('--config', type=str, default='model_confs.yaml')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=500, # 1000,
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--data_dir', type=Path, default='data')
    parser.add_argument('--dataset', type=str, default='charged', choices=['charged', 'gravity'],
                        help='Dataset to use (default: charged)')
    parser.add_argument('--max_samples', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--only_test', type=str2bool, default=False)
    parser.add_argument('--traj_len', type=int, default=10,
                        help='Trajectory lenght in case of testing on rollout')
    parser.add_argument('--test_interval', type=int, default=5,
                        help='Test every test_interval epochs')
    parser.add_argument('--n_balls', type=int, default=5,
                        help='Number of balls in the nbody dataset')
    parser.add_argument('--outf', type=Path, default='results', help='Output folder')
    parser.add_argument('--rollout', type=str2bool, default=True)
    
    # Experiment parameters
    parser.add_argument('--varDT', type=str2bool, default=False, choices=[True, False],)
    
    parser.add_argument('--num_timesteps', type=int, default=10, choices=[2, 5, 10],
                    help='Distance in time between one snaphot an the other.')
    parser.add_argument('--num_inputs', type=int, default=1, choices=[1, 2, 3, 4],
                        help='The number of inputs to give for each prediction step.')
    parser.add_argument('--use_wb', type=str2bool, default=False,
                        help='Use wandb for logging')
    return parser.parse_args()


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)[args.model.upper()]

    print(args)
    args.data_dir = Path(args.data_dir)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    # Common objects
    model_save_path = args.outf / args.exp_name / args.model / f'seed={seed}_n_part={args.n_balls}_n_inputs={args.num_inputs}_varDT={args.varDT}_num_timesteps={args.num_timesteps}.pth'
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Model saved to {model_save_path}')
    early_stopping = EarlyStopping(patience=25, verbose=True, path=model_save_path)
    loss_mse = nn.MSELoss()
    loss_mse_no_red = nn.MSELoss(reduction='none')

    results = {'eval epoch': [], 'val loss': [], 'test loss': [], 'train loss': []}
    best_val_loss = 1e8
    best_epoch = 0

    if args.model == 'segno':
        from SEGNO.nbody.models.model import SEGNO
        from SEGNO.nbody.dataset_nbody import NBodyDataset #from nbody.dataset_nbody import NBodyDataset
        from SEGNO.nbody.train_nbody import run_epoch

        nbody_name = config['other_params']['nbody_name']

        dataset_train = NBodyDataset(args.data_dir, partition='train', dataset_name=nbody_name, dataset=args.dataset,
                                    max_samples=args.max_samples, n_balls=args.n_balls)
        loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

        dataset_val = NBodyDataset(args.data_dir, partition='val', dataset_name=nbody_name, dataset=args.dataset, n_balls=args.n_balls)
        loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

        dataset_test = NBodyDataset(args.data_dir, partition='test', dataset_name=nbody_name, dataset=args.dataset, n_balls=args.n_balls)
        loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)

        params = config['model_params'] | dict(varDT=args.varDT, device=device)
        params['n_inputs'] = args.num_inputs
        # if args.num_inputs > 1:
        #     # All dynamical node_features for each input + the static one
        #     params['in_node_nf'] = (params['in_node_nf'] - 1) * args.num_inputs + 1
        model = SEGNO(**params)
        criterion = [loss_mse,loss_mse_no_red]
        print(args.varDT,args.num_inputs,args.only_test)
    else:
        from EGNO.simulation.dataset_simple import NBodyDynamicsDataset as SimulationDataset
        from EGNO.model.egno import EGNO
        from EGNO.main_simulation_simple_no import run_epoch

        args.varDT = True if args.varDT and args.num_inputs>1 else False

        dataset_train = SimulationDataset(data_dir=args.data_dir, partition='train', max_samples=args.max_samples, dataset=args.dataset, n_balls=args.n_balls, 
                                          num_timesteps=args.num_timesteps,num_inputs=args.num_inputs, varDT=args.varDT) #, num_inputs=args.num_inputs
        loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

        dataset_val = SimulationDataset(data_dir=args.data_dir, partition='val', n_balls=args.n_balls, dataset=args.dataset,
                                        num_timesteps=args.num_timesteps,num_inputs=args.num_inputs, varDT=args.varDT)#num_inputs=args.num_inputs
        loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                                num_workers=0)

        dataset_test = SimulationDataset(data_dir=args.data_dir, partition='test', n_balls=args.n_balls, dataset=args.dataset,
                                         num_timesteps=args.num_timesteps, num_inputs=args.num_inputs, rollout=True, 
                                         traj_len=args.traj_len, varDT= args.varDT)
        loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                                num_workers=0)
        
        params = config['model_params'] | dict(num_timesteps=args.num_timesteps, num_inputs=args.num_inputs, varDT=args.varDT, device=device)
        model = EGNO(**params)
        criterion = loss_mse_no_red
        print(args.rollout,args.num_inputs,args.varDT, args.n_balls)

    optimizer = optim.Adam(model.parameters(), lr=float(config['training_params']['lr']), weight_decay=float(config['training_params']['weight_decay']))

    wandb.init(project="Particle-Physics", entity="egno", config=args, name=model_save_path.stem, mode="online" if args.use_wb else "disabled")
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, optimizer, criterion, epoch, loader_train, args)
        results['train loss'].append(train_loss)
        if (epoch +1) % args.test_interval == 0 or epoch == args.epochs-1:
            val_loss = run_epoch(model, optimizer, criterion, epoch, loader_val, args, backprop=False)
            
            results['eval epoch'].append(epoch)
            results['val loss'].append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
            print("*** Best Val Loss: %.5f \t  Best epoch %d"
                % (best_val_loss, best_epoch))
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early Stopping.")
                break
        
    model.load_state_dict(torch.load(model_save_path, weights_only=False))
    test_loss, trajectories = run_epoch(model, optimizer, criterion, epoch, loader_test, args, backprop=False, rollout=args.rollout)
    results['test loss'].append(test_loss)
        
    json_object = json.dumps(results, indent=4)
    with open(model_save_path.with_suffix('.json'), "w") as outfile:
        outfile.write(json_object)

    torch.save(Data.from_dict(trajectories), model_save_path.parent / f'{model_save_path.stem}_results.pt')
    
    if args.use_wb:
        wandb.log({"train loss": train_loss, "val loss": val_loss, "test loss": test_loss})
        wandb.finish()
    return best_val_loss, test_loss, best_epoch


if __name__ == '__main__':
    args = get_args()
    best_val_loss, test_loss, best_epoch = main(args)
    print(f"Best Val Loss: {best_val_loss}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Test Loss: {test_loss}")