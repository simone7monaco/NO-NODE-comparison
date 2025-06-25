import argparse
from argparse import Namespace
import pickle
import torch
import torch.utils.data
from .simulation.dataset_simple import NBodyDynamicsDataset as SimulationDataset
from .model.egno import EGNO
from .utils import EarlyStopping, cumulative_random_tensor_indices_capped, random_ascending_tensor
from torch_geometric.utils import to_dense_batch
import os
from torch import nn, optim
import json

import random
import numpy as np
import wandb


def get_args():
    parser = argparse.ArgumentParser(description='EGNO')
    parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                        help='how many epochs to wait before logging test')
    parser.add_argument('--outf', type=str, default='exp_results', metavar='N',
                        help='folder to output the json log file')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                        help='learning rate')
    parser.add_argument('--nf', type=int, default=64, metavar='N',
                        help='hidden dim')
    parser.add_argument('--model', type=str, default='egno', metavar='N')
    parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                        help='number of layers for the autoencoder')
    parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                        help='maximum amount of training samples')
    parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                        help='timing experiment')
    parser.add_argument('--data_dir', type=str, default='',
                        help='Data directory.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument("--config_by_file", default=None, nargs="?", const='', type=str, )

    parser.add_argument('--lambda_link', type=float, default=1,
                        help='The weight of the linkage loss.')
    parser.add_argument('--n_cluster', type=int, default=3,
                        help='The number of clusters.')
    parser.add_argument('--flat', action='store_true', default=False,
                        help='flat MLP')
    parser.add_argument('--interaction_layer', type=int, default=3,
                        help='The number of interaction layers per block.')
    parser.add_argument('--pooling_layer', type=int, default=3,
                        help='The number of pooling layers in EGPN.')
    parser.add_argument('--decoder_layer', type=int, default=1,
                        help='The number of decoder layers.')
    parser.add_argument('--norm', action='store_true', default=False,
                        help='Use norm in EGNO')

    parser.add_argument('--varDT', type=bool, default=False,
                        help='The number of inputs to give for each prediction step.')
    parser.add_argument('--rollout', type=bool, default=True,
                        help='The number of inputs to give for each prediction step.')
    parser.add_argument('--variable_deltaT', type=bool, default=False,
                        help='The number of inputs to give for each prediction step.')
    parser.add_argument('--only_test', type=bool, default=True,
                        help='The number of inputs to give for each prediction step.')
    parser.add_argument('--num_inputs', type=int, default=1,
                        help='The number of inputs to give for each prediction step.')
    parser.add_argument('--traj_len', type=int, default=10,
                        help='The lenght of the trajectory in case of rollout.')
    parser.add_argument('--num_timesteps', type=int, default=10,
                        help='The number of time steps.')
    parser.add_argument('--time_emb_dim', type=int, default=32,
                        help='The dimension of time embedding.')
    parser.add_argument('--num_modes', type=int, default=5,
                        help='The number of particles.')
    parser.add_argument('--n_balls', type=int, default=5,
                        help='The number of modes.')
    return parser.parse_args()

time_exp_dic = {'time': 0, 'counter': 0}

# Build the dictionary from the parser arguments
#params = {arg.dest.replace('-', '_'): {'value': arg.default} for arg in parser._actions if arg.dest != 'help'}


def main(config=None):

    #if wandb sweep use config
    # if config is not None:
    #     print("wandb sweep")
    #     args = config
    #     args.cuda = not args.no_cuda and torch.cuda.is_available()
    #     device = torch.device("cuda" if args.cuda else "cpu")

    print(args)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    varDt = True if args.varDT and args.num_inputs>1 else False
    
    dataset_train = SimulationDataset(partition='train', max_samples=args.max_training_samples,
                                      data_dir=args.data_dir,n_balls=args.n_balls, num_timesteps=args.num_timesteps,num_inputs=args.num_inputs, varDT=varDt) #, num_inputs=args.num_inputs
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                               num_workers=0)

    dataset_val = SimulationDataset(partition='val',
                                    data_dir=args.data_dir, n_balls=args.n_balls, num_timesteps=args.num_timesteps,num_inputs=args.num_inputs, varDT=varDt)#num_inputs=args.num_inputs
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                             num_workers=0)

    dataset_test = SimulationDataset(partition='test',data_dir=args.data_dir, n_balls=args.n_balls, num_timesteps=args.num_timesteps, 
                                num_inputs=args.num_inputs, traj_len=args.traj_len, varDT= varDt)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                              num_workers=0)
    
    if args.model == 'egno':
        model = EGNO(n_layers=args.n_layers, in_node_nf=1, in_edge_nf=2, hidden_nf=args.nf, device=device,
                     with_v=True, flat=args.flat, activation=nn.SiLU(), norm=args.norm, use_time_conv=True,
                     num_modes=args.num_modes, num_timesteps=args.num_timesteps, time_emb_dim=args.time_emb_dim, num_inputs=args.num_inputs, varDT=args.varDT)
    else:
        raise NotImplementedError('Unknown model:', args.model)

    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model_save_path = args.outf + '/' + args.exp_name + '/' + 'saved_model'+"_seed="+str(seed)+"_n_part="+str(args.n_balls)+"_n_inputs="+str(args.num_inputs)+"_varDT="+str(varDt)+"_num_timesteps="+str(args.num_timesteps)+"_n_layers="+str(args.n_layers)+"_lr="+str(args.lr)+"_wd="+str(args.weight_decay)+'.pth'
    print(f'Model saved to {model_save_path}')
    early_stopping = EarlyStopping(patience=25, verbose=True, path=model_save_path)

    results = {'eval epoch': [], 'val loss': [], 'test loss': [], 'train loss': [],'traj_loss':[]}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best_train_loss = 1e8
    
    for epoch in range(0, args.epochs):
        train_loss = run_epoch(model, optimizer, epoch, loader_train,args)
        results['train loss'].append(train_loss)
        if (epoch +1) % args.test_interval == 0:
            val_loss = run_epoch(model, optimizer, epoch, loader_val,args, backprop=False)

            results['eval epoch'].append(epoch)
            results['val loss'].append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_epoch = epoch
                # Save model is move to early stopping.
            print("*** Best Val Loss: %.5f \t  Best epoch %d"
                  % (best_val_loss, best_epoch))
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early Stopping.")
                break
                
    model = EGNO(n_layers=args.n_layers, in_node_nf=1, in_edge_nf=2, hidden_nf=args.nf, device=device,
                     with_v=True, flat=args.flat, activation=nn.SiLU(), norm=args.norm, use_time_conv=True,
                     num_modes=args.num_modes, num_timesteps=args.num_timesteps, time_emb_dim=args.time_emb_dim, num_inputs=args.num_inputs, varDT=args.varDT)# Create a new instance of the model
    model.load_state_dict(torch.load(model_save_path, weights_only=False))

    test_loss, avg_num_steps, losses, trajectories = run_epoch(model, optimizer, epoch, loader_test, args, backprop=False)
    results['test loss'].append(test_loss)
    results['traj_loss'].append(losses)
    json_object = json.dumps(results, indent=4)

    with open(args.outf + "/" + args.exp_name + "/loss"+"_seed="+str(seed)+"_n_part="+str(args.n_balls)+"_n_inputs="+str(args.num_inputs)+"_varDT="+str(varDt)+"_num_timesteps="+str(args.num_timesteps)+"_n_layers="+str(args.n_layers)+"_lr="+str(args.lr)+"_wd="+str(args.weight_decay)+"_.json", "w") as outfile:
        outfile.write(json_object)

    #save trajectories in pickle
    with open(args.outf + "/" + args.exp_name + "/trajectories"+"_seed="+str(seed)+"_n_part="+str(args.n_balls)+"_n_inputs="+str(args.num_inputs)+"_varDT="+str(varDt)+"_num_timesteps="+str(args.num_timesteps)+"_n_layers="+str(args.n_layers)+"_lr="+str(args.lr)+"_wd="+str(args.weight_decay)+"_.pkl", "wb") as f:
        pickle.dump(trajectories, f)

    return best_train_loss, best_val_loss, best_test_loss, best_epoch


def run_epoch(model, optimizer, criterion, epoch, loader, args, backprop=True, rollout=False, num_timesteps=10, only_first=False):
    # if only_first: returns the loss only for the first predicted step (ignored if rollout is True or num_inputs > 1)
    dT = loader.dataset.dT
    device = args.device
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch,'losses': [], 'loss': 0,"tot_num_steps": 0,"avg_num_steps": 0, 'counter': 0, 'lp_loss': 0}
    
    #print(f"this is the {loader.dataset.partition} partition")
    if rollout:
        first = True ## 0: target, 1: prediction
    
    for batch_idx, data in enumerate(loader):
        data = [d.to(device) for d in data]
        loc, vel, edge_attr, charges, loc_true, out_indices = data #loc_true.shape:[B, N, T, 3]
         #loc.shape : [B, num_inputs, N, 3], edge_attr.shape: [B, num_inputs*N, 1]
        
        n_nodes = args.n_balls
        if backprop:
            optimizer.zero_grad()
        
        timesteps = torch.arange(args.num_timesteps).to(device) * dT
        if args.num_inputs > 1 : #and rollout
            
            start = 30
            loc = loc.transpose(0,1) #T,B,N,3
            vel = vel.transpose(0,1)
            
            if args.varDT:
                timesteps = random_ascending_tensor(length=args.num_inputs, max_value=num_timesteps-1).to(device)
                loc = loc[start + timesteps]
                vel = vel[start + timesteps]
            
            batch_size = loc.shape[1]
            edges = loader.dataset.get_edges(batch_size, n_nodes)
            edges = [edges[0].to(device), edges[1].to(device)]
            rows, cols = edges
            
            edge_attr_o = edge_attr.view(-1, edge_attr.shape[-1])

            loc_inputs = []
            vel_inputs = []
            loc_mean = []
            edge_attrs = []
            nodes = []
            
            for i in range(args.num_inputs):
                
                loc_inputs.append(loc[i].reshape(-1, loc[i].shape[-1]))
                loc_mean.append(loc[i].mean(dim=1, keepdim=True).repeat(1, n_nodes, 1).reshape(-1, loc[i].size(2)))
                vel_inputs.append(vel[i].reshape(-1, vel[i].shape[-1]))
                loc_dist = torch.sum((loc[i].reshape(-1, loc[i].shape[-1])[rows] - loc[i].reshape(-1, loc[i].shape[-1])[cols])**2, 1).unsqueeze(1)
                edge_attrs.append(torch.cat([edge_attr_o, loc_dist], 1).detach())
                nodes.append(torch.sqrt(torch.sum(vel[i].reshape(-1, vel[i].shape[-1]) ** 2, dim=1)).unsqueeze(1).detach())

            loc = torch.stack(loc_inputs)
            vel = torch.stack(vel_inputs)
            edge_attr = torch.stack(edge_attrs)
            nodes = torch.stack(nodes) # num_inputs, BN, 1
            if charges is not None:
                nodes = torch.cat([nodes, charges.reshape(-1, charges.shape[-1]).unsqueeze(0).repeat(args.num_inputs, 1, 1)], dim=-1)
                
            loc_mean = torch.stack(loc_mean)
        else:
            loc_mean = loc.mean(dim=1, keepdim=True).repeat(1, n_nodes, 1).view(-1, loc.size(2))  # [BN, 3]

            loc = loc.view(-1, loc.shape[-1])
            vel = vel.view(-1, vel.shape[-1])
            
            batch_size = loc.shape[0] // n_nodes
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            if charges is not None:
                nodes = torch.cat([nodes, charges.view(-1, 1)], dim=1)
            edges = loader.dataset.get_edges(batch_size, n_nodes)
            edges = [edges[0].to(device), edges[1].to(device)]

            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr_o = edge_attr.view(-1, edge_attr.shape[-1])
            edge_attr = torch.cat([edge_attr_o, loc_dist], 1).detach()  # concatenate all edge properties
        
        
        if rollout:
            traj_len = args.traj_len
            
            #print(locs_true.shape)
            if False: #variable_deltaT
                #print(loc_true.shape) #[100, 519, 5, 3]
                loc_true = loc_true.transpose(0,1).reshape(-1, batch_size*n_nodes, 3) #[519, 500, 3]
                #print(loc_true.shape)
                start = 30
                locs_pred, steps = rollout_fn(model, nodes, loc, edges, vel, edge_attr_o, edge_attr,loc_mean, n_nodes, traj_len, batch_size,
                                                charges=charges, variable_deltaT=args.variable_deltaT)
                locs_pred = locs_pred.to(device) # (T,BN,3)
                end = steps[-1] + start
                
                locs_true = loc_true[start:end] #.view(batch_size * n_nodes, steps[-1], 3).transpose(0, 1)
                
            else:
                locs_pred, energies, energies_allsteps = rollout_fn(model, nodes, loc, edges, vel, edge_attr_o, edge_attr,loc_mean, n_nodes, traj_len, batch_size,
                                                                    charges=charges, num_steps=num_timesteps, timesteps=timesteps, 
                                                                    energy_fun=loader.dataset.energy_fun)
                locs_pred = locs_pred.to(device)
                locs_true = loc_true.view(batch_size * n_nodes, num_timesteps*traj_len, 3).transpose(0, 1)

            corr, avg_num_steps, first_invalid_idx = pearson_correlation_batch(locs_pred, locs_true, n_nodes) #locs_pred[::10]
            #print(first_invalid_idx)
            num_elements = int(0.4 * args.traj_len*num_timesteps)  # Calculate 40% of the total elements
            if args.traj_len*num_timesteps >= 50:
                num_elements = 20
            
            sup =  num_elements #first_invalid_idx if first_invalid_idx > 15 else
            
            locs_pred = locs_pred[:sup]
            locs_true = locs_true[:sup]
            energies_allsteps = energies_allsteps[:sup]
            #print(torch.isnan(locs_pred).any(), torch.isinf(locs_pred).any())

            # print("check reshape:")
            # print(torch.sum(locs_pred-locs_true))
            # shape at this moment of locs:(T, B*N,3)
            batch = torch.arange(batch_size).repeat_interleave(n_nodes).to(locs_pred.device)  # [BN]
            targets = to_dense_batch(locs_true.permute(1,0,2), batch)[0].permute(0, 2, 1, 3) # (B, T, N, 3)
            preds = to_dense_batch(locs_pred.permute(1,0,2), batch)[0].permute(0, 2, 1, 3) # (B, T, N, 3)
            energies_allsteps = energies_allsteps.permute(1,0,2) # (B, T, 1)
            if first:
                traj_targ = targets
                traj_pred = preds
                traj_energies = energies_allsteps
                first = False
            else:
                traj_targ = torch.cat((traj_targ, targets), dim=0)
                traj_pred = torch.cat((traj_pred, preds), dim=0)
                traj_energies = torch.cat((traj_energies, energies_allsteps), dim=0)
            
            # print(torch.sum(locs_pred-locs_true))
            # print("checked")

            res["tot_num_steps"] += avg_num_steps*batch_size # TODO: take inspiration
            
            #loss with metric (A-MSE)
            losses = criterion(locs_pred, locs_true).view(sup, batch_size * n_nodes, 3) #num_timesteps*traj_len
            losses = torch.mean(losses, dim=(1, 2))
            loss = torch.mean(losses) 
            res['losses'].append(losses.cpu().tolist())
        else:
            loc_end = loc_true.view(batch_size * n_nodes, num_timesteps, 3).transpose(0, 1).contiguous().view(-1, 3)
            loc_pred, vel_pred, _ = model(loc, nodes, edges, edge_attr, v=vel, loc_mean=loc_mean, timesteps_out=timesteps)
            #pearson_correlation_batch(loc_pred.reshape(num_timesteps,batch_size * n_nodes, 3),loc_end,n_nodes)
            losses = criterion(loc_pred, loc_end).view(num_timesteps, batch_size * n_nodes, 3)
            losses = torch.mean(losses, dim=(1, 2))
            
            loss = losses[0] if only_first else losses.mean()
    
        if backprop:
            loss.backward()
            optimizer.step()
        if rollout:
            res['loss'] += loss.item() * batch_size
        else:
            if only_first:
                res['loss'] += loss.item() * batch_size
            else:
                res['loss'] += losses[-1].item() * batch_size
        res['counter'] += batch_size
        res["avg_num_steps"] = res["tot_num_steps"] / res["counter"]
    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f avg lploss: %.5f'
          % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter'], res['lp_loss'] / res['counter']))
    
    avg_loss = res['loss'] / res['counter']
    

    if rollout:
        wandb.log({f"{loader.dataset.partition}_loss": avg_loss,"avg_num_steps": res['avg_num_steps']}, step=epoch)
        return avg_loss, {'targets': traj_targ, 'preds': traj_pred, 'energy_conservation': traj_energies, 'test_loss': avg_loss}
        # torch.stack((traj_targ,traj_pred), dim=0)
    else:
        wandb.log({f"{loader.dataset.partition}_loss": avg_loss}, step=epoch)
    return avg_loss


@torch.no_grad()
def rollout_fn(model, nodes, loc, edges, v, edge_attr_o, edge_attr, 
               loc_mean, n_nodes, traj_len, batch_size, charges=None,
               num_steps=10,variable_deltaT=False, timesteps=None, 
               energy_fun=None):
    
    vel = v
    BN = batch_size*n_nodes
    batch = torch.arange(batch_size).repeat_interleave(n_nodes).to(loc.device)  # [BN]
    if variable_deltaT:
    #   calculate random indices
        steps, steps_size = cumulative_random_tensor_indices_capped(N=traj_len,start=1,end=num_steps+3, MAX=num_steps*traj_len)
        tot_num_step = steps[-1] 
        #change shape of loc preds
        loc_preds = torch.zeros((tot_num_step*BN,3))
    # -> pass to egno at each cicle the respective number of steps for that iter
        #print(loc_preds.shape,steps,steps_size)
    else:
        loc_preds = torch.zeros((traj_len,batch_size*num_steps*n_nodes,3))

    energies = []
    energies_allsteps = []
    for i in range(traj_len):
        #print("Inside loop \n")
        
        if variable_deltaT:
            raise NotImplementedError("It should not be used with rollout")
            #print(i,steps[i],steps_size[i])
            if i == 0:
                loc, vel, _ = model(loc.detach(), nodes, edges, edge_attr,v=vel.detach(), loc_mean=loc_mean, num_timesteps=steps_size[i])
                loc_preds[:steps[i]*BN] = loc
                
            else:
                loc, vel, _ = model(loc.detach(), nodes, edges, edge_attr,v=vel.detach(), loc_mean=loc_mean, num_timesteps=steps_size[i])
                loc_preds[steps[i-1]*BN:steps[i]*BN] = loc

            loc = loc.view(steps_size[i],-1, loc.shape[-1])[-1] #get last element in the inner trajectory
            vel = vel.view(steps_size[i], -1, vel.shape[-1])[-1] 
        else:

            loc, vel, _ = model(loc, nodes, edges, edge_attr,v=vel, loc_mean=loc_mean, timesteps_out=timesteps)
            # timesteps=None
            loc_preds[i] = loc
            loc_all = loc.view(num_steps,-1, loc.shape[-1])  #shape: [num_steps, BN, 3]
            vel_all = vel.view(num_steps, -1, vel.shape[-1]) #shape: [num_steps, BN, 3]
            loc = loc_all[-1]  #get last element in the inner trajectory
            vel = vel_all[-1] #get last element in the inner trajectory
            
        
        nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1)
        if charges is not None:
            nodes = torch.cat([nodes, charges.view(-1, 1)], dim=1)
        rows, cols = edges
        loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
        edge_attr = torch.cat([edge_attr_o, loc_dist], 1)  # concatenate all edge properties
        loc = loc.view(-1, n_nodes, loc.shape[-1])
        loc_mean = loc.mean(dim=1, keepdim=True).repeat(1, n_nodes, 1).view(-1, loc.size(2))
        loc = loc.view(-1, loc.shape[-1])
    
        if energy_fun is not None:
            for j in range(num_steps):
                en = energy_fun(loc_all[j], vel_all[j], nodes[:, -1:], batch=batch)
                energies_allsteps.append(en)
                if j == num_steps-1:
                    energies.append(en)
    
    energies = torch.tensor(np.stack(energies)).unsqueeze(-1) if energy_fun is not None else None
    energies_allsteps = torch.tensor(np.stack(energies_allsteps)).unsqueeze(-1) if energy_fun is not None else None
    # print("\n outside loop \n")
    if not variable_deltaT:
        loc_preds = loc_preds.reshape(traj_len*num_steps, -1, 3)
        return loc_preds, energies, energies_allsteps
    else:
        loc_preds = loc_preds.reshape(tot_num_step, -1, 3)
        return loc_preds, steps, energies, energies_allsteps
    
    

def pearson_correlation_batch(x, y, N):
    """
    Compute the Pearson correlation for each time step (T) in each batch (B).
    
    Args:
    - x: Tensor of shape (T, B*N, 3), predicted states.
    - y: Tensor of shape (T, B*N, 3), ground truth states.
    
    Returns:
    - correlations: Tensor of shape (B, T), Pearson correlation for each time step in each batch.
    """
    
    # Reshape to (B, T, N*3) 
    
    T = x.shape[0] 
    cut = int(0.4 * T)  # Calculate 40% of the total elements to avoid NaN values
    B = x.size(1) // N
    x = x.reshape( T, B, -1)[:cut].transpose(0,1)  # Flatten N and 3 into a single dimension
    y = y.reshape( T, B, -1)[:cut].transpose(0,1)
    
    
    # Mean subtraction
    mean_x = x.mean(dim=2, keepdim=True)
    mean_y = y.mean(dim=2, keepdim=True)
    
    xm = x - mean_x
    ym = y - mean_y

    # Compute covariance between x and y along the flattened dimensions
    covariance = (xm * ym).sum(dim=2)

    # Compute standard deviations along the flattened dimensions
    std_x = torch.sqrt((xm ** 2).sum(dim=2))
    std_y = torch.sqrt((ym ** 2).sum(dim=2))

    # Compute Pearson correlation for each sample in the batch
    correlation = covariance / (std_x * std_y)

    #number of steps before reaching a value of correlation, between prediction and ground truth for each timesteps, lower than 0.5
    num_steps_batch = []

    for i in range(correlation.shape[0]):
        
        if any(correlation[i] < 0.5):
            num_steps_before = (correlation[i] < 0.5).nonzero(as_tuple=True)[0][0].item()
            
        else:
            num_steps_before = cut
        num_steps_batch.append(num_steps_before)

    # Check if all values along B dimension are >= 0.5 for each T
    mask = torch.all(correlation >= 0.5, dim=0)

    # Convert the boolean mask to int for argmax
    first_failure_index = torch.argmax(~mask.int()).item()
    #print(first_failure_index,torch.mean(torch.Tensor(num_steps_batch)),correlation[0])
    # If no failures, return the number of columns as the "end"
    if mask.all():
        first_failure_index = correlation.size(1)       
    #print("first invalid")
    #print(first_failure_index,torch.mean(torch.Tensor(num_steps_batch)))
    #exit()
    #return the average (in the batch) number of steps before reaching a value of correlation lower than 0.5
    #return the minimum first index along T dimension after which correlation drops below the threshold                                 
    return correlation, torch.mean(torch.Tensor(num_steps_batch)), first_failure_index 
 

if __name__ == "__main__":
    args = get_args()
    if args.config_by_file is not None:
        if len(args.config_by_file) == 0:
            job_param_path = './configs/config_simulation_simple_no.json'
        else:
            job_param_path = args.config_by_file
        with open(job_param_path, 'r') as f:
            hyper_params = json.load(f)
            # Only update existing keys
            args = vars(args)
            args.update((k, v) for k, v in hyper_params.items() if k in args)
            args = Namespace(**args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()



    device = torch.device("cuda" if args.cuda else "cpu")
    loss_mse = nn.MSELoss(reduction='none')

    print(args)
    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    try:
        os.makedirs(args.outf + "/" + args.exp_name)
    except OSError:
        pass

    varDt = False

    best_train_loss, best_val_loss, best_test_loss, best_epoch = main()
    print("best_train = %.6f" % best_train_loss)
    print("best_val = %.6f" % best_val_loss)
    print("best_test = %.6f" % best_test_loss)
    print("best_epoch = %d" % best_epoch)
    print("best_train = %.6f, best_val = %.6f, best_test = %.6f, best_epoch = %d"
          % (best_train_loss, best_val_loss, best_test_loss, best_epoch))

    #wandb.finish()
