import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import time
from utils import pearson_correlation_batch, cumulative_random_tensor_indices_capped, random_ascending_tensor
from torch_geometric.utils import to_dense_batch


import wandb

time_exp_dic = {'time': 0, 'counter': 0}



def train(device, model, interval, args):
    from dataset_nbody import NBodyDataset
    args.device = device

    dataset_train = NBodyDataset(root=args.data_path, partition='train', dataset_size=args.dataset_size,
                                 max_samples=args.max_samples)
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=0, shuffle=True, drop_last=True)

    dataset_val = NBodyDataset(root=args.data_path, partition='val', dataset_size=args.dataset_size)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=0, shuffle=False, drop_last=False)

    dataset_test = NBodyDataset(root=args.data_path, partition='test', dataset_size=args.dataset_size)
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=0, shuffle=False, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_mse = nn.MSELoss()


    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best = {'long_loss': {}}
    for epoch in range(0, args.epochs):
        train_loss, _ = run_epoch(model, optimizer, loss_mse, epoch, loader_train, args=args)
        if (epoch % args.test_interval == 0 or epoch == args.epochs-1) and epoch > 0:
            val_loss, res = run_epoch(model, optimizer, loss_mse, epoch, loader_val, device, args=args, backprop=False)
            test_loss, res = run_epoch(model, optimizer, loss_mse, epoch, loader_test,
                                  device, args=args, backprop=False)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_epoch = epoch
                best = res
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d" %
                  (best_val_loss, best_test_loss, best_epoch))


    return best_val_loss, best_test_loss, best_epoch


# def run_epoch(model, optimizer, criterion, epoch, loader, interval, device, args, backprop=True):
def run_epoch(model, optimizer, criterion, epoch, loader, args, backprop=True, rollout=False, num_timesteps=10, **kwargs):
    
    device = args.device
    varDt = getattr(args, 'varDT', False)
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'losses': [], "tot_num_steps": 0,"avg_num_steps": 0, 'counter': 0, 'long_loss': {}}

    if isinstance(criterion, tuple):
        criterion, loss_mse_no_red = criterion[0], criterion[1]
    n_nodes = loader.dataset.n_balls
    batch_size = args.batch_size
    batch = torch.arange(0, batch_size).repeat_interleave(n_nodes).long().to(device)
    if rollout:
        first = True

    edges = loader.dataset.get_edges(args.batch_size, n_nodes)
    edges = [edges[0], edges[1]]
    rows, cols = edges
    edge_index = torch.stack(edges)

    for batch_idx, data in enumerate(loader):
        start = loader.dataset.start
        data = [d.to(device) for d in data]
        for i in range(len(data)):
            if len(data[i].shape) == 4:
                data[i] = data[i].transpose(0, 1).contiguous()
                data[i] = data[i].view(data[i].size(0), -1, data[i].size(-1))
            else:
                data[i] = data[i][:, :data[i].size(1), :].contiguous()
                data[i] = data[i].view(-1, data[i].size(-1))  

        locs, vels, edge_attr, charges = data
        prod_charges = charges[rows] * charges[cols]

        T = num_timesteps # TODO: set eventually here a variable number of steps for output too

        if args.num_inputs > 1:
            steps = None
            if varDt: # pass steps to rollout to call the model at each iter with the corerct T
                steps = (np.random.randint(1, num_timesteps// args.num_inputs, size=args.num_inputs-1)).tolist()
            else:
                steps = [num_timesteps // args.num_inputs for _ in range(args.num_inputs-1)]
            indices = np.flip(start - np.cumsum([0] + steps))
            if (indices < 0).any():
                # push to the first frame
                indices += -indices.min() # translate to positive indices (and start as well)
                start = indices.min()
            indices = indices.copy()

            loc = locs[indices].transpose(0, 1).contiguous() # BN, T, 3
            vel = vels[indices].transpose(0, 1).contiguous()
            end = indices[-1] + T 
            loc_end = locs[end]
            in_steps = torch.tensor(indices - start).int().to(device) 
            h = torch.sqrt(torch.sum(vel ** 2, dim=-1)).unsqueeze(-1) # (BN, T, 1)
            loc_dist = torch.sum((loc[rows, -1, :] - loc[cols, -1, :])**2, 1).unsqueeze(1)  # relative distances among LAST locations
        else:
            in_steps = None
            end = start + T
            loc, loc_end, vel = locs[start], locs[end], vels[start]
            h = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(-1).detach()
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
        edge_attr = torch.cat([prod_charges, loc_dist], 1).detach()  # concatenate all edge properties

        # if args.time_exp:
        #     torch.cuda.synchronize()
        #     t1 = time.time()
        if backprop:
            optimizer.zero_grad()

        if rollout: # rollout mode
            # if varDt: # TODO: pass variable steps to the rollout function
            #     T = (np.random.randint(num_timesteps//2, num_timesteps, size=args.traj_len-1)).tolist()
            # else:
            T = [num_timesteps for _ in range(args.traj_len)]

            pred_indices = start + np.cumsum(T)
            pred_indices = pred_indices[pred_indices < locs.size(0)]
            locs_end = locs[pred_indices].to(device) # (T, BN, 3)

            locs_pred, energies = rollout_fn(model, h, loc, edge_index, vel, edge_attr, batch, 
                                             len(pred_indices), num_steps=T, num_prev=args.num_inputs, charges=charges,
                                             energy_fun=loader.dataset.energy_fun, in_steps=in_steps)
            #locs_pred shape: [T, BN, 3], energy shape: [T, B, 1]

            corr, avg_num_steps, first_invalid_idx = pearson_correlation_batch(locs_pred, locs_end, n_nodes)
            res["tot_num_steps"] += avg_num_steps*batch_size
            
            targets = to_dense_batch(locs_end.permute(1,0,2), batch)[0].permute(0, 2, 1, 3) # (B, T, N, 3)
            preds = to_dense_batch(locs_pred.permute(1,0,2), batch)[0].permute(0, 2, 1, 3) # (B, T, N, 3)
            energies = energies.permute(1,0,2) # (B, T, 1)
            if first:
                traj_targ = targets
                traj_pred = preds
                traj_energies = energies
                first = False
            else:
                traj_targ = torch.cat((traj_targ, targets), dim=0)
                traj_pred = torch.cat((traj_pred, preds), dim=0)
                traj_energies = torch.cat((traj_energies, energies), dim=0)

            #loss with metric (A-MSE)
            losses = loss_mse_no_red(locs_pred, locs_end).view(-1, batch_size * n_nodes, 3)
            losses = torch.mean(losses, dim=(1, 2))
            loss = torch.mean(losses)
            res['losses'].append(losses.cpu().tolist())
        else:
            loc_pred, h, vel = model(h, loc.detach(), edges, vel.detach(), edge_attr, T=T, in_steps=in_steps)
            loss = criterion(loc_pred, loc_end)

        # if args.time_exp:
        #     torch.cuda.synchronize()
        #     t2 = time.time()
        #     time_exp_dic['time'] += t2 - t1
        #     time_exp_dic['counter'] += 1

        if backprop:    
            loss.backward()
            optimizer.step()
        
        wandb.log({f'{loader.dataset.partition}/loss': loss.item()})
        res['loss'] += loss.item()*batch_size
        res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    avg_loss = res['loss'] / res['counter']
    print('%s epoch %d avg loss: %.5f' % (prefix+loader.dataset.partition, epoch, avg_loss))
    if rollout:
        return avg_loss, {'targets': traj_targ, 
                          'preds': traj_pred, 
                          'energies': traj_energies,
                          'pred_indices': pred_indices,} | res
    return avg_loss#, res



@torch.no_grad()
def rollout_fn(model, h, loc, edge_index, vel, edge_attr, batch, 
               traj_len,num_steps=10, num_prev=1, charges=None,
               energy_fun=None, in_steps=None):

    rows, cols = edge_index
    prod_charges = charges[rows] * charges[cols] if charges is not None else None
    loc_preds = torch.zeros((traj_len,loc.shape[0],loc.shape[-1])).to(loc) # (T, BN, 3)
    energies = []
    if isinstance(num_steps, list):
        assert len(num_steps) == traj_len, "num_steps should be a list of length traj_len"
    for i in range(traj_len):
        T = num_steps[i] if isinstance(num_steps, list) else num_steps
        loc_p, _, vel_p = model(h, loc, edge_index, vel, edge_attr, T=T, in_steps=in_steps)
        
        if energy_fun is not None:
            energies.append(energy_fun(loc_p, vel_p, charges, batch=batch)) # TODO: matrix or node vector?
        
        loc_preds[i] = loc_p

        if num_prev > 1:
            # assume loc to have shape (BN, T, 3), remove the first T and concat the predicted loc, same for vel
            loc = torch.cat((loc[:, 1:, :], loc_p.unsqueeze(1)), dim=1)  # (BN, T, 3)
            vel = torch.cat((vel[:, 1:, :], vel_p.unsqueeze(1)), dim=1)  # (BN, T, 3)
            h = torch.sqrt(torch.sum(vel ** 2, dim=-1)).unsqueeze(-1) # (BN, T, 1)
            loc_dist = torch.sum((loc[rows, -1, :] - loc[cols, -1, :])**2, 1).unsqueeze(1)  # relative distances among LAST locations
            in_steps = torch.tensor(in_steps.tolist()[1:] + [T]).to(in_steps) - T
        else:
            loc = loc_p  # predicted
            vel = vel_p
            h = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            
        edge_attr = torch.cat([prod_charges, loc_dist], 1).detach()     

    energies = torch.tensor(np.stack(energies)).unsqueeze(-1) if energy_fun is not None else None
    return loc_preds, energies