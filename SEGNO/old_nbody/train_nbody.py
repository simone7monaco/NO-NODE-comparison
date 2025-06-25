import numpy as np
import torch
import os
from torch import nn, optim
from .models.model import SEGNO
from torch_geometric.nn import knn_graph
from .dataset_nbody import NBodyDataset #from nbody.dataset_nbody import NBodyDataset
import json
import wandb 
from torch_geometric.utils import to_dense_batch

time_exp_dic = {'time': 0, 'counter': 0}

torch.manual_seed(40)

def cumulative_random_tensor_indices(n, start, end):
    # Generate the cumulative numpy array as before
    random_array = np.random.randint(start, end, size=n)
    #print(random_array)
    cumulative_array = np.cumsum(random_array)
    #print(cumulative_array)
    
    # Convert the cumulative numpy array to a PyTorch tensor
    cumulative_tensor = torch.tensor(cumulative_array, dtype=torch.long)
    
    return cumulative_tensor, torch.tensor(random_array)

def cumulative_random_tensor_indices_capped(N, start, end, MAX=100):
    """
    Generates a random integer tensor and adjusts it so that its cumulative sum equals MAX.
    
    Args:
    - N (int): Length of the tensor.
    - start (int): Minimum value for random integers (inclusive).
    - end (int): Maximum value for random integers (exclusive).
    - MAX (int): Desired cumulative sum target (default is 100).
    
    Returns:
    - torch.Tensor: The adjusted random tensor.
    - torch.Tensor: The cumulative sum of the adjusted random tensor.
    """
    # Step 1: Generate a random integer tensor of size N within [start, end)
    random_array = torch.randint(start, end, (N,))
    
    # Step 2: Calculate the initial sum and scale values to approach MAX
    initial_sum = random_array.sum().item()
    
    # If initial sum is zero, reinitialize random_array to avoid division by zero
    while initial_sum == 0:
        random_array = torch.randint(start, end, (N,))
        initial_sum = random_array.sum().item()

    # Scale values to approximate the sum to MAX
    scaled_array = torch.round((random_array.float() / initial_sum) * MAX).int()

    # Step 3: Correct any rounding difference to ensure sum equals MAX
    diff = MAX - scaled_array.sum().item()
    
    if diff != 0:
        # Randomly adjust elements to make the sum exactly MAX
        indices = torch.randperm(N)
        for i in indices:
            # Ensure values stay within the [start, end) range after adjustment
            if start <= scaled_array[i] + diff < end:
                scaled_array[i] += diff
                break  # Exit once sum is corrected
    
    # Step 4: Calculate cumulative sum tensor
    cumulative_tensor = torch.cumsum(scaled_array, dim=0)

    return  cumulative_tensor,scaled_array

# varDt = False

def train(gpu, args):
    if args.gpus == 0:
        device = 'cpu'
    else:
        device = torch.device('cuda:' + str(gpu))
    
    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    try:
        os.makedirs(args.outf + "/" + args.exp_name)
    except OSError:
        pass

    varDt = args.varDT#True if args.varDT and args.num_inputs>1 else False #fix
    print(args)
    model = SEGNO(in_node_nf=1, in_edge_nf=1, hidden_nf=64, device=device, n_layers=args.layers,
                         recurrent=True, norm_diff=False, tanh=False, use_previous_state=args.num_inputs, variableDT=args.varDT)

    dataset_train = NBodyDataset(partition='train', dataset_name=args.nbody_name,
                                 max_samples=args.max_samples, n_balls=args.n_balls)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    dataset_val = NBodyDataset(partition='val', dataset_name=args.nbody_name,n_balls=args.n_balls)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

    dataset_test = NBodyDataset(partition='test', dataset_name=args.nbody_name,n_balls=args.n_balls)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_mse = nn.MSELoss()
    loss_mse_no_red = nn.MSELoss(reduction='none')
    results = {'eval epoch': [], 'val loss': [], 'test loss': [], 'train loss': [],'traj_loss':[]}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best = {'long_loss': {}}
    print(args.varDT,args.num_inputs,args.only_test)
    for epoch in range(0, args.epochs):
        train_loss, _ = run_epoch(model, optimizer, [loss_mse,loss_mse_no_red], epoch, loader_train, device, args, use_previous_state=args.use_previous_state)
        results['train loss'].append(train_loss)
        if (epoch+1) % args.test_interval == 0 or epoch == args.epochs-1:

            val_loss, res = run_epoch(model, optimizer, [loss_mse,loss_mse_no_red], epoch, loader_val, device, args, backprop=False,use_previous_state=args.use_previous_state)
            test_loss, res = run_epoch(model, optimizer, [loss_mse,loss_mse_no_red], epoch, loader_test,
                                  device, args, backprop=False,rollout=True)
            
            results['eval epoch'].append(epoch)
            results['val loss'].append(val_loss)
            results['test loss'].append(test_loss)
            results['traj_loss'].append(res['losses'])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_epoch = epoch
                best = res
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best Test avg num steps: %.4f \t Best epoch %d" %
                  (best_val_loss, best_test_loss, best["avg_num_steps"], best_epoch))
            # print(best['long_loss'])
    
    json_object = json.dumps(results, indent=4)
    with open(args.outf + "/" + args.exp_name + "/loss"+"_seed="+str(args.seed)+"_n_part="+str(args.n_balls)+"_n_steps="+str(args.num_timesteps)+"_n_inputs="+str(args.num_inputs)+"_varDT="+str(varDt)+"_lr"+str(args.lr)+"_wd"+str(args.weight_decay)+"_onlytest="+str(args.only_test)+"_.json", "w") as outfile:
        outfile.write(json_object)

    # traj_losses = torch.stack(best['losses'], dim=0)
    # torch.save(traj_losses,'traj_losses.pt')

    return best_val_loss, best_test_loss, best_epoch


def run_epoch(model, optimizer, criterion, epoch, loader, args, backprop=True, rollout=False, num_timesteps=10, **kwargs):
    device = args.device
    varDt = args.varDT
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'losses': [], "tot_num_steps": 0,"avg_num_steps": 0, 'counter': 0, 'long_loss': {}}
    criterion, loss_mse_no_red = criterion[0], criterion[1]
    n_nodes = args.n_balls
    if rollout:
        first = True

    for batch_idx, data in enumerate(loader):
        data = [d.to(device) for d in data]
        for i in range(len(data)):
            if len(data[i].shape) == 4:
                
                data[i] = data[i].transpose(0, 1).contiguous()
                
                data[i] = data[i].view(data[i].size(0), -1, data[i].size(-1))
                
            else:
                
                data[i] = data[i][:, :data[i].size(1), :].contiguous()
                
                data[i] = data[i].view(-1, data[i].size(-1))  
                

        locs, vels = data   #locs shape: [519, 500, 3] (T,BN,3)
        start = loader.dataset.start # 30 for charged small
        if locs.shape[2] > 3:
            h_nodes = locs[0, :, 3:] # node features (charges, masses, etc.)
            locs = locs[:, :, :3]
        else:
            h_nodes = None
        loc, loc_end, vel = locs[start], locs[start+num_timesteps], vels[start]

        #print(loc.shape)
        batch_size = loc.shape[0] // loader.dataset.n_balls
        batch = torch.arange(0, batch_size).repeat_interleave(n_nodes).long().to(device)
        
        edge_index = knn_graph(loc, 4, batch) # Considers positions only for edge index
        #print(f"edge index shape :{edge_index.shape}")
        h = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
        if h_nodes is not None:
            h = torch.cat((h, h_nodes), dim=1).detach()
        rows, cols = edge_index
        loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
        edge_attr = loc_dist.detach()
        
        if rollout: 
            num_prev = args.num_inputs

            if varDt: 
                #pass steps to rollout to call the model at each iter with the corerct T
                _, steps = cumulative_random_tensor_indices_capped(N=args.traj_len + num_prev - 1,start=1,end=num_timesteps+3, MAX=num_timesteps*(args.traj_len + 1))
                steps = steps.tolist()
                T = steps[args.num_inputs-1:]    
            else:
                steps = [num_timesteps for _ in range(num_prev + args.traj_len - 1)]
                T = num_timesteps
            
            all_indices = start + np.cumsum([0] + steps)
            pred_indices = all_indices[num_prev:]
            locs_true = locs[pred_indices].to(device) # (T, BN, 3)
            
            loc_list = locs[all_indices[:num_prev]].permute(1, 0, 2).squeeze() # BN, (num_prev,) 3
            vel_list = vels[all_indices[:num_prev]].permute(1, 0, 2).squeeze() # BN, (num_prev,) 3

            if num_prev > 1: # recompute h for all num_prevs
                h = torch.sqrt(torch.sum(vel_list ** 2, dim=-1)).T.unsqueeze(-1) # (T, BN, 1)
                if h_nodes is not None:
                    h = torch.cat((h, h_nodes.unsqueeze(0).expand(h.shape[0], -1, -1)), 
                                  dim=-1).permute(1, 0, 2)
            
            locs_pred, energies = rollout_fn(model, h, loc_list, edge_index, vel_list, edge_attr, batch, args.traj_len,
                                   num_steps=T, num_prev=num_prev, h_nodes=h_nodes,
                                   energy_fun=loader.dataset.energy_fun, gt=locs_true)
            locs_pred = locs_pred.to(device)
            #locs_pred shape: [T, BN, 3], energy shape: [T, B, 1]
            corr, avg_num_steps = pearson_correlation_batch(locs_pred, locs_true, n_nodes)
            res["tot_num_steps"] += avg_num_steps*batch_size
            
            targets = to_dense_batch(locs_true.permute(1,0,2), batch)[0].permute(0, 2, 1, 3) # (B, T, N, 3)
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
            losses = loss_mse_no_red(locs_pred, locs_true).view(args.traj_len, batch_size * n_nodes, 3)
            losses = torch.mean(losses, dim=(1, 2))
            loss = torch.mean(losses)
            res['losses'].append(losses.cpu().tolist())
        else:
            T = num_timesteps
            if args.num_inputs > 1 and not args.only_test:
                steps = None
                if varDt:
                    #pass steps to rollout to call the model at each iter with the corerct T
                    indices, steps = cumulative_random_tensor_indices_capped(N=args.traj_len,start=1,end=num_timesteps+3, MAX=num_timesteps*args.traj_len)#cumulative_random_tensor_indices(args.num_inputs,1,10)
                    steps = steps.tolist()[:args.num_inputs]
                    indices = indices[:args.num_inputs]

                start = 30
                half_step = num_timesteps
                steps = steps if steps is not None else [half_step for _ in range(args.num_inputs)]

                loc = locs[start + np.cumsum([0] + steps[:-1])].transpose(0, 1).contiguous() # BN, T, 3
                vel = vels[start + np.cumsum([0] + steps[:-1])].transpose(0, 1).contiguous()
                loc_end = locs[start + np.sum(steps)]

                h = torch.sqrt(torch.sum(vel ** 2, dim=-1)).T.unsqueeze(-1) # H (T, BN, 1)
                if h_nodes is not None:
                    h = torch.cat((h, h_nodes.unsqueeze(0).expand(h.shape[0], -1, -1)), 
                                  dim=-1).permute(1, 0, 2).contiguous().detach() 
                T = steps[-1]
                # loc_pred, h, _ = model(h, loc.detach(), edge_index, vel.detach(), edge_attr, T=) 
                #NOT T=sum(steps), T is assumed to be the default: the distance from the last step

            loc_pred, h, _ = model(h, loc.detach(), edge_index, vel.detach(), edge_attr, T=T)
            loss = criterion(loc_pred, loc_end)

        if backprop:    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
    if rollout:
        res["avg_num_steps"] = res["tot_num_steps"] / res["counter"]
    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f avg num steps %.4f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter'], res['avg_num_steps']))
    avg_loss = res['loss'] / res['counter']
    if rollout:
        wandb.log({f"{loader.dataset.partition}_loss": avg_loss,"avg_num_steps": res['avg_num_steps']}, step=epoch)
        return avg_loss, {'targets': traj_targ, 
                          'preds': traj_pred, 
                          'energies': traj_energies, 
                          'traj_losses': res['losses'],
                          'pred_indices': pred_indices,}
    else:
        wandb.log({f"{loader.dataset.partition}_loss": avg_loss}, step=epoch)
        return avg_loss


@torch.no_grad()
def rollout_fn(model, h, loc, edge_index, vel, edge_attr, batch, 
               traj_len,num_steps=10, num_prev=1, h_nodes=None,
               energy_fun=None, gt=None):

    loc_preds = torch.zeros((traj_len,loc.shape[0],loc.shape[-1])) # (T, BN, 3)
    energies = []
    for i in range(traj_len):
        T = num_steps[i] if isinstance(num_steps, list) else num_steps
        loc_p, _, vel_p = model(h, loc, edge_index, vel, edge_attr, T=T)
        
        if energy_fun is not None:
            energies.append(energy_fun(loc_p, vel_p, h_nodes, batch=batch))
        
        loc_preds[i] = loc_p

        if num_prev > 1:
            # assume loc to have shape (BN, T, 3), remove the first T and concat the predicted loc, same for vel
            loc = torch.cat((loc[:, 1:, :], loc_p.unsqueeze(1)), dim=1)  # (BN, T, 3)
            vel = torch.cat((vel[:, 1:, :], vel_p.unsqueeze(1)), dim=1)  # (BN, T, 3)
            edge_index = knn_graph(loc[:, 0, :], 4, batch) 
            rows, cols = edge_index
            edge_attr = torch.sum((loc[:, 0, :][rows] - loc[:, 0, :][cols])**2, 1).unsqueeze(1)  # relative distances among locations

            h_new = torch.sqrt(torch.sum(vel_p ** 2, dim=1)).unsqueeze(1).detach()
            h = torch.cat((h[:, 1:, :], torch.cat((h_new, h_nodes), dim=1).unsqueeze(1)), 
                          dim=1)    
        else:
            loc = loc_p  # predicted
            vel = vel_p
            edge_index = knn_graph(loc, 4, batch)
            rows, cols = edge_index
            edge_attr = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            h = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1)
            if h_nodes is not None:
                h = torch.cat((h, h_nodes), dim=1)      

    energies = torch.tensor(np.stack(energies)).unsqueeze(-1) if energy_fun is not None else None
    return loc_preds, energies


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
    B = x.size(1) // N
    x = x.reshape( T, B, -1).transpose(0,1)  # Flatten N and 3 into a single dimension
    y = y.reshape( T, B, -1).transpose(0,1)

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
            num_steps_before = T
        num_steps_batch.append(num_steps_before)

    #return the average (in the batch) number of steps before reaching a value of correlation lower than 0.5
    return correlation, torch.mean(torch.Tensor(num_steps_batch))


