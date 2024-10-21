import numpy as np
import torch
from torch import nn, optim
from models.model import SEGNO
from torch_geometric.nn import knn_graph
from nbody.dataset_nbody import NBodyDataset
import json
time_exp_dic = {'time': 0, 'counter': 0}

torch.manual_seed(40)


def train(gpu, args):
    if args.gpus == 0:
        device = 'cpu'
    else:
        device = torch.device('cuda:' + str(gpu))

    model = SEGNO(in_node_nf=1, in_edge_nf=1, hidden_nf=64, device=device, n_layers=args.layers,
                         recurrent=True, norm_diff=False, tanh=False, use_previous_state=args.use_previous_state)

    dataset_train = NBodyDataset(partition='train', dataset_name=args.nbody_name,
                                 max_samples=args.max_samples)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    dataset_val = NBodyDataset(partition='val', dataset_name=args.nbody_name)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

    dataset_test = NBodyDataset(partition='test', dataset_name=args.nbody_name)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_mse = nn.MSELoss()
    loss_mse_no_red = nn.MSELoss(reduction='none')
    results = {'eval epoch': [], 'val loss': [], 'test loss': [], 'train loss': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best = {'long_loss': {}}
    print(args.use_previous_state)
    for epoch in range(0, args.epochs):
        train_loss, _ = run_epoch(model, optimizer, [loss_mse,loss_mse_no_red], epoch, loader_train, device, args, use_previous_state=args.use_previous_state)
        results['train loss'].append(train_loss)
        if epoch % args.test_interval == 0 or epoch == args.epochs-1:

            val_loss, res = run_epoch(model, optimizer, [loss_mse,loss_mse_no_red], epoch, loader_val, device, args, backprop=False,use_previous_state=args.use_previous_state)
            test_loss, res = run_epoch(model, optimizer, [loss_mse,loss_mse_no_red], epoch, loader_test,
                                  device, args, backprop=False,rollout=True)
            
            results['eval epoch'].append(epoch)
            results['val loss'].append(val_loss)
            results['test loss'].append(test_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_epoch = epoch
                best = res
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best Test avg num steps: %.4f \t Best epoch %d" %
                  (best_val_loss, best_test_loss, best["avg_num_steps"], best_epoch))
            # print(best['long_loss'])
    
    json_object = json.dumps(results, indent=4)
    with open("results.json", "w") as outfile:
        outfile.write(json_object)

    traj_losses = torch.stack(best['losses'], dim=0)
    torch.save(traj_losses,'traj_losses.pt')

    return best_val_loss, best_test_loss, best_epoch


def run_epoch(model, optimizer, criterion, epoch, loader, device, args, backprop=True,rollout=False,use_previous_state=False):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'losses': [], "tot_num_steps": 0,"avg_num_steps": 0, 'counter': 0, 'long_loss': {}}
    criterion, loss_mse_no_red = criterion[0], criterion[1]
    n_nodes = 5
    batch_size = args.batch_size

    for batch_idx, data in enumerate(loader):
        data = [d.to(device) for d in data]
        for i in range(len(data)):
            if len(data[i].shape) == 4:
                
                data[i] = data[i].transpose(0, 1).contiguous()
                
                data[i] = data[i].view(data[i].size(0), -1, data[i].size(-1))
                
            else:
                
                data[i] = data[i][:, :data[i].size(1), :].contiguous()
                
                data[i] = data[i].view(-1, data[i].size(-1))  
                

        locs, vels, loc_ends = data
        loc, loc_end, vel = locs[30], locs[40], vels[30]
        
        batch = torch.arange(0, batch_size)
        batch = batch.repeat_interleave(n_nodes).long().to(device)
        
        edge_index = knn_graph(loc, 4, batch)
        h = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
        rows, cols = edge_index
        loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
        edge_attr = loc_dist.detach()

        if rollout:
            locs_true = locs[40:130:10].to(device)
            traj_len = locs_true.shape[0]
            num_prev = 1
            loc_list = []
            for i in range(num_prev):
                loc_list.append(locs[i*10+30]) #start from 30
            locs_pred = rollout_fn(model,h, loc_list, edge_index, vel, edge_attr, batch, traj_len, num_prev=num_prev).to(device)

            corr, avg_num_steps = pearson_correlation_batch(locs_pred, locs_true, n_nodes)
            res["tot_num_steps"] += avg_num_steps*batch_size
            res["avg_num_steps"] = res["tot_num_steps"] / res["counter"]

            #loss with metric (A-MSE)
            losses = loss_mse_no_red(locs_pred, locs_true).view(traj_len, batch_size * n_nodes, 3)
            losses = torch.mean(losses, dim=(1, 2))
            loss = torch.mean(losses)
            res['losses'].append(losses)
        else:
            if use_previous_state:
                x, h, _ = model(h, loc.detach(), edge_index, vel.detach(), edge_attr)
                prev_x = x
                pred_x = x
                #call model again with new observed values and prev_x
                loc, vel = locs[35], vels[35] #take sample corresponding to half the delta t (just a try)
                #loc_end = torch.cat(loc,loc_end)
                batch = torch.arange(0, batch_size)
                batch = batch.repeat_interleave(n_nodes).long().to(device)
                
                edge_index = knn_graph(loc, 4, batch)
                h = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
                rows, cols = edge_index
                loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
                edge_attr = loc_dist.detach()

                loc_pred, h, _ = model(h, loc.detach(), edge_index, vel.detach(), edge_attr, prev_x)
                #loc_pred = torch.cat(pred_x, loc_pred)
                loss = criterion(loc_pred, loc_end) #maybe consider intermediate steps for loss computation

            else:
                loc_pred, h, _ = model(h, loc.detach(), edge_index, vel.detach(), edge_attr)
                loss = criterion(loc_pred, loc_end)

        res['loss'] += loss.item()*batch_size

        if backprop:    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            res['counter'] += batch_size
        res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f avg num steps %.4f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter'], res['avg_num_steps']))

    return res['loss'] / res['counter'], res


def rollout_fn(model, h, loc_list, edge_index, v, edge_attr, batch, traj_len, num_prev=0):

    loc_preds = torch.zeros((traj_len,loc.shape[0],loc.shape[1]))
    vel = v
    prev = None
    prevs = 0
    loc = loc_list[0]

    loc, _, vel = model(h, loc.detach(), edge_index, vel.detach(), edge_attr)
    
    for i in range(traj_len):
        
        if num_prev is not 0 and i < num_prev:
            prev = loc   #predicted
            prevs +=1
            loc_preds[i] = loc
            if len(loc_preds) == traj_len:
                break
            loc = loc_list[prevs] #observed
            edge_index = knn_graph(loc, 4, batch)
            h = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            rows, cols = edge_index
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = loc_dist.detach()
            loc, _, vel = model(h, loc.detach(), edge_index, vel.detach(), edge_attr, prev)
        else:
            loc_preds[i] = loc
            if len(loc_preds) == traj_len:
                break
            edge_index = knn_graph(loc, 4, batch)
            h = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            rows, cols = edge_index
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = loc_dist.detach()
            loc, _, vel = model(h, loc.detach(), edge_index, vel.detach(), edge_attr)

    
    return loc_preds

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
    x = x.reshape( B, T, -1)  # Flatten N and 3 into a single dimension
    y = y.reshape( B, T, -1)

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


