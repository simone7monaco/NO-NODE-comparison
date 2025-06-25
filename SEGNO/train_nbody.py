import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import time


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

    res = {'epoch': epoch, 'loss': 0, 'counter': 0}
    if isinstance(criterion, tuple):
        criterion, loss_mse_no_red = criterion[0], criterion[1]
    n_nodes = loader.dataset.n_balls
    batch_size = args.batch_size

    edges = loader.dataset.get_edges(args.batch_size, n_nodes)
    edges = [edges[0], edges[1]]
    edge_index = torch.stack(edges)

    for batch_idx, data in enumerate(loader):
        data = [d.to(device) for d in data]
        for i in range(len(data)):
            if len(data[i].shape) == 4:
                data[i] = data[i].transpose(0, 1).contiguous()
                data[i] = data[i].view(data[i].size(0), -1, data[i].size(-1))
            else:
                data[i] = data[i][:, :data[i].size(1), :].contiguous()
                data[i] = data[i].view(-1, data[i].size(-1))  

        locs, vels, edge_attr, charges, loc_ends = data
        prod_charges = charges[edge_index[0]] * charges[edge_index[1]]

        loc, loc_end, vel = locs[loader.dataset.start], locs[loader.dataset.start+num_timesteps], vels[loader.dataset.start]

        # if args.time_exp:
        #     torch.cuda.synchronize()
        #     t1 = time.time()

        optimizer.zero_grad()

        h = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
        rows, cols = edges
        loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
        edge_attr = torch.cat([prod_charges, loc_dist], 1).detach()  # concatenate all edge properties
        loc_pred, h = model(h, loc.detach(), edges, vel.detach(), edge_attr)
        loss = criterion(loc_pred, loc_end)

        # if args.time_exp:
        #     torch.cuda.synchronize()
        #     t2 = time.time()
        #     time_exp_dic['time'] += t2 - t1
        #     time_exp_dic['counter'] += 1

        #     if epoch % 100 == 0:
        #         print("Forward average time: %.6f" % (time_exp_dic['time'] / time_exp_dic['counter']))

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
    print('%s epoch %d avg loss: %.5f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter']#, res
