import numpy as np
import torch
from torch import nn, optim
from models.model import SEGNO
from torch_geometric.nn import knn_graph
from nbody.dataset_nbody import NBodyDataset

time_exp_dic = {'time': 0, 'counter': 0}

torch.manual_seed(40)


def train(gpu, args):
    if args.gpus == 0:
        device = 'cpu'
    else:
        device = torch.device('cuda:' + str(gpu))

    model = SEGNO(in_node_nf=1, in_edge_nf=1, hidden_nf=64, device=device, n_layers=args.layers,
                         recurrent=True, norm_diff=False, tanh=False)

    dataset_train = NBodyDataset(partition='train', dataset_name=args.nbody_name,
                                 max_samples=args.max_samples)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    dataset_val = NBodyDataset(partition='val', dataset_name=args.nbody_name)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

    dataset_test = NBodyDataset(partition='test', dataset_name=args.nbody_name)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_mse = nn.MSELoss()

    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best = {'long_loss': {}}
    for epoch in range(0, args.epochs):
        train_loss, _ = run_epoch(model, optimizer, loss_mse, epoch, loader_train, device, args)

        if epoch % args.test_interval == 0 or epoch == args.epochs-1:

            val_loss, res = run_epoch(model, optimizer, loss_mse, epoch, loader_val, device, args, backprop=False)
            test_loss, res = run_epoch(model, optimizer, loss_mse, epoch, loader_test,
                                  device, args, backprop=False)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_epoch = epoch
                best = res
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d" %
                  (best_val_loss, best_test_loss, best_epoch))
            # print(best['long_loss'])

    return best_val_loss, best_test_loss, best_epoch


def run_epoch(model, optimizer, criterion, epoch, loader, device, args, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'counter': 0, 'long_loss': {}}
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
        loc_pred, h = model(h, loc.detach(), edge_index, vel.detach(), edge_attr)
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
    print('%s epoch %d avg loss: %.5f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter'], res
