import numpy as np
import torch
import random
from pathlib import Path


class NBodyDataset():
    def __init__(self, root, partition='train', max_samples=1e8, dataset="charged",
                 dataset_size="small", n_balls=5):
        
        self.root = Path(root)
        self.partition = partition
        if self.partition == 'val':
            self.suffix = 'valid'
        else:
            self.suffix = self.partition
        self.dataset = dataset
        self.n_balls = n_balls
        self.suffix += f"_{dataset}{n_balls}_initvel1{dataset_size}"

        self.start = 30 if dataset == "charged" else 0 # gravity dataset starts at 0

        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def load(self):
        loc = np.load(self.root / f'loc_{self.suffix}.npy')
        vel = np.load(self.root / f'vel_{self.suffix}.npy')
        edges = np.load(self.root / f'edges_{self.suffix}.npy')
        charges = np.load(self.root / f'charges_{self.suffix}.npy')
        
        # TODO: check dimensions and charges in gravity dataset
        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        return (loc, vel, edge_attr, charges), edges

    def preprocess(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory
        charges = charges[0:self.max_samples]
        edge_attr = []

        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(
            2)  # swap n_nodes <--> batch_size and add nf dimension

        return torch.Tensor(loc), torch.Tensor(vel), torch.Tensor(edge_attr), edges, torch.Tensor(charges)

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()
    
    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]

        return loc, vel, edge_attr, charges, loc

    def __len__(self):
        return len(self.data[0])

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges


if __name__ == "__main__":
    NBodyDataset()
