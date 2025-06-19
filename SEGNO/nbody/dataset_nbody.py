import numpy as np
import torch
from utils import conserved_energy_fun
from torch_geometric.utils import to_dense_batch


class NBodyDataset():
    """
    NBodyDataset
    """

    def __init__(self, data_dir, partition='train', max_samples=1e8, dataset="charged",dataset_name="nbody_small", n_balls=5):
        self.partition = partition
        self.data_dir = data_dir
        if self.partition == 'val':
            self.suffix = 'valid'
        else:
            self.suffix = self.partition
        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            self.suffix += f"_{dataset}5_initvel1"
        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            self.suffix += f"_{dataset}{n_balls}_initvel1small"
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)

        self.n_balls = n_balls
        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.data, self.edges = self.load()

    def energy_fun(self, loc, vel, edges, batch=None):
        return conserved_energy_fun(self.dataset, loc, vel, edges, batch=batch)

    def load(self):
        # loc = np.load(osp.join(dir, 'dataset_gravity', 'loc_' + self.suffix + '.npy'))
        loc = np.load(self.data_dir / f'loc_{self.suffix}.npy')
        vel = np.load(self.data_dir / f'vel_{self.suffix}.npy')
        if loc.shape[-2:] != (self.n_balls, 3):
            # should transpose the last two dimensions
            loc = np.transpose(loc, (0, 1, 3, 2))
            vel = np.transpose(vel, (0, 1, 3, 2))
            assert (loc.shape[-2:] == (self.n_balls, 3) and vel.shape[-2:] == (self.n_balls, 3)), "Shape mismatch!"
       
        charges = np.load(self.data_dir / f'charges_{self.suffix}.npy')
        loc, vel = self.preprocess(loc, vel, charges)
        return (loc, vel), None

    def preprocess(self, loc, vel, charges=None):
        loc, vel = torch.tensor(loc), torch.tensor(vel)
        n_nodes = loc.size(2)
        
        if charges is not None:
            charges = torch.tensor(charges) # [N_sym, n_nodes, 1]
            # expand charges to match the time dimension
            charges = charges.unsqueeze(1).expand(-1, loc.size(1), -1, -1)
            loc = torch.cat((loc, charges), dim=-1)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory

        return loc, vel

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel = self.data
        loc, vel = loc[i], vel[i]

        if self.dataset_name == "nbody":
            frame_0, frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            frame_0, frame_T = 30, 40
        elif self.dataset_name == "nbody_small_out_dist":
            frame_0, frame_T = 20, 30
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)
        #print(loc.shape)
        #loc = torch.transpose(loc, 1, 2)
        #vel = torch.transpose(vel, 1, 2)
         
        #loc shape: [519, 5, 3]
        return loc, vel

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
