import numpy as np
import torch
from utils import conserved_energy_fun
from EGNO.utils import random_ascending_tensor


class NBodyDataset():
    """
    NBodyDataset

    """

    def __init__(self, data_dir, partition='train', max_samples=1e8, dataset="charged", dataset_name="nbody_small",n_balls=5):
        self.partition = partition
        self.data_dir = data_dir
        if self.partition == 'val':
            self.suffix = 'valid'
        else:
            self.suffix = self.partition
        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            self.suffix += f"_{dataset}{n_balls}_initvel1"
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
        loc = np.load(self.data_dir / f'loc_{self.suffix}.npy') # shape (n_samples, n_timesteps, n_balls, 3)
        vel = np.load(self.data_dir / f'vel_{self.suffix}.npy')
        if loc.shape[-2:] != (self.n_balls, 3):
            # should transpose the last two dimensions
            loc = np.transpose(loc, (0, 1, 3, 2))
            vel = np.transpose(vel, (0, 1, 3, 2))
            assert (loc.shape[-2:] == (self.n_balls, 3) and vel.shape[-2:] == (self.n_balls, 3)), "Shape mismatch!"

        # edges = np.load(self.data_dir / f'edges_{self.suffix}.npy')
        charges = np.load(self.data_dir / f'charges_{self.suffix}.npy')
        mat_charges = charges.repeat(charges.shape[1], axis=2)
        edges = np.einsum('tij,tji ->tij', mat_charges, mat_charges)
        print(f"Loaded dataset {self.suffix} with {loc.shape[0]} samples, {loc.shape[2]} nodes, {loc.shape[3]} features")
        
        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        return (loc, vel, edge_attr, charges), edges

    def preprocess(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.tensor(loc).float(), torch.tensor(vel).float()
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
        edge_attr = torch.tensor(np.array(edge_attr)).float().transpose(0, 1).unsqueeze(2) 
        # swap n_nodes <--> batch_size and add nf dimension
        return loc, vel, edge_attr, edges, torch.tensor(charges).float()

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]

        if self.dataset_name == "nbody":
            frame_0, frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            frame_0, frame_T = 30, 40
        elif self.dataset_name == "nbody_small_out_dist":
            frame_0, frame_T = 20, 30
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)

        return loc[frame_0], vel[frame_0], edge_attr, charges, loc[frame_T]

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


class NBodyDynamicsDataset(NBodyDataset):
    def __init__(self, partition='train', data_dir='.', max_samples=1e8, dataset="charged",dataset_name="nbody_small", n_balls=5, num_timesteps=10, 
                 num_inputs=1, traj_len=1, dT = 1, varDT=False):
        self.num_timesteps = num_timesteps
        self.traj_len = traj_len
        self.num_inputs = num_inputs
        self.var_dt = varDT
        self.dT = dT
        frames_0_dict = {'nbody': 6, 'nbody_small': 30, 'nbody_small_out_dist': 20}
        self.start = frames_0_dict.get(dataset_name) if dataset == 'charged' else 0 # 0 for gravity dataset
        if self.start is None:
            raise Exception("Wrong dataset partition %s" % dataset_name)
        super(NBodyDynamicsDataset, self).__init__(data_dir, partition, max_samples, dataset, dataset_name, n_balls=n_balls)

    def __getitem__(self, i):
        assert self.num_inputs <= self.num_timesteps
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]

        frame_0 = self.start
        frame_T = frame_0 + self.num_timesteps*self.traj_len * self.dT
        if self.num_inputs > 1:
            # inputs are all BEFORE frame_0
            if self.var_dt:
                # random inputs
                timesteps_in = random_ascending_tensor(length=self.num_inputs-1, max_value=self.num_timesteps-1, min_value=1) # deltas bethween inputs
                timesteps_in = torch.cat((torch.tensor([0]), timesteps_in), dim=0)  # add first input at frame_0
            else:
                # equispaced inputs
                timesteps_in = (torch.arange(self.num_timesteps) * self.dT)[:self.num_inputs]

            timesteps_in = - torch.flip(timesteps_in, dims=(0,))  # flip to have descending order
            frame_0 = (frame_0 + timesteps_in * self.dT)
            if (frame_0<0).any():
                # push to the first frame
                frame_T += -frame_0.min()
                frame_0 += -frame_0.min()
            out_indices = torch.arange(frame_0[-1]+1, frame_T+1, self.dT)
        else:
            timesteps_in = torch.tensor([0]).int()
            out_indices = torch.arange(frame_0+1, frame_T+1, self.dT)

        if out_indices.max() >= loc.size(0):
            # reduce out_indices to the maximum available frame
            out_indices = out_indices[out_indices < loc.size(0)]
        locs_out = loc[out_indices].transpose(1, 0) 
        # vels_out = vel[out_indices].transpose(1, 0)  
        # shape (n_balls, T, 3)

        return loc[frame_0], vel[frame_0], edge_attr, charges, locs_out, frame_0, out_indices


if __name__ == "__main__":
    dataset = NBodyDynamicsDataset('train', data_dir='./dataset', max_samples=3000, num_timesteps=100)
    for i in dataset[100]:
        print(i.shape)
        print(i)

