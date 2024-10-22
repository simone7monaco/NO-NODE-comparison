from model.basic import EGNN
from model.layer_no import TimeConv, get_timestep_embedding, TimeConv_x
import torch.nn as nn
import torch


class EGNO(EGNN):
    def __init__(self, n_layers, in_node_nf, in_edge_nf, hidden_nf, activation=nn.SiLU(), device='cpu', with_v=False,
                 flat=False, norm=False, use_time_conv=True, num_modes=2, num_timesteps=8, time_emb_dim=32, num_inputs=1):
        self.time_emb_dim = time_emb_dim
        in_node_nf = in_node_nf + self.time_emb_dim
        self.num_inputs = num_inputs
        super(EGNO, self).__init__(n_layers, in_node_nf, in_edge_nf, hidden_nf, activation, device, with_v, flat, norm)
        self.use_time_conv = use_time_conv
        self.num_timesteps = num_timesteps
        self.device = device
        self.hidden_nf = hidden_nf

        if use_time_conv:
            self.time_conv_modules = nn.ModuleList()
            self.time_conv_x_modules = nn.ModuleList()
            for i in range(n_layers):
                self.time_conv_modules.append(TimeConv(hidden_nf, hidden_nf, num_modes, activation, with_nin=False))
                self.time_conv_x_modules.append(TimeConv_x(2, 2, num_modes, activation, with_nin=False))

        self.to(self.device)

    def forward(self, x, h, edge_index, edge_fea, v=None, loc_mean=None):  # [BN, H]

        T = self.num_timesteps

        num_nodes = h.shape[0]
        num_edges = edge_index[0].shape[0]

        cumsum = torch.arange(0, T).to(self.device) * num_nodes
        cumsum_nodes = cumsum.repeat_interleave(num_nodes, dim=0)
        cumsum_edges = cumsum.repeat_interleave(num_edges, dim=0)

        time_emb = get_timestep_embedding(torch.arange(T).to(x), embedding_dim=self.time_emb_dim, max_positions=10000)  # [T, H_t]
        h = h.unsqueeze(0).repeat(T, 1, 1)  # [T, BN, H]
        time_emb = time_emb.unsqueeze(1).repeat(1, num_nodes, 1)  # [T, BN, H_t]
        h = torch.cat((h, time_emb), dim=-1)  # [T, BN, H+H_t]
        h = h.view(-1, h.shape[-1])  # [T*BN, H+H_t]

        h = self.embedding(h)
        
        if self.num_inputs > 1:
            for i in range(self.num_inputs):
                xi = x[i].repeat(T // self.num_inputs, 1)
                vi = v[i].repeat(T // self.num_inputs, 1)
                loci = loc_mean[i].repeat(T // self.num_inputs, 1)
                edge_fea_i = edge_fea[i].repeat(T // self.num_inputs, 1)
                edge_index_0i = edge_index[i][0].repeat(T // self.num_inputs)
                edge_index_1i = edge_index[i][1].repeat(T // self.num_inputs)
                if i == 0:
                    x_tot = xi
                    v_tot = vi
                    loc_tot = loci
                    edge_fea_tot = edge_fea_i
                    edge_index_0tot = edge_index_0i
                    edge_index_1tot = edge_index_1i
                else:
                    x_tot = torch.cat(x_tot, xi)
                    v_tot = torch.cat(v_tot, vi)
                    loc_tot = torch.cat(loc_tot, loci)
                    edge_fea_tot = torch.cat(edge_fea_tot, edge_fea_i)
                    edge_index_0tot = torch.cat(edge_index_0tot, edge_index_0i)
                    edge_index_1tot = torch.cat(edge_index_1tot, edge_index_1i)
            x = x_tot
            v = v_tot
            loc_mean = loc_tot
            edge_fea = edge_fea_tot
            edges_0 = edge_index_0tot + cumsum_edges
            edges_1 = edge_index_1tot + cumsum_edges
            edge_index = [edges_0, edges_1]
        else:
            x = x.repeat(T, 1)
            loc_mean = loc_mean.repeat(T, 1)
            edges_0 = edge_index[0].repeat(T) + cumsum_edges
            edges_1 = edge_index[1].repeat(T) + cumsum_edges
            edge_index = [edges_0, edges_1]
            v = v.repeat(T, 1)

            edge_fea = edge_fea.repeat(T, 1)

        #### do the same for all params

        

        for i in range(self.n_layers):
            if self.use_time_conv:
                time_conv = self.time_conv_modules[i]
                h = time_conv(h.view(T, num_nodes, self.hidden_nf)).view(T * num_nodes, self.hidden_nf)
                x_translated = x - loc_mean
                time_conv_x = self.time_conv_x_modules[i]
                X = torch.stack((x_translated, v), dim=-1)
                temp = time_conv_x(X.view(T, num_nodes, 3, 2))
                x = temp[..., 0].view(T * num_nodes, 3) + loc_mean
                v = temp[..., 1].view(T * num_nodes, 3)

            x, v, h = self.layers[i](x, h, edge_index, edge_fea, v=v)
        return (x, v, h) if v is not None else (x, h)
