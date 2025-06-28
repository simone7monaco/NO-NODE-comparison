from ..model.basic import EGNN
from ..model.layer_no import TimeConv, get_timestep_embedding, TimeConv_x
from ..utils import repeat_elements_to_exact_shape
import torch.nn as nn
import torch


class EGNO(EGNN):
    def __init__(self, n_layers, in_node_nf, in_edge_nf, hidden_nf, activation=nn.SiLU(), device='cpu', with_v=False,
                 flat=False, norm=False, use_time_conv=True, num_modes=2, num_timesteps=8, time_emb_dim=32, 
                 num_inputs=1,varDT=False, fix_out_size=False):
        self.time_emb_dim = time_emb_dim
        if num_inputs > 1:
            in_node_nf = in_node_nf + self.time_emb_dim * 2 #use time embedding for different inputs
        else:
            in_node_nf = in_node_nf + self.time_emb_dim
        
        self.num_inputs = num_inputs
        self.varDT = varDT
        super(EGNO, self).__init__(n_layers, in_node_nf, in_edge_nf, hidden_nf, activation, device, with_v, flat, norm)
        self.use_time_conv = use_time_conv
        self.num_timesteps = num_timesteps if not fix_out_size else 10
        
        self.device = device
        self.hidden_nf = hidden_nf
        num_modes = min(num_timesteps,num_modes) if num_timesteps != 5 else min(num_modes,3)
        print(num_modes, num_timesteps)
        if use_time_conv:
            self.time_conv_modules = nn.ModuleList()
            self.time_conv_x_modules = nn.ModuleList()
            for i in range(n_layers):
                self.time_conv_modules.append(TimeConv(hidden_nf, hidden_nf, num_modes, activation, with_nin=False))
                self.time_conv_x_modules.append(TimeConv_x(2, 2, num_modes, activation, with_nin=False))

        self.to(self.device)

    def forward(self, x, h, edge_index, edge_fea, v=None, loc_mean=None, timesteps_in=None, timesteps_out=None):  # [BN, H]
        # TODO: variable dT may not work with multiple inputs (as input dTs)
        T = self.num_timesteps
        timesteps_out = torch.arange(T).to(x) if timesteps_out is None else timesteps_out
        
        if self.num_inputs > 1:
            timesteps_in = torch.arange(-self.num_inputs+1, 1).to(x) if timesteps_in is None else timesteps_in
            num_nodes = h.shape[1]

            timesteps_in = repeat_elements_to_exact_shape(timesteps_in.T.unsqueeze(1), T).T # [B, T]
            time_emb_in = get_timestep_embedding(timesteps_in, embedding_dim=self.time_emb_dim, max_positions=10000)  # [B, T, H_t]
        else:
            num_nodes = h.shape[0]
        time_emb_out = get_timestep_embedding(timesteps_out, embedding_dim=self.time_emb_dim, max_positions=10000)  # [B, T, H_t]
        
        num_edges = edge_index[0].shape[0]
        cumsum = torch.arange(0, T).to(self.device) * num_nodes
        cumsum_nodes = cumsum.repeat_interleave(num_nodes, dim=0)
        cumsum_edges = cumsum.repeat_interleave(num_edges, dim=0)

        

        if self.num_inputs > 1:
            h = [hi.unsqueeze(0) for hi in h]
            h = repeat_elements_to_exact_shape(h,T,outdims=3)
        else:
            h = h.unsqueeze(0).repeat(T, 1, 1)  # [T, BN, H]
            

        time_emb_out = time_emb_out.transpose(0, 1).unsqueeze(1).repeat(1, num_nodes//time_emb_out.size(0), 1, 1).reshape(T, -1, self.time_emb_dim) # [T, BN, H_t]

        if self.num_inputs > 1:
            time_emb_in = time_emb_in.transpose(0, 1).unsqueeze(1).repeat(1, num_nodes//time_emb_in.size(0), 1, 1).reshape(T, -1, self.time_emb_dim) # [T, BN, H_t]
            h = torch.cat((h, time_emb_in, time_emb_out), dim=-1)  # [T, BN, H+H_t]
        else:
            h = torch.cat((h, time_emb_out), dim=-1)  # [T, BN, H+H_t]

        h = h.view(-1, h.shape[-1])  # [T*BN, H+H_t]

        h = self.embedding(h)
       
        
        if self.num_inputs > 1:
            x = repeat_elements_to_exact_shape(x,T)
            v = repeat_elements_to_exact_shape(v,T)
            loc_mean = repeat_elements_to_exact_shape(loc_mean,T)
            edge_fea = repeat_elements_to_exact_shape(edge_fea,T)
            edges_0 = edge_index[0].repeat(T) + cumsum_edges
            edges_1 = edge_index[1].repeat(T) + cumsum_edges
            edge_index = [edges_0, edges_1]
            
        else:
            x = x.repeat(T, 1)
            loc_mean = loc_mean.repeat(T, 1)
            edges_0 = edge_index[0].repeat(T) + cumsum_edges
            edges_1 = edge_index[1].repeat(T) + cumsum_edges
            edge_index = [edges_0, edges_1]
            v = v.repeat(T, 1)

            edge_fea = edge_fea.repeat(T, 1)


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
