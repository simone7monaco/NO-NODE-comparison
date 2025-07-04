import torch
from torch import nn
from .models.gcl import SEGNO_GCL


class SEGNO(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0,
                 recurrent=False, norm_diff=False, tanh=False, invariant=True, norm_vel=True, 
                 varDT=False, multiple_agg=None):
        super(SEGNO, self).__init__()
        self.hidden_nf = hidden_nf
        self.varDT = varDT
        self.multiple_agg = multiple_agg
        if multiple_agg == 'attn':
            self.enc_attn_net = InvariantTemporalAttention(in_node_nf, hidden_dim=hidden_nf)

        self.device = device
        self.n_layers = n_layers
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.invariant = invariant
        self.norm_vel = norm_vel
        self.sigmoid = nn.Sigmoid()
        self.module = SEGNO_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                        act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent,
                                                        norm_diff=norm_diff, tanh=tanh, norm_vel=norm_vel)
        self.to(self.device)

    def forward(self, his, x, edges, v, edge_attr, T=10, in_steps=None):
        """
        Loc, vel can be n_inputs * 3 as dimension if n_inputs > 1
        """
        if len(x.size()) == 3:  # we are in the case of n_inputs > 1, loc is [n_inputs, batch_size*n_balls, 3]
            if self.multiple_agg == 'attn':
                x, v, his = self.prepare_node_inputs(x, v, his, in_steps=in_steps)
            elif self.multiple_agg == 'sum':
                assert x.size(0) == 2, "For sum aggregation, x should have 2 inputs per forward call."
                x = x.sum(dim=0)
                v = v.sum(dim=0)
                his = his.sum(dim=0)
            else:
                raise ValueError("Invalid multiple aggregation method specified.")

        if self.varDT:               
            self.module.n_layers = T
            self.n_layers = T

        h = self.embedding(his)
        for _ in range(self.n_layers):
            h, x, v, _ = self.module(h, edges, x, v, v, edge_attr=edge_attr)
            
        return x, h, v
    
    def prepare_node_inputs(self, loc_seq, vel_seq, his_seq, in_steps):
        """
        loc_seq: (BN, T, 3)
        vel_seq: (BN, T, 3)
        his_seq: (BN, T, F)
        in_steps: (T)

        Returns:
        - loc_init: (BN, 3)
        - vel_init: (BN, 3)
        - his_init: (BN, F)
        """
        # repeat in_steps to have shape (BN, T, 1)
        in_steps = in_steps.repeat(loc_seq.size(0), 1).unsqueeze(-1)  # (BN, T, 1)
        attn = self.enc_attn_net(vel_seq, his_seq, in_steps)  # (BN, T, 1)

        # Weighted sum over time
        loc_init = (attn * loc_seq).sum(dim=1)  # (BN, 3)
        vel_init = (attn * vel_seq).sum(dim=1)  # (BN, 3)
        his_init = (attn * his_seq).sum(dim=1)  # (BN, F)

        return loc_init, vel_init, his_init
    
    
class InvariantTemporalAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim=32):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_dim+2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, vel_seq, his_seq, in_steps):
        speed = vel_seq.norm(dim=-1, keepdim=True)  # (N, T, 1)
        feats = torch.cat([speed, his_seq, in_steps], dim=-1)  # (N, T, F+2)
        attn_weights = self.attn_mlp(feats).softmax(dim=1)
        return attn_weights

        