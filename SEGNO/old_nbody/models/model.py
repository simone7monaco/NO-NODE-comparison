import torch
from torch import nn
from ..models.models.gcl import GCL, E_GCL, E_GCL_ERGN_vel

class SEGNO(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0,
                 recurrent=True, norm_diff=False, tanh=False, invariant=True, norm_vel=True, emp=True, use_previous_state=0,
                 varDT=False, n_inputs=1):
        super(SEGNO, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.invariant = invariant
        self.norm_vel = norm_vel
        self.use_previous_state = True if use_previous_state > 1 else False
        self.varDT = varDT
        self.emp = emp
        self.sigmoid = nn.Sigmoid()
        # self.forget = nn.Sequential(nn.Linear(hidden_nf + 3, 1)) if invariant else nn.Sequential(nn.Linear(hidden_nf + 6, 1))
        self.module = E_GCL_ERGN_vel(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, n_layers=n_layers,
                                                        act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent,
                                                        norm_diff=norm_diff, tanh=tanh, norm_vel=norm_vel)
        
        if n_inputs > 1:
            self.enc_attn_net = InvariantTemporalAttention(in_node_nf)

        # if not emp:
        #     self.decoder = nn.Sequential(nn.Linear(hidden_nf, hidden_nf),
        #                       act_fn,
        #                       nn.Linear(hidden_nf, 3))

        # for i in range(0, n_layers):
        #     if emp:
        #         self.add_module("gcl_%d" % i, E_GCL_ERGN_vel(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
        #                                                 act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent,
        #                                                 norm_diff=norm_diff, tanh=tanh, norm_vel=norm_vel))
        #     else:
        #         self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=in_edge_nf,
        #                                                 act_fn=act_fn, recurrent=recurrent))
        self.to(self.device)

    def forward(self, his, x, edges, v, edge_attr, prev_x=None, T=10):
        """
        Loc, vel can be n_inputs * 3 as dimension if n_inputs > 1
        """
        if len(x.size()) == 3: # we are in the case of n_inputs > 1, loc is [n_inputs, batch_size*n_balls, 3]
            x, v, his = self.prepare_node_inputs(x, v, his)
        
        # TODO: other alternative with multiple inputs
        # if self.use_previous_state and prev_x is not None: 
        #     #consider using time embedding and change n_layers according to the distance T between input and predicted output
        #     x = x + prev_x         #to combine informations from previously predicted current state (prev_x) and observed current state (x)
        if self.varDT:               
            self.module.n_layers = T
            self.n_layers = T

        h = self.embedding(his)

        for i in range(1, self.n_layers):
            his, x, v, _ = self.module(h, edges, x, v, v, edge_attr=edge_attr)
            h = h + his

        return x, h, v
    
    def prepare_node_inputs(self, loc_seq, vel_seq, his_seq):
        """
        loc_seq: (BN, T, 3)
        vel_seq: (BN, T, 3)
        his_seq: (BN, T, F)

        Returns:
        - loc_init: (BN, 3)
        - vel_init: (BN, 3)
        - his_init: (BN, F)
        """
        attn = self.enc_attn_net(vel_seq, his_seq)  # (BN, T, 1)

        # Weighted sum over time
        loc_init = (attn * loc_seq).sum(dim=1)  # (BN, 3)
        vel_init = (attn * vel_seq).sum(dim=1)  # (BN, 3)
        his_init = (attn * his_seq).sum(dim=1)  # (BN, F)

        return loc_init, vel_init, his_init
    
    
class InvariantTemporalAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim=32):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_dim+1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, vel_seq, his_seq):
        speed = vel_seq.norm(dim=-1, keepdim=True)  # (N, T, 1)
        feats = torch.cat([speed, his_seq], dim=-1)  # (N, T, F+1)
        attn_weights = self.attn_mlp(feats).softmax(dim=1)
        return attn_weights
