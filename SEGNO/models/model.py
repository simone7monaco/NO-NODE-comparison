import torch
from torch import nn
from models.models.gcl import GCL, E_GCL, E_GCL_ERGN_vel

class SEGNO(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0,
                 recurrent=False, norm_diff=False, tanh=False, invariant=True, norm_vel=True, emp=True, use_previous_state=False,
                 variable_T=False):
        super(SEGNO, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.invariant = invariant
        self.norm_vel = norm_vel
        self.use_previous_state = use_previous_state
        self.variable_T = variable_T
        self.emp = emp
        self.sigmoid = nn.Sigmoid()
        self.forget = nn.Sequential(nn.Linear(hidden_nf + 3, 1)) if invariant else nn.Sequential(nn.Linear(hidden_nf + 6, 1))
        self.module = E_GCL_ERGN_vel(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, n_layers=n_layers,
                                                        act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent,
                                                        norm_diff=norm_diff, tanh=tanh, norm_vel=norm_vel)

        if not emp:
            self.decoder = nn.Sequential(nn.Linear(hidden_nf, hidden_nf),
                              act_fn,
                              nn.Linear(hidden_nf, 3))

        for i in range(0, n_layers):
            if emp:
                self.add_module("gcl_%d" % i, E_GCL_ERGN_vel(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                        act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent,
                                                        norm_diff=norm_diff, tanh=tanh, norm_vel=norm_vel))
            else:
                self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=in_edge_nf,
                                                        act_fn=act_fn, recurrent=recurrent))
        self.to(self.device)

    def forward(self, his, loc, edges, vel, edge_attr, prev_x=None,T=10):
        his = self.embedding(his)

        if self.variable_T:
            self.module.n_layers = self.n_layers*T
            #add timestep embedding (maybe not needed)
            
        h, x, v, _ = self.module(his, edges, loc, vel, vel, edge_attr=edge_attr)
        h = his + h
        
        if self.use_previous_state and prev_x is not None: 
            #use time embedding and change n_layers according to the distance T between input and predicted output
            x = x + prev_x         #to combine informations from previously predicted current state (prev_x) and observed current state (x)
                                    #change aggregation method
        for i in range(1, self.n_layers):
            his, x, v, _ = self.module(h, edges, x, v, vel, edge_attr=edge_attr)
            h = h + his

        return x, h, v