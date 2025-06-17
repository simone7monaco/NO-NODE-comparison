from torch import nn
from ...models.models.gcl import GCL, E_GCL, E_GCL_ERGN_vel

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
        self.forget = nn.Sequential(nn.Linear(hidden_nf + 3, 1)) if invariant else nn.Sequential(nn.Linear(hidden_nf + 6, 1))
        self.module = E_GCL_ERGN_vel(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, n_layers=n_layers,
                                                        act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent,
                                                        norm_diff=norm_diff, tanh=tanh, norm_vel=norm_vel)
        
        if n_inputs > 1:
            self.seq_encoder = EquivariantSequenceEncoder((in_node_nf - 1) * n_inputs + 1, in_node_nf, attn_hidden=hidden_nf)

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

    def forward(self, his, loc, edges, vel, edge_attr, prev_x=None, T=10):
        """
        Loc, vel can be n_inputs * 3 as dimension if n_inputs > 1
        """
        if len(loc.size()) == 3: # we are in the case of n_inputs > 1, loc is [n_inputs, batch_size*n_balls, 3]
            his, loc, vel = self.seq_encoder(loc, vel, his) 
        
        his = self.embedding(his)
        n_layers = T if self.varDT else 7 # *self.n_layers 7
        self.module.n_layers = n_layers
       
            #add timestep embedding (maybe not needed)
            
        h, x, v, _ = self.module(his, edges, loc, vel, vel, edge_attr=edge_attr)
        h = his + h
        
        if self.use_previous_state and prev_x is not None: 
            #consider using time embedding and change n_layers according to the distance T between input and predicted output
            x = x + prev_x         #to combine informations from previously predicted current state (prev_x) and observed current state (x)
                                    #change aggregation method
        for i in range(1, self.n_layers):
            his, x, v, _ = self.module(h, edges, x, v, vel, edge_attr=edge_attr)
            h = h + his

        return x, h, v
    

class EquivariantSequenceEncoder(nn.Module):
    def __init__(self, node_feat_dim, node_feat_dim_out, attn_hidden=32):
        super().__init__()
        # for velocity attention:
        self.attn_mlp = nn.Sequential(
            nn.Linear(1, attn_hidden),  # input is speed = ||v||, a scalar
            nn.Tanh(),
            nn.Linear(attn_hidden, 1)
        )
        # if you want an RNN on your his‐sequence:
        self.rnn = nn.Sequential(
            nn.Linear(node_feat_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, node_feat_dim_out))

    def forward(self, loc_seq, vel_seq, his_seq):
        """
        loc_seq: (N, T, 3)
        vel_seq: (N, T, 3)
        his_seq: (N, T, F)  # optional
        """
        # 1) aggregate positions by mean
        loc_init = loc_seq.mean(dim=1)   # (B, N, 3)

        # 2) compute attention weights over speeds
        speeds = vel_seq.norm(dim=-1, keepdim=True)  # (N, T, 1)
        attn = self.attn_mlp(speeds).softmax(dim=1)  # (N, T, 1)
        vel_init = (attn * vel_seq).sum(dim=1)       # (N, 3)

        # 3) aggregate historical node‐features
        #    you could also just do his_seq.mean(dim=2)
        N, T, F = his_seq.shape
        _, hN = self.rnn(his_seq)  # (1, N, F)
        his_init = hN.squeeze(0)

        return his_init, loc_init, vel_init
