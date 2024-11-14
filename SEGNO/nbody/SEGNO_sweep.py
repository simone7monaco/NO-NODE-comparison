import wandb

from train_nbody import train as train_model
from sweep_params import param_dicts

MAX_RUN = 50

#WANDB key for login: d8446caf4ba3380287984852fefa3ca3557b664d

wandb.login()


#            config={
#     "learning_rate": args.lr,
#     "weight_decay": args.weight_decay,
#     "hidden_dim": args.nf,
#     "dropout": args.dropout,
#     "batch_size": args.batch_size,
#     "epochs": args.epochs,
#     "model": args.model,
#     "nlayers": args.n_layers,  
#     "time_emb_dim": args.time_emb_dim,
#     "num_modes": args.num_modes,  
#     "rollout": args.rollout,  
#     "num_timesteps": args.num_timesteps,
#     "num_inputs": args.num_inputs,
#     "only_test": args.only_test,
#     "varDT": args.varDT,
#     "variable_deltaT": args.variable_deltaT
#     })

parameters_dict = {
        'lr': {
            'values': [0.01, 0.001, 0.0001]
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'epochs': {
            'value': 500
        },
        'exp_name': {
            'value': "exp_r"
        },
        'no-cuda': {
            'value': False
        },


    }

sweep_config = {
    'method': 'grid',  # or 'random'
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'parameters': param_dicts
}



def train(config=None):
    
    # Set up W&B config for the sweep
    with wandb.init(config=config):
    # Access the hyperparameters through wandb.config
        
        config = wandb.config
        
        if config.num_inputs <=1:
            config.varDT = False

        if config.gpus == 0:
            config.mode = 'cpu'
            train_model(0, config)
        
        elif config.gpus == 1:
            config.mode = 'gpu'
            train_model(0, config)
        

sweep_id = wandb.sweep(sweep_config, project="SEGNO-sweep-test")
wandb.agent(sweep_id, train, count=3) #MAX_RUN