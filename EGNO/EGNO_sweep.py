import wandb

from main_simulation_simple_one_test import main
from sweep_params import param_dicts

MAX_RUN = 50

#WANDB key for login: d8446caf4ba3380287984852fefa3ca3557b664d

wandb.login()


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
    #     # Access the hyperparameters through wandb.config
        
        config = wandb.config

        if config.num_inputs <=1 and config.varDT:
            wandb.finish()
        elif config.num_inputs > config.num_timesteps:
            wandb.finish()
            
        main(config)
        

sweep_id = wandb.sweep(sweep_config, project="EGNO-sweep-new")

wandb.agent(sweep_id, train) #MAX_RUN

