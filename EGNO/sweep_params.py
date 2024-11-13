param_dicts = {
    #default, no need to change for sweep
    'exp_name': {
        'value': 'exp_1'
    }, 
    'batch_size': {
        'value': 100
    }, 
    'epochs': {
        'value': 5 #00
        }, 
    'no_cuda': {
        'value': False
        }, 
    'seed': {
        'value': 1
        }, 
    'log_interval': {
        'value': 1
        }, 
    'test_interval': {
        'value': 4#5
        }, 
    'outf': {
        'value': 'exp_results'
        }, 
    'model': {
        'value': 'egno'
        }, 
    'max_training_samples': {
        'value': 3000
        }, 
    'data_dir': {
        'value': ''
        }, 
    'dropout': {
        'value': 0.5
        }, 
    'config_by_file': {
        'value': None
        }, 
    'lambda_link': {
        'value': 1
        }, 
    'n_cluster': {
        'value': 3
        }, 
    'flat': {
        'value': False
        }, 
    'interaction_layer': {
        'value': 3
        }, 
    'pooling_layer': {
        'value': 3
        }, 
    'decoder_layer': {
        'value': 1
        }, 
    'norm': {
        'value': False
        }, 
    ###params to search through
    'weight_decay': {
        'value': 1e-12
        }, 
    'lr': {
        'values': [0.0005,0.0001,0.00005]
        }, 
    'nf': {
        'value': 64
        }, 
    'n_layers': {
        'value': 4
        }, 
    'varDT': {
        'value': False  #check constraint on num_inputs
        }, 
    'rollout': {
        'value': True
        }, 
    'variable_deltaT': {
        'value': False
        }, 
    'only_test': {
        'value': True  #check code
        }, 
    'num_inputs': {
        'value': 1
        },
    'traj_len': {
        'value': 10   #static?
        }, 
    'num_timesteps': {
        'value': 10
        }, 
    'time_emb_dim': {
        'value': 32
        }, 
    'num_modes': {
        'value': 5
        },
    'n_balls': {
        'value': 5}}  #3/5/8/20/50