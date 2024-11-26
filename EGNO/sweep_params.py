param_dicts = {
    #default, no need to change for sweep
    'exp_name': {
        'value': 'exp_1'
    }, 
    'batch_size': {
        'value': 100
    }, 
    'epochs': {
        'value': 500
        }, 
    'no_cuda': {
        'value': False
        }, 
    'seed': {
        'values': [1, 22, 39, 42, 58, 64, 70, 78, 86, 93]
        }, 
    'log_interval': {
        'value': 1
        }, 
    'test_interval': {
        'value': 10
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
        'values': [0.0005,0.00005]
        }, 
    'nf': {
        'value': 64
        }, 
    'n_layers': {
        'values': [3,4,5]
        }, 
    'varDT': {
        'values': [False, True]  #check constraint on num_inputs
        }, 
    'rollout': {
        'value': True
        }, 
    'variable_deltaT': {
        'value': False
        }, 
    'only_test': {
        'value': False 
        }, 
    'num_inputs': {
        'values': [1,2,3,4]
        },
    'traj_len': {
        'value': 10   #static?
        }, 
    'num_timesteps': {
        'values': [2,5,10]
        }, 
    'time_emb_dim': {
        'value': 32
        }, 
    'num_modes': {
        'value': 5
        },
    'n_balls': {
        'values': [5]}}  #3/5/8/20/50