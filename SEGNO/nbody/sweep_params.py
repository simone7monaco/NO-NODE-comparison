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
    'nbody_name': {
        'value': "nbody_small"
        }, 
    'seed': {
        'value': 1
        }, 
    'log': {
        'value': False
        }, 
    'num_workers': {
        'value': 4
        }, 
    'save_dir': {
        'value': "saved models"
        },
    'root': {
        'value': "datasets"
        },
    'download': {
        'value': False
        },
    'test_interval': {
        'value': 5
        }, 
    'time_exp': {
        'value': False
        }, 
    'outf': {
        'value': 'exp_results'
        }, 
    'model': {
        'value': 'segno'
        }, 
    'max_samples': {
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
        'value': "instance"
        }, 
    'pool': {
        'value': "avg"
        }, 
    'conv_type': {
        'value': "linear"
        }, 
    'gpus': {
        'value': 1 #change for gpu
        }, 
    'subspace_type': {
        'value': "weightbalanced"
        },
    ###params to search through
    'neighbours': {
        'value': 6
        }, 
    'weight_decay': {
        'value': 1e-12
        }, 
    'lr': {
        'values': [0.0005,0.00005]
        }, 
    'hidden_features': {
        'value': 128
        }, 
    'lmax_h': {
        'value': 2
        }, 
    'lmax_attr': {
        'value': 3
        },
    'layers': {
        'value': 1
        }, 
    'varDT': {
        'values': [False, True]
        }, 
    'rollout': {
        'value': True
        }, 
    'variable_deltaT': {
        'value': False
        }, 
    'only_test': {
        'values': [True, False]  
        }, 
    'num_inputs': {
        'values': [1,2,3]
        },
    'traj_len': {
        'value': 10  
        }, 
    'num_steps': {
        'values': [2,5,10]
        }, 
    'n_balls': {
        'values': [5]}}  #3/5/8/20/50