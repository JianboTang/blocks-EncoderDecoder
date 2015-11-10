config = {}
config['batch_size'] = 50 #number of samples taken per each update
config['hidden_size'] = 128 # number of hidden units per layer
config['num_layers'] = 2
config['learning_rate'] = 0.002
config['learning_rate_decay'] = 0.97 # set to 0 to not decay learning

config['decay_rate'] = 0.95
config['step_clipping'] = 1.0
config['dropout'] = 0

config['model'] = 'gru'  # 'rnn' or 'gru' 'lstm'
config['nepochs'] = 1000

config['seq_length'] = 50
config['hdf5_file'] = 'input.hdf5'
config['source_file'] = 'post.txt'
config['target_file'] = 'cmnt.txt'
config['train_size'] = 0.95
config['save_path'] = '{0}_best.pkl'.format(config['model'])
config['load_path'] = '{0}_saved.pkl'.format(config['model'])
config['last_path'] = '{0}_last.pkl'.format(config['model'])
