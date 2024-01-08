
import pymc as pm
import numpy as np
# np.bool = np.bool_
import aesara.tensor as at
import pickle
from utils import *

n_groups = 5
n_components = 3

sim_data_path = './sim/save/pagsim_w_stimuli_600s_inh/'
brain_regions = [
                'VMH',
                'ACC',
                'IC',
                'SC',
                'PMD'
                ]
presyn_binned = np.load(sim_data_path+'presyn_binned.npy')
presyn_smooth = np.load(sim_data_path+'presyn_exponential_smooth.npy')
bin_size = 0.001
n_input_neurons, n_bins = presyn_binned.shape
n_neurons_per_group = np.load(sim_data_path+'n_neurons_per_group.npy')
_total_length = n_bins / 1000

n_PAG_to_use = 1 # specify how many PAG neurons to learn

train_start = 0
train_end = 0.8
test_start = 8
test_end = 10


if __name__ == '__main__':
    
    pag_df = extract_sim_as_df(sim_data_path, 'PAG')
    pag_timings = extract_timings(pag_df, 'PAG')
    pag_binned_spikes = bin_spikes(pag_timings, start_time=0, end_time=_total_length, bin_size=0.001)[:n_PAG_to_use]
    print(f'Average PAG firing rate: {np.sum(pag_binned_spikes) / pag_binned_spikes.shape[0] / _total_length} Hz.')

    train_bins = int((train_end - train_start) / bin_size)
    test_bins = int((test_end - test_start) / bin_size)
    X_train, X_test = np.zeros((n_input_neurons*n_PAG_to_use, n_PAG_to_use * train_bins)), np.zeros((n_input_neurons*n_PAG_to_use, n_PAG_to_use * test_bins))
    y_train, y_test = np.zeros((1, train_bins * n_PAG_to_use)), np.zeros((1, test_bins * n_PAG_to_use))

    for n in range(n_PAG_to_use):
        X_train[n_input_neurons*n:n_input_neurons*(n+1), n * train_bins: (n+1) * train_bins] = presyn_smooth[:, int(train_start/bin_size):int(train_end/bin_size)] # chunks of n_neurons * train_set_bin_size
        X_test[n_input_neurons*n:n_input_neurons*(n+1), n * test_bins :(n+1) * test_bins] = presyn_smooth[:, int(test_start/bin_size):int(test_end/bin_size)]
        
        y_train[:, n * train_bins: (n+1)* train_bins] = pag_binned_spikes[n, int(train_start/bin_size): int(train_end/bin_size)]
        y_test[:, n * test_bins: (n+1)* test_bins] = pag_binned_spikes[n, int(test_start/bin_size): int(test_end/bin_size)]
    assert X_train.shape[1] == y_train.shape[1]
    assert X_test.shape[1] == y_test.shape[1]

    print('Number of PAG spikes in the training set: ', np.sum(y_train))
    print('Number of PAG spikes in the test set: ', np.sum(y_test))
    
    
    
    with pm.Model() as model:
        
        # priors for mixture components
        mu = pm.Normal('mu', mu=0, sigma=10, shape=(n_groups, n_components))
        sigma = pm.HalfNormal('sigma', sigma=10, shape=(n_groups, n_components))
        bias = pm.Normal('global_bias', mu=0, sigma=10)
        r = pm.Dirichlet('w',
                        a=np.ones(n_components),
                        shape=(n_groups, n_components))
        
        # mixture weights
        weights = []
        for m in range(n_groups):
            weight = pm.NormalMixture(f'weight_group_{m}',
                                    w=r[m],
                                    mu=mu[m],
                                    sigma=sigma[m],
                                    shape=int(n_neurons_per_group[m]))
            weights.append(weight)
            
        all_weights = at.concatenate(weights).reshape((1,-1))
        
        y_est = pm.math.sigmoid(at.dot(all_weights, X_train) + bias)
        
        likelihood = pm.Bernoulli('likelihood', p=y_est, observed=y_train)
        
        trace = pm.sample(1000, tune=500, target_accept=0.95, progressbar=True)
        assert False