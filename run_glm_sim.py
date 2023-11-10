import numpy as np
from glm import BernoulliGLM, BernoulliGLMwReg, BernoulliGLMPyTorch
import pandas as pd
from preprocess_data import *

sim_data_path = './sim/save/pagsim/'
use_torch = True

all_dfs = []
all_timings = []
all_binned_spikes = []
brain_regions = [
                'VMH',
                # 'ACC',
                # 'IC',
                # 'SC',
                # 'PMD'
                ]       

# extract all input spikes and bin them
for brain_region in brain_regions:
    df = extract_sim_as_df(sim_data_path, brain_region)
    all_dfs.append(df)
    timings = extract_timings(df, brain_region)
    all_timings.append(timings)

_total_length = np.max([total_length(df) for df in all_dfs])
for i, brain_region in enumerate(brain_regions):
    binned_spikes = (bin_spikes(all_timings[i], start_time=0, end_time=_total_length, bin_size=0.001))
    all_binned_spikes.append(binned_spikes)
    print(f'Average {brain_region} firing rate: {np.sum(binned_spikes) / len(all_timings[i]) / _total_length} Hz.')


pag_df = extract_sim_as_df(sim_data_path, 'PAG')
pag_timings = extract_timings(pag_df, 'PAG')
pag_binned_spikes = bin_spikes(pag_timings, start_time=0, end_time=_total_length, bin_size=0.001)
print(f'Average PAG firing rate: {np.sum(pag_binned_spikes) / len(pag_timings) / _total_length} Hz.')

X_ones_step = np.vstack(all_binned_spikes)
X = np.repeat(X_ones_step, len(pag_df), axis=1) # repeat input timings (n_PAG_neuron) times
y = pag_binned_spikes.swapaxes(0,1).reshape(1,-1) # flatten all pag neurons out to (1, T * n_PAG_neurons)
assert X.shape[1] == y.shape[1]

if not use_torch:

    glm = BernoulliGLMwReg(
        n_neurons_per_group=np.array([len(df) for df in all_dfs]),
        reg_params = 0 * np.ones(len(brain_regions)),
        link_fn='logistic',
        init_strategy='gaussian',
        seed=123,
        reg='min_weights_within_group'
        )

    glm.fit(X, y,
            mode='sgd', 
            n_iter=250,
            lr=0.0001,
            decay=0.99,
            threshold=1e-2,
            verbose=1)
else:    
    import torch
    import torch.nn as nn
    import torch.distributions as dist
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    glm = BernoulliGLMPyTorch(
        n_neurons_per_group=np.array([len(df) for df in all_dfs]),
        link_fn='logistic',
        reg_params=1
        ).to(device)
    glm.fit(X.T, y.T, n_iter=10000, lr=1e-3, verbose=1)
