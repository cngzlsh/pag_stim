
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def compute_cross_corr_single_output(input_spike_data, output_spike_data, max_lag):
    n_input_neurons = input_spike_data.shape[0]
    
    if len(output_spike_data.shape) == 2:
        if output_spike_data.shape[0] == 1:
            output_spike_data = output_spike_data.squeeze()
        else:
            raise ValueError('Function must be called with single output neuron')
    
    corrs = {}
    
    for i in range(n_input_neurons):
        if n_input_neurons > 1:
            input_spike_i = input_spike_data[i]
        else:
            input_spike_i = input_spike_data
        correlation = [np.correlate(input_spike_i, np.roll(output_spike_data, lag))[0]
                       for lag in range(-max_lag, max_lag + 1)]
        corrs[f'Input neuron {i}'] = correlation
    
    return corrs

def plot_cross_corr(corrs, max_lag):
    lags = np.arange(-max_lag, max_lag + 1)
    
    plt.figure()
    
    for neuron, corr in corrs.items():
        plt.plot(lags, corr, label=neuron)
    
    plt.xlabel('Lag')
    plt.ylabel('Cross-Correlation')
    plt.title('Cross-Correlation between Input Neurons and Output Neuron')
    plt.legend()
    plt.show()
        


sim_data_path = './sim/save/pagsim_w_stimuli_1s_inh/'
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
n_PAG_to_use = 1
n_input_neurons, n_bins = presyn_binned.shape
n_neurons_per_group = np.load(sim_data_path+'n_neurons_per_group.npy')
_total_length = n_bins / 1000
pag_df = extract_sim_as_df(sim_data_path, 'PAG')
pag_timings = extract_timings(pag_df, 'PAG')
pag_binned_spikes = bin_spikes(pag_timings, start_time=0, end_time=_total_length, bin_size=0.001)[:n_PAG_to_use]
print(f'Average PAG firing rate: {np.sum(pag_binned_spikes) / pag_binned_spikes.shape[0] / _total_length} Hz.')


corrs = compute_cross_corr_single_output(presyn_binned[0:5,:], pag_binned_spikes, 10)
plot_cross_corr(corrs, 10)
