
from networkx import is_connected
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.stats as stats
from torch import threshold
from tqdm import tqdm
import pickle
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
        correlation = [np.correlate(input_spike_i, 
                                    np.concatenate(
                                        (np.zeros(lag), np.roll(output_spike_data, lag)[lag:])
                                                   ))[0]
                       for lag in range(0, max_lag + 1)]
        corrs[f'Input neuron {i}'] = correlation
    
    return corrs

def compute_cross_corr_single_output_pearson(input_spike_data, output_spike_data, max_lag):
    n_input_neurons = input_spike_data.shape[0]
    
    if len(output_spike_data.shape) == 2:
        if output_spike_data.shape[0] == 1:
            output_spike_data = output_spike_data.squeeze()
        else:
            raise ValueError('Function must be called with single output neuron')
    
    corrs = {}
    
    for i in tqdm(range(n_input_neurons)):
        if n_input_neurons > 1:
            input_spike_i = input_spike_data[i]
        else:
            input_spike_i = input_spike_data
        correlation = [np.corrcoef(input_spike_i,
                                #    np.concatenate(
                                #         (np.zeros(lag), np.roll(output_spike_data, lag)[lag:]))
                                   np.roll(output_spike_data, lag))[0,1]
                       for lag in range(-max_lag, max_lag + 1)]
        corrs[f'Input neuron {i}'] = correlation
    
    return corrs

def plot_cross_corr(corrs, max_lag, conns, pag_idx=0):
    lags = np.arange(-max_lag, max_lag + 1)
    
    plt.figure()
    
    for i, corr in enumerate(corrs.values()):
        is_inh, is_connected = check_if_inh_and_connected(i, n_neurons_per_group, conns, pag_idx, thresh=64) # type: ignore
        
        if is_inh:
            c = 'r'
        else:
            c = 'b' if is_connected else 'g'
            
        plt.plot(lags, corr, c=c)
    
    plt.xlabel('Lag')
    plt.ylabel('Cross-Correlation')
    plt.title('Cross-Correlation between Input Neurons and Output Neuron')
    # plt.legend()
    plt.show()
    
    for group_i, n_neurons in enumerate(n_neurons_per_group):
        for j in range(n_neurons):
            cumulative_index = int(np.concatenate((np.zeros(1), np.cumsum(n_neurons_per_group)))[group_i] + j)
            if j > n_excitatory_cells_per_group[group_i] -1:
                c = 'r'
            else:
                c = 'b'
            # plt.plot(lags[cumulative_index][int(lags[0].shape[0]/2-max_lag):int(lags[0].shape[0]/2+max_lag)], corr[cumulative_index][int(lags[0].shape[0]/2-max_lag):int(lags[0].shape[0]/2+max_lag)], c=c)
            plt.plot(lags, list(corrs.values())[cumulative_index], c=c)
            
def xcorr(x, y, mode='summed'):
    # x: input neuron signal, of shape (n_input_neurons, bins)
    # y: pag signal, of shape (bins)
    corr, lags = [], []
    
    for n in tqdm(range(x.shape[0])):
        if mode == 'summed':
            corr.append(signal.correlate(x[n,:], y, mode='same'))
        lags.append(signal.correlation_lags(len(x[n,:]), len(y), mode='same'))
        
    return lags, corr

def check_if_inh_and_connected(i, n_neurons_per_group, conns, pag_idx, thresh=64):
    '''
    Helper function.
    Given the index of input neuron, check if this neuron is an inhibitory neuron and if not, if it's connected to the PAG
    '''
    cum_neurons_per_group = np.cumsum(n_neurons_per_group)
    for n in range(len(n_neurons_per_group)):
        if i < cum_neurons_per_group[n]:
            if n == 0:
                return i > thresh -1, conns[n][pag_idx, i] >0
            else:
                i = i - cum_neurons_per_group[n-1]
                return i > thresh -1, conns[n][pag_idx, i] >0
            
def peak_detection(corr, lags, threshold=0.001):
    peaks = []
    length = len(lags)
    
    for i in range(1, length - 1):
        if abs(corr[i]) > threshold and (
            (corr[i] > corr[i-1] and corr[i] > corr[i+1]) or
            (corr[i] < corr[i-1] and corr[i] < corr[i+1])
        ):
            peaks.append(lags[i])
        
    return peaks

def peak_detection_multi_neurons(corrs, lags, threshold=0.001):
    n_input_neurons = len(list(corrs.values()))
    all_peaks = []
    for n in range(n_input_neurons):
        peaks = peak_detection(list(corrs.values())[n], lags, threshold=threshold)
        all_peaks.append(peaks)
    all_peaks_flattened = [x for peaks in all_peaks for x in peaks]
    return np.unique(np.array(all_peaks_flattened), return_counts=True)
    

if __name__ == '__main__':

    sim_data_path = './sim/save/pagsim_w_stimuli_600s_inh32/'
    brain_regions = [
                    'VMH',
                    'ACC',
                    'IC',
                    'SC',
                    'PMD'
                    ]
    with open(f'{sim_data_path}conns.pkl', 'rb') as f:
        conns = pickle.load(f)
    
    presyn_binned = np.load(sim_data_path+'presyn_binned.npy')
    
    presyn_smooth = np.load(sim_data_path+'presyn_exponential_smooth.npy')
    
    bin_size = 0.001
    n_excitatory_cells_per_group = [64, 64, 64, 64, 64]
    n_PAG_to_use = 1
    n_input_neurons, n_bins = presyn_binned.shape
    n_neurons_per_group = np.load(sim_data_path+'n_neurons_per_group.npy')
    _total_length = n_bins / 1000
    pag_df = extract_sim_as_df(sim_data_path, 'PAG')
    pag_timings = extract_timings(pag_df, 'PAG')
    pag_binned_spikes = bin_spikes(pag_timings, start_time=0, end_time=_total_length, bin_size=0.001)[:n_PAG_to_use]
    print(f'Average PAG firing rate: {np.sum(pag_binned_spikes) / pag_binned_spikes.shape[0] / _total_length} Hz.') # type: ignore

    print(n_neurons_per_group)

    max_lag = 10
    lags = np.arange(-max_lag, max_lag + 1)
    for p in range(n_PAG_to_use):
        # lags, corr = xcorr(presyn_binned[:,:], pag_binned_spikes[p,:])
       
        corrs = compute_cross_corr_single_output_pearson(presyn_binned, pag_binned_spikes[p,:], max_lag)
        plot_cross_corr(corrs, max_lag, conns)
    
    # plt.figure()
    # inh_count = 0
    # for group_i, n_neurons in enumerate(n_neurons_per_group):
    #     for j in range(n_neurons):
    #         cumulative_index = int(np.concatenate((np.zeros(1), np.cumsum(n_neurons_per_group)))[group_i] + j)
    #         if j > n_excitatory_cells_per_group[group_i] -1:
    #             c = 'r'
    #         else:
    #             c = 'b'
    #         plt.plot(lags[cumulative_index][int(lags[0].shape[0]/2-max_lag):int(lags[0].shape[0]/2+max_lag)], corr[cumulative_index][int(lags[0].shape[0]/2-max_lag):int(lags[0].shape[0]/2+max_lag)], c=c)
            
    #         if c == 'r':
    #             inh_count += 1
    #             print(cumulative_index)
    #     # plt.plot(lags[i][int(lags[0].shape[0]/2-max_lag):int(lags[0].shape[0]/2+max_lag)], corr[i][int(lags[0].shape[0]/2-max_lag):int(lags[0].shape[0]/2+max_lag)], c=c)
    # # plt.xticks(np.arange(-max_lag, max_lag))
    # plt.title(f'Total {inh_count} inhibitory (red) and {n_input_neurons-inh_count} excitatory (blue) neurons found.')
    # plt.show()

