
from networkx import is_connected
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.stats as stats
from torch import threshold
from tqdm import tqdm
import pickle
from utils import *
            
def xcorr(x, y, mode='summed'):
    # x: input neuron signal, of shape (n_input_neurons, bins)
    # y: pag signal, of shape (bins)
    corr, lags = [], []
    
    for n in tqdm(range(x.shape[0])):
        if mode == 'summed':
            corr.append(signal.correlate(x[n,:], y, mode='same'))
        lags.append(signal.correlation_lags(len(x[n,:]), len(y), mode='same'))
        
    return lags, corr

            
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
        plot_cross_corr(corrs, max_lag, conns, n_neurons_per_group, n_excitatory_cells_per_group=([64,64,64,64,64]),)
    


