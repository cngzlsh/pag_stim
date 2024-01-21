from re import T
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.data import Dataset
import pandas as pd
from loguru import logger
from tqdm import tqdm
import os
import pickle

def extract_rec_as_df(fd):
    import json
    with open(fd) as f:
        logger.debug(f'Reading recording file {fd}')
        data = json.load(f)
    return pd.DataFrame(data['data'], index=data['index'], columns=data['columns'])

def extract_sim_as_df(fd, brain_region):
    '''
    Creates a data frame of recording times from brian2 simulations.
    '''
    import json
    with open(fd + brain_region + '_spike_times.json') as f:
        data = json.load(f)
    
    df = pd.DataFrame(index=data.keys(), columns=['spikeTimes', 'brain_region'])
    df.index = [brain_region + i for i in data.keys()] # type: ignore
    df['spikeTimes'] = [np.array(l)/1000 for l in list(data.values())] # convert ms to s
    df['brain_region'] = brain_region
    return df

def extract_timings(df, brain_region):
    '''
    Extract spike timings of all neurons in a specific brain region.
    Returns a dataframe of lists.
    '''
    if brain_region not in np.unique(df["brain_region"]):
        raise ValueError(f'{brain_region} is not in the list of recorded region: {list(np.unique(df["brain_region"]))}.')

    idxs = df.index[df['brain_region'] == brain_region]
    spike_times = df.loc[idxs]['spikeTimes']
    return spike_times

def extract_all_simulation_timings(source_folder, has_inh, brain_regions):
    
    '''
    This function extracts all spikes trains, 
    preprocesses any inhibitory neurons, 
    and parse them into timings. 
    Updates the connectivity file to incorporate 
    inhibitory neurons accordingly.
    '''
    
    for brain_region in brain_regions:
        assert os.path.exists(f'{source_folder}+connectivity_{brain_region.lower()}2pag.npy')
    assert os.path.exists(source_folder+'connectivity_inh2pag.npy')
    
    all_dfs = []
    all_timings = []

    for brain_region in brain_regions:
        df = extract_sim_as_df(source_folder, brain_region)
        all_dfs.append(df)
    _total_length = np.max([total_length(df) for df in all_dfs])
    
    if has_inh:
        inh_df = extract_sim_as_df(source_folder, 'InhNeuron')
        inh_split_cumu_idx = list(map(int, np.concatenate((np.zeros(1), np.cumsum(np.random.dirichlet(alpha=np.array([1,1,1,1,1]))) * len(inh_df)))))
        inh_split_idx = [list(np.arange(inh_split_cumu_idx[i], inh_split_cumu_idx[i+1])) for i in range(len(brain_regions))]

    for i, brain_region in enumerate(brain_regions):
        if has_inh:
            for j in inh_split_idx[i]:
                inh_df.at[f'InhNeuron{j}', 'brain_region'] = brain_region
            all_dfs[i] = pd.concat((all_dfs[i], inh_df.iloc[inh_split_idx[i]]))
        
        timings = extract_timings(all_dfs[i], brain_region)
        all_timings.append(timings)
    
    try:
        with open(f'{source_folder}conns_inh.pkl', 'rb') as f:
            conns = pickle.load(f)
    except:
        conns = [np.nan_to_num(np.load(f'{source_folder}connectivity_{brain_region.lower()}2pag.npy'),  nan=0.0).T * 1e9 for brain_region in brain_regions]
        
        if has_inh:
            inh_conns = -np.nan_to_num(np.load(f'{source_folder}connectivity_inh2pag.npy'),  nan=0.0).T * 1e9
            for i, brain_region in enumerate(brain_regions):
                conns[i] = np.hstack((conns[i], inh_conns[:,inh_split_idx[i]]))
                assert conns[i].shape[1] == len(all_dfs[i])
                
        with open(f'{source_folder}conns_inh.pkl', 'wb') as f:
                pickle.dump(conns, f)

    return all_timings, _total_length, conns

def create_sparse_binned_smooth_spikes(spike_timings, start_time, end_time, bin_size, convolve=True, kernel='exponential', kernel_params={'tau':15}):
    '''
    This function pre-processes the spike timings into sparse tensors.
    :params:
    spike_timings:      pandas dataframe or list of spike timings.
    bin_size:           bin size in seconds
    convolve:           whether to produce sparse matrix for convolution
    '''
    dense_binned_spikes = bin_spikes(spike_timings, start_time, end_time, bin_size=0.001)
    
    sparse_binned_by_neuron, sparse_convolved_by_neuron = [], []
    num_non_zeros_binned_by_neuron, num_non_zeros_sparse_by_neuron, recording_length = [], [], []
    
    for n, timings in enumerate(spike_timings):
        sparse_binned = np.zeros(3) # R: neuron_idx, C: bin_idx, V: value
        
        num_non_zeros_binned = 0
        num_non_zeros_sparse = 0
        recording_length = 0
        
        for t in timings:
            sparse_binned = np.vstack((sparse_binned, np.array([n, int(t/bin_size), 1])))
            num_non_zeros_binned += 1
            
        sparse_binned_by_neuron.append(sparse_binned_by_neuron)
        num_non_zeros_binned_by_neuron.append(num_non_zeros_binned)
        
        if convolve:
            dense_convolved = convolve_spike_train(dense_binned_spikes[n], kernel=kernel, kernel_params=kernel_params)
            sparse_convolved = np.zeros(3) # R: neuron_idx, C: bin_idx, V: value
            for b, v in enumerate(dense_convolved):
                if v != 0:
                    sparse_convolved = np.vstack((sparse_convolved, np.array([n, b, v])))
                    num_non_zeros_sparse += 1
            
            sparse_convolved_by_neuron.append(sparse_convolved)
            num_non_zeros_sparse_by_neuron.append(num_non_zeros_sparse)
        
            
        
            
    
    
    pass

def bin_spikes(spike_timings, start_time, end_time, bin_size=0.001):
    '''
    Create binary response variables at specified bin size for each neuron.
    :params:
    spike_timings:      pandas dataframe or list of spike timings.
    total_length:       length of recording in seconds
    bin_size:           bin size in seconds
    
    :returns:
    binned_spikes:      binary numpy array of shape (n_neurons, n_bins)
    '''
    n_neurons = len(spike_timings)
    n_bins = int(np.ceil((end_time - start_time)/bin_size))
    binned_spikes = np.zeros((n_neurons, n_bins))
    
    for n, ts in enumerate(spike_timings):
        for t in ts:
            if t > start_time and t < end_time:
                binned_spikes[n, int(np.floor(t/bin_size))] = 1
    logger.debug(f'Created spike bin matrix with {n_neurons} neurons and {n_bins} bins. Total number of spikes: {np.sum(binned_spikes)}')
    
    return binned_spikes
    

def total_length(df):
    '''
    Returns the timing of the last spike in the dataframe.
    '''
    return np.max([np.max(t) for t in df['spikeTimes']])

def normalise_array(array):
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    return (array - np.mean(array)) / np.std(array)

def generate_bernoulli_spikes(probs):
    return dist.Bernoulli(probs=probs).sample()

def rbf_rate_convolution_1d(spikes, sigma, dt=0.01):
    
    # Convolution with Gaussian kernel
    time_window = np.arange(-5*sigma, 5*sigma, dt)
    gaussian_kernel = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * (time_window / sigma)**2)
    smooth_rate = np.convolve(spikes, gaussian_kernel, mode='same')
    
    return smooth_rate

def convolve_spike_train(spike_bins, kernel='exponential', kernel_params={'tau':15}):
    n_bins = spike_bins.shape[-1]
    convolved_signals = np.zeros_like(spike_bins)
    
    if kernel != 'exponential':
        raise NotImplementedError
    tau = kernel_params['tau']
    # exponential_kernel = np.exp(-t_vector / tau) # kernel = np.exp(-np.arange(0, 5 * tau) / tau)
    exponential_kernel = np.concatenate((np.zeros(5*tau-1), np.exp(-np.arange(0, 5 * tau) / tau)))
    
    if len(spike_bins.shape) == 2:
        n_neurons, _ = spike_bins.shape
        # Create the time vector
        convolved_signals = np.zeros_like(spike_bins)
        for n in tqdm(range(n_neurons)):
            convolved_signals[n, :] = np.convolve(spike_bins[n,:], exponential_kernel, mode='same') # (n_neurons, n_bins)
    
    elif len(spike_bins.shape) == 1:
        
        convolved_signals = np.convolve(spike_bins, exponential_kernel, mode='same') # (n_bins)
    
    return convolved_signals

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

def compute_cross_corr_single_output_pearson(input_spike_data, output_spike_data, max_lag, use_torch=False):
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
        
        if use_torch:
            correlation = [torch.corrcoef(input_spike_i,
                                    #    np.concatenate(
                                    #         (np.zeros(lag), np.roll(output_spike_data, lag)[lag:]))
                                    torch.roll(output_spike_data, lag))[0,1] # type: ignore
                        for lag in range(-max_lag, max_lag + 1)]
        else:
            correlation = [np.corrcoef(input_spike_i,
                                    #    np.concatenate(
                                    #         (np.zeros(lag), np.roll(output_spike_data, lag)[lag:]))
                                    np.roll(output_spike_data, lag))[0,1]
                        for lag in range(-max_lag, max_lag + 1)]
        
        corrs[f'Input neuron {i}'] = correlation
    
    return corrs

def plot_cross_corr(corrs, max_lag, conns, n_neurons_per_group, n_excitatory_cells_per_group=([64,64,64,64,64]), pag_idx=0):
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
            
if __name__ == '__main__':