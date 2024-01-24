from re import T
from xml.dom import INDEX_SIZE_ERR
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
from pathlib import Path
import json

#%% utils for dealing with simulation data

def load_dfs(read_path, simulation, n_PAG_to_use = 'all'):
    '''
    read_path : 
        if simulation is True: read_path is a folder
        if not simulation: read_path is a file path

    '''    
    if simulation:
        presyn_df, _, _ = extract_all_presynaptic_simulation_timings(read_path, has_inh=True, brain_regions=['VMH','ACC', 'IC','SC','PMD'])

        pag_df = extract_sim_as_df(read_path, 'PAG')
        
        if n_PAG_to_use != 'all':
            pag_df = pag_df[:n_PAG_to_use]
        
        
    else:
        
        df = pd.read_json(read_path, orient = 'split') # ['spikeTimesDev3_cut', 'cleaned_brain_region', 'brain_region']
        df = df.rename(columns = {'spikeTimesDev3_cut': 'spikeTimes'})
        
        presyn_df = df.loc[df['cleaned_brain_region'].isin(['SC', 'IC', 'VMH', 'ACC', 'PMd'])]
        pag_df = df.loc[df['cleaned_brain_region'].isin(['PAG'])]
        
        if n_PAG_to_use != 'all':
            pag_df = pag_df.iloc[:n_PAG_to_use]        
    
    last_spike_time = np.max([np.max([np.max(timings) for timings in df]) for df in [presyn_df['spikeTimes'], pag_df['spikeTimes']]])
    
    return presyn_df, pag_df, last_spike_time

    
def extract_rec_as_df(fd):
    ''' THIS FUNCTION DOESNT SEEM TO BE USED. DELETE?
    '''

    with open(fd) as f:
        logger.debug(f'Reading recording file {fd}')
        data = json.load(f)
    return pd.DataFrame(data['data'], index=data['index'], columns=data['columns'])

def extract_sim_as_df(fd, brain_region):
    '''
    Creates a data frame of recording times from brian2 simulations.
    '''
    
    fd = Path(fd)
    
    with open((fd / (brain_region + '_spike_times.json'))) as f:
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

def extract_all_presynaptic_simulation_timings(source_folder, has_inh, brain_regions):
    
    '''
    This function extracts all spikes trains, 
    preprocesses any inhibitory neurons, 
    and parse them into timings. 
    Updates the connectivity file to incorporate 
    inhibitory neurons accordingly.
    '''
    source_folder = Path(source_folder)
    
    for brain_region in brain_regions:
        assert (source_folder / ('connectivity_' + brain_region.lower() + '2pag.npy')).exists()
    assert (source_folder / 'connectivity_inh2pag.npy').exists()
    
    all_dfs = []
    all_timings = []

    for brain_region in brain_regions:
        df = extract_sim_as_df(source_folder, brain_region)
        all_dfs.append(df)
        
    _total_length = np.max([np.max([np.max(t) for t in df['spikeTimes']]) for df in all_dfs])
    
    if has_inh:
        inh_df = extract_sim_as_df(source_folder, 'InhNeuron')
        inh_split_cumu_idx = list(map(int, np.concatenate((np.zeros(1), (np.cumsum(np.random.dirichlet(alpha=np.array([1,1,1,1,1])))+0.01)  * len(inh_df))))) # numerical stability
        inh_split_idx = [list(np.arange(inh_split_cumu_idx[i], inh_split_cumu_idx[i+1])) for i in range(len(brain_regions))]

    for i, brain_region in enumerate(brain_regions):
        if has_inh:
            inh_df_temp = inh_df.iloc[inh_split_idx[i]].copy()
            inh_df_temp.index=[f'{brain_region}_IN{j}' for j in inh_split_idx[i]]
            inh_df_temp['brain_region'] = brain_region
                
            all_dfs[i] = pd.concat((all_dfs[i], inh_df_temp)) # type: ignore
    
    conns = [np.nan_to_num(np.load((source_folder / ('connectivity_' + brain_region.lower() + '2pag.npy'))),  nan=0.0).T * 1e9 for brain_region in brain_regions]
    
    if has_inh:
        inh_conns = -np.nan_to_num(np.load((source_folder / 'connectivity_inh2pag.npy')),  nan=0.0).T * 1e9
        for i, brain_region in enumerate(brain_regions):
            conns[i] = np.hstack((conns[i], inh_conns[:,inh_split_idx[i]])) # type: ignore
            assert conns[i].shape[1] == len(all_dfs[i])
            
    with open((source_folder /'conns_inh.pkl'), 'wb') as f:
            pickle.dump(conns, f)
    
    all_dfs = pd.concat(all_dfs)
    
    return all_dfs, _total_length, conns

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

#%% sparse arrays

def sparsify_spike_train(spike_train, start_time, end_time, bin_size=0.001):
    if not isinstance(spike_train, np.ndarray): spike_train = np.array(spike_train)
    time_bin_idxs = np.floor(spike_train[(spike_train > start_time) & (spike_train < end_time+bin_size)]/bin_size).astype(int)
    sparse_binary = np.zeros((len(time_bin_idxs), 3)) 
    sparse_binary[:, 0] = 0 
    sparse_binary[:, 1] = time_bin_idxs
    sparse_binary[:, 2] = 1 
    
    return sparse_binary

#%%

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
            
# if __name__ == '__main__':
#     read_path = './sim/save/pagsim_w_stimuli_600s_inh32/'
#     input_sparse, output_sparse = aggregate_multiple_sparse_matrices(read_path)
    
#     from glm import BernoulliGLMPyTorch

#     n_neurons_per_group = np.load(read_path + 'n_neurons_per_group.npy')
#     glm = BernoulliGLMPyTorch(
#     group_names=['VMH',
#                 'ACC',
#                 'IC',
#                 'SC',
#                 'PMD'
#                 ],
#     n_neurons_per_group=n_neurons_per_group,
#     link_fn='logistic',
#     n_sessions=16,
#     regs = ['weights_within_group',
#             'weights_sparsity'],
#     reg_params={'weights_within_group':100,
#                 'weights_sparsity':100,
#                 }, # type: ignore
#     ).to('cuda')
        

#     glm.fit(input_sparse.T, output_sparse.T, n_iter=20000, lr=1e-3, verbose=3, decay=0.9999, batch_size=1024)