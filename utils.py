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
    df.index = [brain_region + i for i in data.keys()]
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

def convolve_spike_train(spike_bins, bin_size=1, kernel='exponential', kernel_params={'tau':20}):
    n_bins = spike_bins.shape[-1]
    convolved_signals = np.zeros_like(spike_bins)
    t_vector = np.arange(0, n_bins*bin_size, bin_size)
    tau = kernel_params['tau']
    exponential_kernel = np.exp(-t_vector / tau)
    
    if len(spike_bins.shape) == 2:
        n_neurons, _ = spike_bins.shape
        # Create the time vector
        convolved_signals = np.zeros_like(spike_bins)
        for n in tqdm(range(n_neurons)):
            convolved_signals[n, :] = np.convolve(spike_bins[n,:], exponential_kernel, mode='same')
    
    elif len(spike_bins.shape) == 1:
        n_bins = spike_bins.shape[0]
        
        if kernel == 'exponential':
            convolved_signals = np.convolve(spike_bins, exponential_kernel, mode='same')
    
    return convolved_signals

class BNN_Dataset(Dataset):
    '''
    Dataset class for creating iterable dataloader
    '''
    def __init__(self, X, Y):
        self.inputs = X
        self.labels = Y

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input = self.inputs[idx,:]
        label = self.labels[idx,:]
        return input, label