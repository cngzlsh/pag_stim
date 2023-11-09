import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

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



if __name__ == '__main__':
    DATA_PATH = "./bio/data/1103076/230706_npx_all_goods_spikePropertiesByClust.json"

    df = extract_rec_as_df(DATA_PATH)
    pag_spike_timings = extract_timings(df, 'PAG')
    pag_binned_spikes = bin_spikes(pag_spike_timings, start_time=0, end_time=total_length(df), bin_size=0.001)
    print(f'Average PAG firing rate: {np.sum(pag_binned_spikes) / len(pag_spike_timings) / total_length(df)} Hz.')
    
    vmh_spike_timings = extract_timings(df, 'VMH')
    vmh_binned_spikes = bin_spikes(vmh_spike_timings, start_time=0, end_time=total_length(df), bin_size=0.001)
    print(f'Average VMH firing rate: {np.sum(vmh_binned_spikes) / len(vmh_spike_timings) / total_length(df)} Hz.')
    
    # acc_spike_timings = extract_timings(df, 'ACC')
    # acc_binned_spikes = bin_spikes(acc_spike_timings, start_time=0, end_time=total_length(df), bin_size=0.001)
    # print(f'Average ACC firing rate: {np.sum(acc_binned_spikes) / len(acc_spike_timings) / total_length(df)} Hz.')
    
    pmd_spike_timings = extract_timings(df, 'PMd')
    pmd_binned_spikes = bin_spikes(pmd_spike_timings, start_time=0, end_time=total_length(df), bin_size=0.001)
    print(f'Average PMd firing rate: {np.sum(pmd_binned_spikes) / len(pmd_spike_timings) / total_length(df)} Hz.')
    
    # sc_spike_timings = extract_timings(df, 'SC')
    # sc_binned_spikes = bin_spikes(sc_spike_timings, start_time=0, end_time=total_length(df), bin_size=0.001)
    # print(f'Average SC firing rate: {np.sum(sc_binned_spikes) / len(sc_spike_timings) / total_length(df)} Hz.')
    
    icd_spike_timings = extract_timings(df, 'ICd')
    icd_binned_spikes = bin_spikes(icd_spike_timings, start_time=0, end_time=total_length(df), bin_size=0.001)
    print(f'Average ICd firing rate: {np.sum(icd_binned_spikes) / len(icd_spike_timings) / total_length(df)} Hz.')
