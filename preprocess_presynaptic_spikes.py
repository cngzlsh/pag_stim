'''
preprocess_presynaptic_spikes.py: sorts data (synthetic or biological) into binned spikes, then optionally performs convolution.
'''

from utils import *
import numpy as np
import pandas as pd
import pickle
    
np.random.seed(1234)


source_folder = './sim/save/pagsim_w_stimuli_1s_inh/' # fd + brain_region + '_spike_times.json'
synthetic = True
pre_convolve_spikes = True
n_PAG = 8
bin_size = 0.001 # secs
kernel = 'exponential'
kernel_param = {'tau': 15}
save_folder = 'same'



brain_regions = [
                'VMH',
                'ACC',
                'IC',
                'SC',
                'PMD'
                ]

if __name__ == '__main__':
    all_dfs = []
    all_timings = []
    all_binned_spikes = []
    
    for brain_region in brain_regions:
        if synthetic:
            df = extract_sim_as_df(source_folder, brain_region)
        else:
            raise NotImplementedError
        all_dfs.append(df)
        # timings = extract_timings(df, brain_region)
        # all_timings.append(timings)
        
    _total_length = np.max([total_length(df) for df in all_dfs])
    
    # for i, brain_region in enumerate(brain_regions):
    #     binned_spikes = (bin_spikes(all_timings[i], start_time=0, end_time=_total_length, bin_size=bin_size))
    #     all_binned_spikes.append(binned_spikes)
    #     print(f'Average {brain_region} firing rate: {np.sum(binned_spikes) / len(all_timings[i]) / _total_length} Hz.')
    
    # process inhibitory neurons
    inh_df = extract_sim_as_df(source_folder, 'InhNeuron')
    n_inh_neurons = len(inh_df)
    inh_split_cumu_idx = list(map(int, np.concatenate((np.zeros(1), np.cumsum(np.random.dirichlet(alpha=np.array([1,1,1,1,1]))) * 32))))
    inh_split_idx = [list(np.arange(inh_split_cumu_idx[i], inh_split_cumu_idx[i+1])) for i in range(len(brain_regions))]
    for i, brain_region in enumerate(brain_regions):
        for j in inh_split_idx[i]:
            inh_df.at[f'InhNeuron{j}', 'brain_region'] = brain_region
        all_dfs[i] = pd.concat((all_dfs[i], inh_df.iloc[inh_split_idx[i]]))
        
        timings = extract_timings(all_dfs[i], brain_region)
        all_timings.append(timings)
        binned_spikes = (bin_spikes(all_timings[i], start_time=0, end_time=_total_length, bin_size=bin_size))
        all_binned_spikes.append(binned_spikes)
        print(f'Average {brain_region} firing rate: {np.sum(binned_spikes) / len(all_timings[i]) / _total_length} Hz.')
    
    # update conns
    conns = [np.nan_to_num(np.load(f'{source_folder}connectivity_{brain_region.lower()}2pag.npy'),  nan=0.0).T * 1e9 for brain_region in brain_regions]
    inh_conns = np.nan_to_num(np.load(f'{source_folder}connectivity_inh2pag.npy'),  nan=0.0).T * 1e9
    for i, brain_region in enumerate(brain_regions):
        conns[i] = np.hstack((conns[i], inh_conns[:,inh_split_idx[i]]))
        assert conns[i].shape[1] == len(all_dfs[i])
    
    
    n_neurons_per_group = np.array([len(i) for i in all_dfs])
    
    X_binned = np.vstack(all_binned_spikes)
    X_smooth = np.zeros_like(X_binned)
    if pre_convolve_spikes:
        for neuron in tqdm(range(X_binned.shape[0])):
            X_smooth[neuron,:] = convolve_spike_train(X_binned[neuron,:], bin_size=1, kernel=kernel, kernel_params=kernel_param)
    
    if save_folder == 'same':
        save_folder = source_folder
        if pre_convolve_spikes:
            print('saving convolved...')
            np.save(f'{save_folder}presyn_{kernel}_smooth.npy', X_smooth, allow_pickle=True)
        else:
            print('saving binned ...')
            np.save(f'{save_folder}presyn_binned.npy', X_binned, allow_pickle=True)
        np.save(f'{save_folder}n_neurons_per_group.npy', n_neurons_per_group, allow_pickle=True)
        print(f'File saved to {source_folder}')
        
        with open(f'{save_folder}conns.pkl', 'wb') as f:
            pickle.dump(conns, f)
    
    