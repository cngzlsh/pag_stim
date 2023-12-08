'''
preprocess_presynaptic_spikes.py: sorts data (synthetic or biological) into binned spikes, then optionally performs convolution.
'''

from utils import *

source_folder = './sim/save/pagsim_w_stimuli/' # fd + brain_region + '_spike_times.json'
synthetic = True
pre_convolve_spikes = True
n_PAG = 8
bin_size = 0.001 # secs
pre_convolve_spikes = True
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
        timings = extract_timings(df, brain_region)
        all_timings.append(timings)
        
    _total_length = np.max([total_length(df) for df in all_dfs])
    
    for i, brain_region in enumerate(brain_regions):
        binned_spikes = (bin_spikes(all_timings[i], start_time=0, end_time=_total_length, bin_size=bin_size))
        all_binned_spikes.append(binned_spikes)
        print(f'Average {brain_region} firing rate: {np.sum(binned_spikes) / len(all_timings[i]) / _total_length} Hz.')
        
    X_binned = np.vstack(all_binned_spikes)
    X_smooth = np.zeros_like(X_binned)
    if pre_convolve_spikes:
        for neuron in tqdm(range(X_binned.shape[0])):
            X_smooth[neuron,:] = convolve_spike_train(X_binned[neuron,:], bin_size=1, kernel=kernel, kernel_params=kernel_param)
    
    if save_folder == 'same':
        save_folder = source_folder
        if pre_convolve_spikes:
            np.save(f'{save_folder}presyn_{kernel}_smooth.npy', X_smooth, allow_pickle=True)
        else:
            np.save(f'{save_folder}presyn_binned_{str(bin_size)}ms.npy', X_binned, allow_pickle=True)
        print(f'File saved to {source_folder}')
    
    