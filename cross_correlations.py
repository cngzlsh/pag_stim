
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.stats as stats
from torch import threshold
from tqdm import tqdm
import pickle
from utils import sparsify_spike_train, extract_sim_as_df, extract_timings, load_dfs
from scipy import sparse
from pathlib import Path

def compute_cross_corr_single_output(binarised_spike_trains_pre, binarised_spike_train_post, max_lag, corr_type = 'pearson'):
    
    n_input_neurons = binarised_spike_trains_pre.shape[0]
    
    if len(binarised_spike_train_post.shape) == 2:
        if binarised_spike_train_post.shape[0] == 1:
            binarised_spike_train_post = binarised_spike_train_post.squeeze()
        else:
            raise ValueError('Function must be called with single output neuron')
    
    corrs = np.zeros(n_input_neurons, max_lag*2+1)
    
    for i in tqdm(range(n_input_neurons)):
        if n_input_neurons > 1:
            binarised_spike_train_pre = binarised_spike_trains_pre[i]
        else:
            binarised_spike_train_pre = binarised_spike_trains_pre
        
        if corr_type == 'pearson':
            correlation = [np.corrcoef(binarised_spike_train_pre,
                                    np.roll(binarised_spike_train_post, lag))[0,1]
                        for lag in range(-max_lag, max_lag + 1)]
        elif corr_type == 'summed':
            correlation = signal.correlate(binarised_spike_train_pre, binarised_spike_train_post, mode='same')
            correlation = correlation[len(correlation)/2-max_lag:len(correlation)/2+max_lag]
            
        corrs[i, :] = correlation
    
    return corrs


def compute_pairwise_cross_corr_sparse(spike_train_pre, spike_train_post, max_lag):
    
    spike_train_pre = sparsify_spike_train(spike_train_pre, 0, np.inf)
    spike_train_post = sparsify_spike_train(spike_train_post, 0, np.inf)
    
    n_bins = int(np.max([np.max(x[:,1]) for x in [spike_train_pre, spike_train_post]]))+1
                             
    def to_scipy_sparse(sparse_binary, n_bins): 
        assert max(sparse_binary[:,1].astype(int)) < n_bins
        return sparse.coo_array((sparse_binary[:,2], (sparse_binary[:,0].astype(int), sparse_binary[:,1].astype(int))), shape = (1, n_bins))

    correlation = np.zeros(max_lag*2+1)
    
    for lag_i, lag in enumerate(range(-max_lag, max_lag + 1)):

        if lag >= 0:
            # for positive lags, shift the presynaptic spike trains to the future 
            spike_train_pre_rolled = spike_train_pre.copy()
            spike_train_pre_rolled[:,1]+=lag
            spike_train_post_rolled = spike_train_post
        elif lag < 0: 
             # for negative lags, shift the postsynaptic spike trains to the future 
             spike_train_post_rolled = spike_train_post.copy()
             spike_train_post_rolled[:,1]-=lag
             spike_train_pre_rolled = spike_train_pre
             
        coeff = scipy_sparse_pearson_corr(to_scipy_sparse(spike_train_pre_rolled, n_bins+np.abs(lag)),
                                          to_scipy_sparse(spike_train_post_rolled, n_bins+np.abs(lag)))
        correlation[lag_i] = coeff
    
    return correlation
    
def scipy_sparse_pearson_corr(A, B=None, sparse_format = 'coo'):
    'inputs are scipy sparse matrices'
    
    if B is not None:
        A = sparse.vstack((A, B), format=sparse_format)
        
    A = A.astype(np.float64)
    n = A.shape[1]

    # Compute the covariance matrix
    rowsum = A.sum(1)
    centering = rowsum.dot(rowsum.T.conjugate()) / n
    C = (A.dot(A.T.conjugate()) - centering) / (n - 1)

    # The correlation coefficients are given by
    # C_{i,j} / sqrt(C_{i} * C_{j})
    d = np.diag(C)
    coeffs = C / np.sqrt(np.outer(d, d))
    
    return coeffs[0,1]



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


# def plot_cross_corr(corrs, max_lag, conns, n_neurons_per_group, n_excitatory_cells_per_group=([64,64,64,64,64]), pag_idx=0):
#     lags = np.arange(-max_lag, max_lag + 1)
    
#     plt.figure()
    
#     # for i, corr in enumerate(corrs):
#     #     is_inh, is_connected = check_if_inh_and_connected(i, n_neurons_per_group, conns, pag_idx, thresh=64) # type: ignore
        
#     #     if is_inh:
#     #         c = 'r'
#     #     else:
#     #         c = 'b' if is_connected else 'g'
            
#     #     plt.plot(lags, corr, c=c)
    
#     plt.xlabel('Lag')
#     plt.ylabel('Cross-Correlation')
#     plt.title('Cross-Correlation between Input Neurons and Output Neuron')
#     # plt.legend()
#     plt.show()
    
#     for group_i, n_neurons in enumerate(n_neurons_per_group):
#         for j in range(n_neurons):
#             cumulative_index = int(np.concatenate((np.zeros(1), np.cumsum(n_neurons_per_group)))[group_i] + j)
#             if j > n_excitatory_cells_per_group[group_i] -1:
#                 c = 'r'
#             else:
#                 c = 'b'
#             # plt.plot(lags[cumulative_index][int(lags[0].shape[0]/2-max_lag):int(lags[0].shape[0]/2+max_lag)], corr[cumulative_index][int(lags[0].shape[0]/2-max_lag):int(lags[0].shape[0]/2+max_lag)], c=c)
#             plt.plot(lags, corrs[cumulative_index,:], c=c)
            

def plot_cross_corr(corrs, max_lag, conns_for_one_pag):
    lags = np.arange(-max_lag, max_lag + 1)
    
    plt.figure()
    
    plt.xlabel('Lag')
    plt.ylabel('Cross-Correlation')
    plt.title('Cross-Correlation between Input Neurons and Output Neuron')
    
    if conns_for_one_pag is not None: 
        c = np.select([conns_for_one_pag==0, conns_for_one_pag>0, conns_for_one_pag<0], ['b','g','r'])
    else:
        c = 'k'
    
    corrs = corrs - np.median(corrs[:,0:int(max_lag*0.75)], axis = 1)[:, np.newaxis]
    plt.plot(lags, corrs.T, c=c)            
            
#%%

def run_cross_corrs(read_path, save_folder, simulation, plot=True, pag_i = None):
    
    presyn_df, pag_df, last_spike_time = load_dfs(read_path, simulation = simulation, n_PAG_to_use=1)
    if pag_i is not None: pag_df = pag_df.iloc[[pag_i]]
    
    if simulation: 
        with open((read_path/'conns_inh.pkl'), 'rb') as f: conns = pickle.load(f)
        conns = np.hstack(conns)
        
    max_lag = 50
    
    for pag_i, spike_train_post in enumerate(pag_df['spikeTimes']):
        corrs = np.zeros((len(presyn_df), max_lag*2+1))
        
        for input_neuron_i, spike_train_pre in enumerate(presyn_df['spikeTimes']):
            correlation = compute_pairwise_cross_corr_sparse(spike_train_pre, spike_train_post, max_lag)
            corrs[input_neuron_i, :] = correlation
            
        if plot:
            if simulation: 
                plot_cross_corr(corrs, max_lag, conns[pag_i,:])
            else:
                plot_cross_corr(corrs, max_lag, None)
        
        if simulation:
            session_name = read_path.stem
        else:
            session_name = read_path.stem[:6]
            
        if save_folder is not None: np.save((save_folder / (session_name + '_cross_corrs_pag' + pag_i + '.npy')), corrs)

if __name__ == '__main__':
    
    read_path = Path(r'Y:\Daniel_npx_modelling\pagsim_w_stimuli_600s_inh32')
    read_path = r'Y:/Daniel_npx_modelling/data/1103076/curated/single_unit_only/230718_JRCcurated_npx_all_cutGLM_goods_spikePropertiesByClust.json'
    save_folder = None
    simulation=False
    pag_i = None    
    plot = True
    
    run_cross_corrs(read_path, save_folder, simulation, plot, pag_i)
    