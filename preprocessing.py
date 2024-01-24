import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from tqdm import tqdm
import os
import pickle
from pathlib import Path
from utils import extract_all_presynaptic_simulation_timings, sparsify_spike_train, load_dfs

### preprocessing functions 

def binarise_spike_trains(spike_timings, start_time, end_time, bin_size=0.001):
    '''
    Create binary response variables at specified bin size for each neuron.
    :params:
    spike_timings:      pandas dataframe or list of spike trains. to run on single spike train, pass as list of length 1 
    bin_size:           bin size in seconds
    
    :returns:
    binned_spikes:      binary numpy array of shape (n_neurons, n_bins)
    '''
    n_neurons = len(spike_timings)
    n_bins = int(np.ceil((end_time - start_time)/bin_size))
    
    if n_neurons > 1:
        binarised_spike_trains = np.zeros((n_neurons, n_bins))
        
        for n, ts in enumerate(spike_timings):
            for t in ts:
                if t > start_time and t < end_time:
                    binarised_spike_trains[n, int(np.floor(t/bin_size))] = 1
        logger.debug(f'Created spike bin matrix with {n_neurons} neurons and {n_bins} bins. Total number of spikes: {np.sum(binarised_spike_trains)}')
    
    else:
        binarised_spike_trains = np.zeros(n_bins)
        for t in spike_timings[0]:
            if t > start_time and t < end_time:
                binarised_spike_trains[int(np.floor(t/bin_size))] = 1       
        
    return binarised_spike_trains

def convolve_spike_trains(binarised_spike_trains, kernel_params={'type': 'exponential', 'tau':15}):
    '''
    Create binary response variables at specified bin size for each neuron.
    :params:
    binarised_spike_trains: binary 2D numpy array of shape (n_neurons, n_bins). for 1 spike train, shape would be (1, n_bins)
                            OR binary 1D numpy array of shape n_bins i.e. shape = (n_bins,)
    output preserves the shape of the input
    '''
    
    if kernel_params['type'] != 'exponential':
        raise NotImplementedError
    tau = kernel_params['tau']
    # exponential_kernel = np.exp(-t_vector / tau) # kernel = np.exp(-np.arange(0, 5 * tau) / tau)
    exponential_kernel = np.concatenate((np.zeros(5*tau-1), np.exp(-np.arange(0, 5 * tau) / tau)))
    
    convolved_spike_trains = np.zeros_like(binarised_spike_trains)
   
    if len(binarised_spike_trains.shape) == 2: # if shape of input is (n_neurons, n_bins)
        n_neurons, _ = binarised_spike_trains.shape
        for n in tqdm(range(n_neurons)):
            convolved_spike_trains[n, :] = np.convolve(binarised_spike_trains[n,:], exponential_kernel, mode='same') # (n_neurons, n_bins)
    
    elif len(binarised_spike_trains.shape) == 1: # if a 1D np array was passed as input, ie shape is (n_bins,)
        convolved_spike_trains = np.convolve(binarised_spike_trains, exponential_kernel, mode='same') # (n_bins)
    
    return convolved_spike_trains


def sparsify_array(dense_array):
    '''
    R: neuron_idx, C: time_bin_idx, V: value 
    
    '''
    
    time_bin_idxs = np.where(dense_array > 0)[0]
    vals = dense_array[time_bin_idxs]
    
    sparse_array = np.zeros((len(time_bin_idxs), 3)) 
    sparse_array[:,1] = time_bin_idxs
    sparse_array[:,2] = vals
    
    return sparse_array
    
    
def create_sparse_preprocessed_spikes(spike_timings, start_time, end_time, bin_size, convolve=True, 
                                       kernel_params={'type': 'exponential', 'tau':15}):
    '''
    This function pre-processes the spike timings into sparse matrices in a single session.
    :params:
    spike_timings:      pandas dataframe or list of spike timings.
    bin_size:           bin size in seconds
    convolve:           whether to produce sparse matrix for convolution
    '''
    if isinstance(spike_timings, list): # input interpreted as multiple brain regions
        try:
            spike_timings = pd.concat(spike_timings)
        except:
            raise ValueError
    
    sparse_processed_by_neuron, num_non_zeros_by_neuron = [], []
    
    for neuron_i, spike_train in enumerate(spike_timings):
        
        if convolve:
            # binarise, convolve and sparsify each presynaptic neuron individually to avoid storing large dense matrix of all neurons
            binarised_spike_train = binarise_spike_trains([spike_train], start_time, end_time+bin_size, bin_size=0.001)
            dense_convolved = convolve_spike_trains(binarised_spike_train, kernel_params=kernel_params)
            
            sparse_convolved = sparsify_array(dense_convolved)
            sparse_convolved[:,0] = neuron_i #shift the row, so that neurons will be in different rows in the combined 2D input matrix
            
            sparse_processed_spike_train = sparse_convolved
        
        else:
            # create sparse matrix directly from spike times. row kept at 0, because needs to be 1D when combined. value is always 1 because binary
            # if not convolved, not necessary to binarise (i.e. make dense) only to then sparsify again. 
            
            sparse_binary = sparsify_spike_train(spike_train, start_time, end_time, bin_size)
            
            # time_bin_idxs = np.floor(spike_train[(spike_train > start_time) & (spike_train < end_time+bin_size)]/bin_size).astype(int)
            # sparse_binary = np.zeros((len(time_bin_idxs), 3)) 
            # sparse_binary[:, 0] = 0 #keep at 0, because needs to be 1D when combined
            # sparse_binary[:, 1] = time_bin_idxs
            # sparse_binary[:, 2] = 1 #because spike_train is binary, value is always 1
  
            assert sparse_binary[0,1] == int(spike_train[0] / bin_size)
            assert sparse_binary[-1,1] == int(spike_train[-1] / bin_size)
            
            sparse_processed_spike_train = sparse_binary
        
        num_non_zeros_sparse = len(sparse_processed_spike_train)
        
        sparse_processed_by_neuron.append(sparse_processed_spike_train)
        num_non_zeros_by_neuron.append(num_non_zeros_sparse)
        
    return sparse_processed_by_neuron, num_non_zeros_by_neuron


def preprocess_one_session(read_path, save_folder, simulation, bin_size=0.001, start_time = 0, end_time = None, n_PAG_to_use='all'):
    '''
    Parameters
    ----------
    read_path : 
        if simulation is True: read_path is a folder
        if not simulation: read_path is a file path
    save_folder :
    simulation : boolean
    bin_size : time bin size in seconds. The default is 0.001.

    Returns
    -------
    None.

    '''
    
    read_path = Path(read_path); save_folder = Path(save_folder)
    
    # load spike times from path
    presyn_df, pag_df, last_spike_time = load_dfs(read_path, simulation, n_PAG_to_use)
    
    if end_time is None: end_time = last_spike_time
    
    # create sparse binned and convolved for presynaptic cells
    presyn_sparse_convolved_by_neuron, presyn_num_non_zeros_by_neuron = create_sparse_preprocessed_spikes( 
        presyn_df['spikeTimes'], 
        start_time=start_time, end_time=end_time, bin_size=bin_size,
        convolve=True, 
        kernel_params={'type': 'exponential', 'tau':15}) # type: ignore
    
    # create sparse binned for pag cells
    pag_sparse_binarised_by_neuron, pag_num_non_zeros_by_neuron = create_sparse_preprocessed_spikes(
        pag_df['spikeTimes'],
        start_time=start_time, end_time=end_time, bin_size=bin_size,
        convolve=False,
        kernel_params={'type': 'exponential', 'tau':15}) # type: ignore

    recording_length = int(np.ceil((end_time - start_time)/bin_size))
    
    # SAVE THEMMMMM
    if simulation:
        session_name = read_path.stem
    else:
        session_name = read_path.stem[:6]
    
    np.save((save_folder / (session_name + '_input_conv_sparse.npy')), np.vstack(presyn_sparse_convolved_by_neuron)) # the sparsified 2D input matrix for this session
    np.save((save_folder / (session_name + '_output_binary_sparse.npy')), np.array(pag_sparse_binarised_by_neuron, dtype=object)) # np object array of each sparsified PAG spike train 
    np.save((save_folder / (session_name +'_preprocessing_metadata.npy')), np.array([np.sum(presyn_num_non_zeros_by_neuron), # total number of non-zero values in input matrix for this session 
                                                                                  pag_num_non_zeros_by_neuron, # number of non-zero values for each PAG spike train for this session 
                                                                                  recording_length, 
                                                                                  len(pag_num_non_zeros_by_neuron)], # number of PAG cells
                                                                                    dtype=object))

def aggregate_sparse_matrices_across_sessions_PAGcells(read_path, output_type='numpy'):
    '''
    Aggregates multiple sparse matrices into a single sparse matrix. Outputs torch.sprase_coo_tensor.
    :params:
    read_path:  path to folder containing multiple sessions of saved sparse matrices
    
    return:
    input_sprase
    output_sprase
    '''
    
    total_non_zero_events_input = [] #ends up as list of number of non-zeros in each session
    total_non_zero_events_output = [] #ends up as list of list of number of non-zeros for each PAG cell, in each session
    recording_lengths = [] #ends up as list of nbins for each session
    
    for session_filepath in [os.path.join(read_path,x) for x in os.listdir(read_path) if '_preprocessing_metadata.npy' in x]:
        session_preprocessing_metadata = np.load(session_filepath, allow_pickle=True)
        total_non_zero_events_input.append(session_preprocessing_metadata[0] * session_preprocessing_metadata[3])
        total_non_zero_events_output.append(session_preprocessing_metadata[1])
        recording_lengths.append(session_preprocessing_metadata[2])
    
    input_sparse = np.zeros((np.sum(total_non_zero_events_input), 3))
    output_sparse = np.zeros((np.sum(total_non_zero_events_output), 3))
    
    # these refer to idxs in the sparse matrix
    session_input_start_idxs = np.concatenate(([0],np.cumsum(total_non_zero_events_input))) # starts of where to insert session_input sparse matrices into input_sparse
    session_pag_start_idxs = np.concatenate(([0],np.cumsum([sum(x) for x in total_non_zero_events_output])))
    
    # these refer to rows and columns in the dense matrix
    last_cell_idx = 0
    last_time_bin_idx = 0
    
    for session_i, pag_session_path in enumerate([os.path.join(read_path,x) for x in os.listdir(read_path) if '_output_binary_sparse.npy' in x]):
        
        input_session_path = pag_session_path.replace('_output_binary_sparse.npy', '_input_conv_sparse.npy')
        
        session_pag_sparse_arrays = np.load(pag_session_path, allow_pickle=True)
        session_input_sparse_matrix = np.load(input_session_path)
        
        pag_starts_current_session =  np.concatenate(([0],np.cumsum(total_non_zero_events_output[session_i])))
        
        for pag_i, pag_sparse_array in enumerate(session_pag_sparse_arrays): # this is a list of sparse matrices
            
            # fill sparse input matrix, such that each session input matrix is placed along the diagonal in the dense version            
            modified_session_input_sparse_matrix = session_input_sparse_matrix.copy() #make a copy because each PAG cell will use the same session_input, dont want to modify it
            modified_session_input_sparse_matrix[:,0] += last_cell_idx
            modified_session_input_sparse_matrix[:,1] += last_time_bin_idx
            
            start_i = session_input_start_idxs[session_i]+pag_i*len(session_input_sparse_matrix)
            end_i = session_input_start_idxs[session_i]+(pag_i+1)*len(session_input_sparse_matrix)
            input_sparse[start_i:end_i] = modified_session_input_sparse_matrix
            
            # fill sparse output array, such that each PAG array is placed along the time dimension in the dense version
            pag_sparse_array[:,1] += last_time_bin_idx
            
            start_i = session_pag_start_idxs[session_i]+pag_starts_current_session[pag_i]
            end_i = session_pag_start_idxs[session_i]+pag_starts_current_session[pag_i+1]
            output_sparse[start_i:end_i, 1:] = pag_sparse_array[:, 1:] #dont take the row value of session_pag, so that the row stays at 0
            
            # move pointers
            last_cell_idx += int(np.max(session_input_sparse_matrix[:,0])+1)
            last_time_bin_idx += recording_lengths[session_i]
            
    vdim = int(np.max(input_sparse[:,0])+1) 
    hdim = int(max(np.max(input_sparse[:,1]), np.max(output_sparse[:,1]))+1)
    
    if output_type == 'numpy':
        return input_sparse, output_sparse
    elif output_type == 'torch':
        import torch
        return torch.sparse_coo_tensor(indices=input_sparse[:,:2].T, 
                                    values=input_sparse[:,2],
                                    dtype=torch.float32,
                                    size=torch.Size([vdim, hdim])
                                    ), \
            torch.sparse_coo_tensor(indices=output_sparse[:,:2].T,
                                    values=output_sparse[:,2],
                                    dtype=torch.float32,
                                    size=torch.Size([1, hdim])
                                    )
    elif output_type == 'scipy':
        from scipy.sparse import coo_matrix
        return coo_matrix((input_sparse[:,2].astype(np.float32), (input_sparse[:,0].astype(np.int32), input_sparse[:,1].astype(np.int32))), shape=(vdim, hdim)), \
            coo_matrix((output_sparse[:,2].astype(np.float32), (output_sparse[:,0].astype(np.int32), output_sparse[:,1].astype(np.int32))), shape=(1, hdim))
    
    else:
        raise ValueError(f'Output type {output_type} not recognised. Must be numpy, torch or scipy.')