from matplotlib.pylab import logistic
import numpy as np
from loguru import logger
import copy
from utils import BNN_Dataset

# the following are numpy implementation of GLMs.

class BernoulliGLM():
    def __init__(self, n_neurons_per_group, link_fn='logistic', init_strategy='gaussian', seed=0):
        self.init_strategy = init_strategy
        self.link_fn = link_fn
        self.seed = seed
        self.n_neurons_per_group = n_neurons_per_group
        self.neuron_group_cumsum = np.concatenate(([0], np.cumsum(n_neurons_per_group)))
        self.neuron_group_idx = np.concatenate([[i for _ in range(self.n_neurons_per_group[i])] for i in range(len(self.n_neurons_per_group))])
        self.weights = np.zeros((1, np.sum(self.n_neurons_per_group)))
        self.bias = np.zeros(1)

    
    def random_init_params(self, X, init_strategy='gaussian', seed=1):
        assert X.shape[0] == np.sum(self.n_neurons_per_group)

        if init_strategy == 'gaussian':
            np.random.seed(seed)
            self.weights = np.random.normal(loc=0, scale=1, size=(1, X.shape[0]))
            self.bias = np.random.normal(0, 1)
            logger.debug('GLM parameters initialised with standard Gaussian distribution.')

        if init_strategy == 'fisher':
            raise NotImplementedError()
        
    
    def predict(self, X):
        '''
        Given an array of flattened neuronal activities, predict for a single 
        :params:
        X:      a flattened array of binary variables, corresponding to each neuron in the input
        '''
        if self.link_fn == 'logistic':
            temp = self.weights @ X + self.bias
            return 1 / (1 + np.exp(-temp))
    
    def calc_log_likelihood(self, X, y):
        '''
        Log likelihood for Bernoulli GLM.
        params:
        y:              binary response, shape (1, T)
        X:              binary inputs, shape(N, T)

        :note:
        self.weights    weights for each input neuron, shape(1, N)
        '''
        return (self.weights @ X + self.bias) @ y.T  - np.sum(np.log(1 + np.exp(self.weights @ X + self.bias)))
    
    def calc_group_statistics(self):
        '''
        Computes the mean and variance of weights within each neuron group.
        '''
        group_means = np.zeros(len(self.n_neurons_per_group))
        group_stds = np.zeros(len(self.n_neurons_per_group))
        
        for m, n in enumerate(self.neuron_group_cumsum[:-1]):
            group_means[m] = np.mean(self.weights[self.neuron_group_cumsum[m]: self.neuron_group_cumsum[m+1]])
            group_stds[m] = np.std(self.weights[self.neuron_group_cumsum[m]: self.neuron_group_cumsum[m+1]])
        
        return group_means, group_stds

    def calc_grad(self, X, y):
        raise NotImplementedError()
    
    def calc_hessian(self, X, y):
        raise NotImplementedError()

    def fit(self, X, y):
        raise NotImplementedError()

class BernoulliGLMwReg(BernoulliGLM):
    def __init__(self, n_neurons_per_group, reg_params, link_fn='logistic', init_strategy='gaussian', seed=0, reg='min_weights_within_group'):
        super().__init__(n_neurons_per_group=n_neurons_per_group, link_fn=link_fn, init_strategy=init_strategy, seed=seed)
        self.reg = reg
        self.reg_params = reg_params
        assert len(self.reg_params) == len(self.n_neurons_per_group)
    
    def calc_cost(self, X, y):
        reg_term = 0
        for n, w_n in enumerate(self.weights[0]):
            m = self.neuron_group_idx[n]
            nrn_grp_start = self.neuron_group_cumsum[m]
            nrn_grp_end = self.neuron_group_cumsum[m+1]
            
            reg_term += self.reg_params[m] * np.sum([w_n ** 2 * (w_n - w_k) ** 2 for w_k in self.weights[0, nrn_grp_start:nrn_grp_end]])

        return - self.calc_log_likelihood(X, y) + reg_term

    def calc_grad(self, X, y):
        '''
        :params:
        y:              binary response, shape (1, T)
        X:              binary inputs, shape(N, T)
        :note:
        self.weights    weights for each input neuron, shape(1, N)
        '''
        dw = np.zeros_like(self.weights)
        db = np.zeros_like(self.bias)

        if self.reg == 'min_weights_within_group':
            for n, w_n in enumerate(self.weights[0]):
                m = self.neuron_group_idx[n]
                nrn_grp_start = self.neuron_group_cumsum[m]
                nrn_grp_end = self.neuron_group_cumsum[m+1]

                dw[0, n] = - y @ X[n,:] \
                    + np.sum(X[n,:] / (1 + np.exp(-(self.weights @ X + self.bias)))) \
                + self.reg_params[m] * w_n * (self.n_neurons_per_group[m] * w_n ** 2 + np.sum([(w_n - w_k) ** 2 \
                                            - w_n * w_k for w_k in self.weights[0, nrn_grp_start:nrn_grp_end]]))

            db = - np.sum(y + 1/(1 + self.weights @ X + self.bias))

        assert dw.shape == self.weights.shape
        return dw, db

    def calc_hessian(self, X, y):
        '''
        Computes the Hessian matrix (second-order derivatives) of the weights.
        '''
        raise NotImplementedError('I think the memory requirement is too big for this, also I cba to compute the second derivative by hand')

    def fit(self, X, y, mode='sgd', n_iter=100, lr=0.01, decay=None, threshold=1e-2, verbose=1):
        '''
        Estimate parameters of the GLM.

        :params:
        y:              binary response, shape (1, T)
        X:              binary inputs, shape(N, T)
        mode:           'sgd' if using first-order gradient descent, 'newton' if using second-order
        n_iter:         number of optimisation iterations
        lr:             learning rate
        decay:          multiplier for exponential decay of learning rate
        verbose:        '0': print log only before and after training; '1': print 10 logs; '2': print every step
        '''
        self.best_cost = np.inf
        logger.debug(f'Initial log-likelihood: {self.calc_log_likelihood(X, y).flatten()} ; initial loss: {self.calc_cost(X,y).flatten()}')

        if mode == 'sgd':
            logger.debug(f'Fitting parameters with gradient descent.')

            for i in range(n_iter):
                
                dw, db = self.calc_grad(X, y)
                self.weights -= lr * dw
                self.bias -= lr * db
                new_cost = self.calc_cost(X,y)
                
                if new_cost < self.best_cost:
                    self.best_weights = self.weights
                    self.best_bias = self.bias
                    self.best_cost = new_cost
                    self.best_log_likelihood = self.calc_log_likelihood(X, y)
                
                if decay is not None:
                    lr *= decay

                if verbose == 2:
                    logger.debug(f'Step {i+1}. Log likelihood: {self.calc_log_likelihood(X, y).flatten()}; loss: {self.calc_cost(X,y).flatten()}')
                elif verbose == 1:
                    if (i+1) % int(n_iter / 10) == 0:
                        logger.debug(f'Step {i+1}. Log likelihood: {self.calc_log_likelihood(X, y).flatten()}; loss: {self.calc_cost(X,y).flatten()}')

        elif mode == 'newton':
            i=0
            raise NotImplementedError('I think the memory requirement is too big for this, also I cba to compute the second derivative by hand')
        
        logger.debug(f'Training complete at {i+1} steps. Log likelihood: {self.calc_log_likelihood(X, y).flatten()}; loss: {self.calc_cost(X,y).flatten()}')
        group_means, group_stds = self.calc_group_statistics()
        logger.debug(f'Mean weights by neuron groups: {list(group_means)}; std: {list(group_stds)}.')

# the following implements glm.fit() function with gradient descent via deep learning library torch.
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.data import DataLoader
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class BernoulliGLMPyTorch(nn.Module):
    def __init__(self,
                 group_names,
                 n_neurons_per_group=None,
                 synapse_origin_group=None,
                 link_fn='logistic',
                 n_sessions=1,
                 regs=['weights_within_group'],
                 reg_params=0):
        super().__init__()
        
        '''
        must specify: EITHER n_neurons_per_group (assuming same input neurons in all session/PAG cell)
                      OR synapse_origin_group (allowing different input neurons in each session/PAG cell)
        
        :param n_neurons_per_group:     array of n_groups, each element is the number of neurons in the group indexed by that element.
        :param synapse_origin_group:    array of n_synapses, each element is the idex of input brain region.
        :param regs:                    lists of regularisations to use, must be in 'weights_ingroup', 'weights_sparsity', 'beta'
        '''
        self.group_names = group_names
        self.n_sessions = n_sessions                            # total number of sessions/PAG cells to repeat
        
        if n_neurons_per_group is not None:
            self.mode = 'neurons'
            # assumes each neuron is connected to all PAG cells, broadcast each neuron by n_sessions
            self.n_neurons_per_group = n_neurons_per_group      # total number of actual neurons per brain region
            self.n_neurons = np.sum(self.n_neurons_per_group)   # total number of actual neurons
            self.n_groups = len(self.n_neurons_per_group)       # total number of brain regions
            self.n_synapses = self.n_sessions * self.n_neurons
            
            neuron_group_cumsum = np.concatenate(([0], np.cumsum(self.n_neurons_per_group))) # cumulative sum of neurons of n_neurons_per_group
            # the group index of each synapse, len(n_sessions * n_neurons), originally self.neuron_group_idx
            self.synapse_origin_group = np.concatenate([np.concatenate([[i for _ in range(self.n_neurons_per_group[i])] for i in range(len(self.n_neurons_per_group))]) for _ in range(n_sessions)]) 
            # the synapse indices of each group, list of n_group sublists, each sublist contains n_session * n_group_neuron indices, originally group_neuron_idx
            self.group_synapse_idx = [np.concatenate([np.arange(neuron_group_cumsum[m], neuron_group_cumsum[m+1]) + self.n_neurons * i for i in range(n_sessions)]) for m in range(self.n_groups)]# the neuron index of each group
            
            self.neuron_group_idx = self.synapse_origin_group # deprecated use
            self.group_neuron_idx = self.group_synapse_idx # deprecated use
            
        else: 
            assert synapse_origin_group is not None
            self.mode = 'synapses'
            self.synapse_origin_group = synapse_origin_group   # group idx of each synapse source
            
            self.n_groups = len(np.unique(synapse_origin_group))
            self.n_synapses = len(synapse_origin_group)
            # the synapse indices of each group, list of n_group sublists, each sublist contains n_session * n_group_neuron indices
            self.group_synapse_idx = [list(np.where(synapse_origin_group == m)[0]) for m in range(self.n_groups)]
            self_n_synpases_per_group = [len(idxs) for idxs in self.group_synapse_idx]
        
        # regularisers
        self.regs = regs
        self.reg_params = {}
        # the following conditional statements are just to make previous versions compatible
        if isinstance(reg_params, int) or isinstance(reg_params, float):
            self.reg_params['weights_within_group'] = reg_params * torch.ones(self.n_groups)
        elif 'weights_within_group' in reg_params.keys():
            if isinstance(reg_params['weights_within_group'], int) or isinstance(reg_params['weights_within_group'], float):
                self.reg_params['weights_within_group'] = reg_params['weights_within_group'] * torch.ones(self.n_groups)
            reg_params.pop('weights_within_group')
        self.reg_params.update(reg_params)
            
        
        self.accepted_regs = ['weights_within_group', 'weights_sparsity', 'beta']
        for reg in self.regs:
            if reg not in self.accepted_regs:
                raise ValueError(f'specify one or more regularisation schemes from {self.accepted_regs}: {reg} is not recognised')
            if reg not in list(self.reg_params.keys()):
                raise ValueError(f'specify hyperparameters for {reg}')
            
        
        self.link_fn = link_fn
        self.linear = nn.Linear(in_features=self.n_synapses, out_features=1)
        if self.link_fn == 'logistic':
            self.activation = nn.Sigmoid()
        # assert np.sum([len(g) for g in self.group_neuron_idx]) == np.prod(self.linear.weight.data.shape)
    
    def forward(self, X):
        '''
        Makes a forward prediction
        '''
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(device)
        return self.activation(self.linear(X))
    
    def calc_log_likelihood(self, X, y):
        if isinstance(y, np.ndarray) or isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(device)
            y = torch.FloatTensor(y).to(device)
        with torch.no_grad():
            return torch.sum(dist.Bernoulli(probs=self.forward(X)).log_prob(y))

    def calc_log_likelihood_w_reg(self, X, y, return_components=False):
        if isinstance(y, np.ndarray) or isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(device)
            y = torch.FloatTensor(y).to(device)
        
        reg_terms = torch.zeros(3).to(device) # weights_within_group, weights_sprasity, beta
        
        if 'weights_within_group' in self.regs:
        # for i, w in enumerate(self.linear.weight[0,:]):
        #     m = self.neuron_group_idx[i]
        #     reg_term += self.reg_params[m] * torch.abs(w) * torch.sum(
        #         torch.pow(
        #             self.linear.weight[0, self.neuron_group_cumsum[m]:self.neuron_group_cumsum[m+1]] - w, 2)
        #         )
        # (self.weights @ X + self.bias) @ y.T  - np.sum(np.log(1 + np.exp(self.weights @ X + self.bias)))
            for m in range(self.n_groups):
                w = self.linear.weight[:, self.group_synapse_idx[m]]
                reg_terms[0] += self.reg_params['weights_within_group'][m] * torch.sum(
                    torch.pow(w, 2) \
                    @ torch.pow((w - w.T), 2)
                    )
        
        if 'weights_sparsity' in self.regs:
            reg_terms[1] = self.reg_params['weights_sparsity'] * torch.sum(torch.abs(self.linear.weight))
            
        if 'beta' in self.regs:
            b = self.linear.bias
            reg_terms[2] = self.reg_params['beta'][0] * b ** self.reg_params['beta'][2] + self.reg_params['beta'][1] * b ** - self.reg_params['beta'][3]

        loss_with_grad = -torch.sum(dist.Bernoulli(probs=self.forward(X)).log_prob(y)) + torch.sum(reg_terms)
        
        if return_components:
            return - torch.sum(dist.Bernoulli(probs=self.forward(X)).log_prob(y)), reg_terms
        else:
            return loss_with_grad
    
    def fit(self, X, y, n_iter, lr=1e-3, verbose=1, decay=1, batch_size=-1):
        '''
        verbose:        -1: prints nothing; 0: prints initial and final losses; 1: prints 20 steps; 2: prints all steps.
        decay:          exponential decay of learning rate.
        batch_size:     defaults to -1, using all data at once; 'auto': using 1/10 data at once; a number: specified batch size
        '''
        if isinstance(y, np.ndarray) or isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(device)
            y = torch.FloatTensor(y).to(device)
        
        if batch_size == -1:
            batch_size = int(np.prod(y.shape))
        elif batch_size == 'auto':
            batch_size = int(np.prod(y.shape)/10)
        else:
            assert isinstance(batch_size, int)
            assert batch_size > 0
        
        if verbose > -1:
            with torch.no_grad(): 
                init_log_like, init_regs = self.calc_log_likelihood_w_reg(X, y, return_components=True)
                logger.debug(f'Training GLM with PyTorch. Initial log like: {init_log_like.cpu().float()}, loss {init_log_like.cpu().float()+np.sum(list(init_regs.cpu().numpy()))}, of which regs {list(init_regs.cpu().numpy())} respectively for {self.accepted_regs}.')
            
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)
        # dataloader = DataLoader(BNN_Dataset(X, y), batch_size=batch_size, drop_last=False, shuffle=False)
            
        self.best_loss = torch.tensor(np.inf)
        best_log_like = self.calc_log_likelihood(X, y)
        best_epoch_i = 1
        
        for epoch_i in range(n_iter):
            
            optimizer.zero_grad()
            # loss = 0
            loss = self.calc_log_likelihood_w_reg(X, y)
            
            # for i, (batch_x, batch_y) in enumerate(iter(dataloader)):
            #     optimizer.zero_grad()
                
            #     batch_loss = self.calc_log_likelihood_w_reg(batch_x, batch_y)
                
            #     batch_loss.backward()
            #     optimizer.step()
                
            #     loss += batch_loss.item()
            
            if loss < self.best_loss:
                with torch.no_grad():
                    best_log_like, best_regs = self.calc_log_likelihood_w_reg(X, y, return_components=True)
                best_epoch_i = epoch_i + 1
                self.best_weight = copy.copy(self.linear.weight.data)
                self.best_bias = copy.copy(self.linear.bias.data)
                self.best_loss = loss
                self.best_regs = best_regs
            
            if verbose == 2:
                logger.debug(f'Step {epoch_i+1}. Log like: {self.calc_log_likelihood(X, y).cpu().float()},  loss: {loss}')
            elif verbose == 1:
                if (epoch_i+1) % int(n_iter / 20) == 0:
                    with torch.no_grad():
                        epoch_log_like, epoch_regs= self.calc_log_likelihood_w_reg(X, y, return_components=True)
                    logger.debug(f'Step {epoch_i+1}. Log like: {epoch_log_like.cpu().float()}, loss {epoch_log_like.cpu().float()+np.sum(list(epoch_regs.cpu().numpy()))}, of which regs {list(epoch_regs.cpu().numpy())} respectively for {self.accepted_regs}.')
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        if verbose > -1:
           
                logger.debug(f'Training complete with best log like: {best_log_like.cpu().float()}, best loss: {self.best_loss}, of which regs {list(best_regs.cpu().numpy())} respectively for {self.accepted_regs} at epoch {best_epoch_i}.')
    
    def load_best_params(self):
        try:
            self.linear.weight.data = self.best_weight.to(device)
            self.linear.bias.data = self.best_bias.to(device)
        except:
            raise AttributeError('Fit model to data first')
    
    def _load_state_dict(self, pth_file):
        # loads state dict, then saves as best weight
        self.load_state_dict(pth_file)
        
        self.best_weight = copy.copy(self.linear.weight.data)
        self.best_bias = copy.copy(self.linear.bias.data)


# class BernoulliGLMBetaRegPyTorch(BernoulliGLMPyTorch):
#     '''
#     Added bias regularisation - Yulin & Dario 11 Dec 2023
#     '''
#     def __init__(self,
#                  group_names,
#                  n_neurons_per_group=None,
#                  synapse_origin_group=None,
#                  link_fn='logistic',
#                  n_sessions=1,
#                  weights_reg_params=0,
#                  beta_reg_params=np.array([1, 1, 3, 2])):
        
#         super().__init__(group_names,
#                          n_neurons_per_group=n_neurons_per_group,
#                          synapse_origin_group=synapse_origin_group,
#                          link_fn=link_fn,
#                          n_sessions=n_sessions,
#                          reg_params=weights_reg_params)
        
#         assert len(beta_reg_params) == 4
#         self.beta_reg_params = beta_reg_params # np.array([m, n, j, k])
#         # beta regularisation: m * beta**j + n * beta ** (-k)
#         assert beta_reg_params[2] % 2 == 1
#         assert beta_reg_params[3] % 2 == 0

#     def calc_log_likelihood_w_reg(self, X, y, return_components=False):
#         if isinstance(y, np.ndarray) or isinstance(X, np.ndarray):
#             X = torch.FloatTensor(X).to(device)
#             y = torch.FloatTensor(y).to(device)
            
#         weights_reg_term = 0
#         for m in range(self.n_groups):
#             w = self.linear.weight[:, self.group_synapse_idx[m]]
#             weights_reg_term += self.reg_params[m] * torch.sum(
#                 torch.pow(w, 2) \
#                 @ torch.pow((w - w.T), 2)
#                 )
        
#         b = self.linear.bias.data
#         bias_reg_term = self.beta_reg_params[0] * b ** self.beta_reg_params[2] + self.beta_reg_params[1] * b ** -self.beta_reg_params[3]
#         if return_components:
#             return - torch.sum(dist.Bernoulli(probs=self.forward(X)).log_prob(y)), weights_reg_term, bias_reg_term        
#         else:
#             return - torch.sum(dist.Bernoulli(probs=self.forward(X)).log_prob(y)) + weights_reg_term + bias_reg_term


class BernoulliGLMwHistoryPyTorch(BernoulliGLMPyTorch):
    def __init__(self,
                 group_names,
                 n_neurons_per_group=None,
                 synapse_origin_group=None,
                 link_fn='logistic',
                 n_sessions=1,
                 regs=['weights_within_group'],
                 reg_params=0,
                 history=1):
        
        super().__init__(group_names,
                         n_neurons_per_group=n_neurons_per_group,
                         synapse_origin_group=synapse_origin_group,
                         link_fn=link_fn,
                         n_sessions=n_sessions,
                         regs=regs,
                         reg_params=reg_params)
        
        assert isinstance(history, int) and history >= 1
        self.history = history
        self.history_filters = nn.Linear(self.history, 1, bias=False)
    
    def forward(self, X):
        '''
        Makes a forward prediction
        '''
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(device) # (bins, input_shape)
        
        with torch.no_grad():
            y_hat_hist = torch.vstack((torch.zeros(self.history, 1).to(device), self.activation(self.linear(X))))

        return self.activation(
            self.linear(X) + self.history_filters(
                torch.hstack([dist.Bernoulli(probs=y_hat_hist[h:h+X.shape[0]]).sample() for h in range(self.history)])
            )
        )

class BernoulliGLMwHistoryMultiSessionPyTorch(BernoulliGLMPyTorch):
    def __init__(self,
                 group_names,
                 n_neurons_per_group=None,
                 synapse_origin_group=None,
                 link_fn='logistic',
                 n_sessions=1,
                 regs=['weights_within_group'],
                 reg_params=0,
                 history=1):
        
        super().__init__(group_names,
                         n_neurons_per_group=n_neurons_per_group,
                         synapse_origin_group=synapse_origin_group,
                         link_fn=link_fn,
                         n_sessions=n_sessions,
                         regs=regs,
                         reg_params=reg_params)
        
        assert isinstance(history, int) and history >= 1
        self.history = history
        # self.history_filters = nn.ModuleList([nn.Linear(self.history, 1, bias=False) for _ in range(self.n_sessions)])
        self.history_filters = nn.ModuleList(nn.Linear(self.history, 1, bias=False) for _ in range(self.n_sessions))
        # assuming history is stacked of shape (n_bins, n_PAG_cells * n_steps)
    
    def forward(self, X):
        '''
        Makes a forward prediction
        '''
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(device) # (n_bins * n_sessions, input_shape)
        
        with torch.no_grad():
            y_hat_naive = self.activation(self.linear(X)) # (n_bins * n_sessions, 1)
            y_hat_naive = y_hat_naive.reshape(-1, self.n_sessions) # (n_bins, n_sessions)
            n_bins = y_hat_naive.shape[0]
        
        # y_hat_hist = torch.stack([torch.vstack([torch.zeros((self.history, 1)).to(device), y_hat_naive[c]]) for c in range(self.n_sessions)])
        y_hat_hist_sampled = torch.vstack(
            [torch.zeros(self.history, self.n_sessions).to(device),
             y_hat_naive])
             # dist.Bernoulli(probs=y_hat_naive).sample()]) # (n_bins + history, n_sessions) stacked like
        del y_hat_naive
        
        y_hat_hist_sampled = torch.hstack([
            y_hat_hist_sampled[h:h+n_bins, :] for h in range(self.history) 
        ])
        assert y_hat_hist_sampled.shape == torch.Size([n_bins, self.history * self.n_sessions]) # (n_bins, n_PAG_cells * n_steps)
        # re-arrange the tensor such that it's [:, [0,2,4,1,3,5]]
        
        hist_terms = torch.vstack([self.history_filters[c](y_hat_hist_sampled[:, np.arange(c, self.n_sessions*self.history, self.n_sessions)]) for c in range(self.n_sessions)])
        del y_hat_hist_sampled
        
        return self.activation(self.linear(X) + hist_terms)
    
    def get_history_filter_weights(self):
        with torch.no_grad():
            return torch.vstack([torch.flip(self.history_filters[c].weight.data, [1]) for c in range(self.n_sessions)]) # (n_pag_cells, history)ß
        

if __name__ == '__main__':

    seed = 0
    np.random.seed(seed)
    # fake data
    T = 11  # total time steps
    N = 5   # number of input neurons
    n = 2   # number of output neurons

    X = np.random.binomial(n=1, p=.7, size=(N, T))
    true_weights = np.random.normal(loc=0, scale=1, size=(n, N))
    true_bias = np.random.normal()
    y = np.random.binomial(1, p=1 / (1 + np.exp(- true_weights @ X + true_bias)))
    
    # construct multi-PAG glm input and target matrix
    X_train = np.zeros((N * n, T * n))
    y_train = y.reshape(1, -1)
    for c in range(n):
        X_train[c*N:(c+1)*N, c*T:(c+1)*T] = X
        
    print('fake data shapes: ',X_train.shape, X_train.shape)
    
    glm = BernoulliGLMwHistoryMultiSessionPyTorch(group_names=['a', 'b', 'c', 'd','e'],
                                                  n_neurons_per_group=[1,1,1,1,1],
                                                  n_sessions=2,
                                                  regs=['weights_within_group', 'weights_sparsity'],
                                                  reg_params={
                                                      'weights_within_group':1,
                                                      'weights_sparsity': 1,
                                                  },
                                                  history=3)
    glm.fit(X_train.T, y_train.T, n_iter= 100)
    glm.get_history_filter_weights()
        
    