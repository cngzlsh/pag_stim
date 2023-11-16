from matplotlib.pylab import logistic
import numpy as np
from loguru import logger
import copy

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


import torch
import torch.nn as nn
import torch.distributions as dist
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class BernoulliGLMPyTorch(nn.Module):
    def __init__(self, n_neurons_per_group, link_fn='logistic', reg_params=0):
        super().__init__()
        self.n_neurons_per_group = n_neurons_per_group
        if not isinstance(reg_params, torch.FloatTensor):
            self.reg_params = reg_params * torch.ones(len(self.n_neurons_per_group))
        else:
            self.reg_params = torch.FloatTensor(reg_params)
        self.neuron_group_cumsum = np.concatenate(([0], np.cumsum(n_neurons_per_group)))
        self.neuron_group_idx = np.concatenate([[i for _ in range(self.n_neurons_per_group[i])] for i in range(len(self.n_neurons_per_group))])
        self.link_fn = link_fn
        self.linear = nn.Linear(in_features=np.sum(n_neurons_per_group), out_features=1)
        if self.link_fn == 'logistic':
            self.activation = nn.Sigmoid()
        assert np.sum(self.n_neurons_per_group) == np.prod(self.linear.weight.data.shape)
    
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
    
    def calc_group_statistics(self):
        '''
        Computes the mean and variance of weights within each neuron group.
        '''
        group_means = torch.zeros(len(self.n_neurons_per_group))
        group_stds = torch.zeros(len(self.n_neurons_per_group))
        
        for m, n in enumerate(self.neuron_group_cumsum[:-1]):
            group_means[m] = torch.mean(self.linear.weight.data[0, self.neuron_group_cumsum[m]: self.neuron_group_cumsum[m+1]])
            group_stds[m] = torch.std(self.linear.weight.data[0, self.neuron_group_cumsum[m]: self.neuron_group_cumsum[m+1]])
        
        return group_means, group_stds

    def calc_log_likelihood_w_reg(self, X, y):
        if isinstance(y, np.ndarray) or isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(device)
            y = torch.FloatTensor(y).to(device)
            
        reg_term = 0
        # for i, w in enumerate(self.linear.weight[0,:]):
        #     m = self.neuron_group_idx[i]
        #     reg_term += self.reg_params[m] * torch.abs(w) * torch.sum(
        #         torch.pow(
        #             self.linear.weight[0, self.neuron_group_cumsum[m]:self.neuron_group_cumsum[m+1]] - w, 2)
        #         )
        # (self.weights @ X + self.bias) @ y.T  - np.sum(np.log(1 + np.exp(self.weights @ X + self.bias)))
        for m in range(len(self.n_neurons_per_group)):
            w = self.linear.weight[:, self.neuron_group_cumsum[m]:self.neuron_group_cumsum[m+1]]
            reg_term += self.reg_params[m] * torch.sum(
                torch.pow(w, 2) \
                @ torch.pow((w - w.T), 2)
                )
        
        return - (self.linear(X).T @ y - torch.sum(torch.log(1 + torch.exp(self.linear(X))))) + reg_term
    
    def fit(self, X, y, n_iter, lr=1e-3, verbose=1, decay=None):
        '''
        verbose:    -1: prints nothing; 0: prints initial and final losses; 1: prints 20 steps; 2: prints all steps.
        '''
        if isinstance(y, np.ndarray) or isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(device)
            y = torch.FloatTensor(y).to(device)
        
        if verbose > -1:
            with torch.no_grad():
                logger.debug(f'Training GLM with PyTorch. Initial log like: {self.calc_log_likelihood(X, y)}, inital loss: {self.calc_log_likelihood_w_reg(X, y).cpu().float()}')
            
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if decay is not None:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)
        self.best_loss = torch.tensor(np.inf)
        
        for epoch_i in range(n_iter):
            
            loss = self.calc_log_likelihood_w_reg(X, y)
            if loss.detach().cpu().float() < self.best_loss:
                best_log_like = self.calc_log_likelihood(X, y)
                best_epoch_i = epoch_i + 1
                self.best_weight = copy.copy(self.linear.weight.data)
                self.best_bias = copy.copy(self.linear.bias.data)
                self.best_loss = loss.detach().cpu().float()
            
            if verbose == 2:
                logger.debug(f'Step {epoch_i+1}. Log like: {self.calc_log_likelihood(X, y).cpu().float()},  loss: {float(loss.detach().cpu().numpy())}')
            elif verbose == 1:
                if (epoch_i+1) % int(n_iter / 20) == 0:
                    logger.debug(f'Step {epoch_i+1}. Log like: {self.calc_log_likelihood(X, y).cpu().float()},  loss: {float(loss.detach().cpu().numpy())}')
            
            loss.backward()
            optimizer.step()
            if decay is not None:
                scheduler.step()
            optimizer.zero_grad()
        
        if verbose > -1:
            with torch.no_grad():
                logger.debug(f'Training complete with best log like: {best_log_like}, best loss: {self.best_loss} at epoch {best_epoch_i}.')
    
    def load_best_params(self):
        try:
            self.linear.weight.data = self.best_weight
            self.linear.bias.data = self.best_bias
        except:
            raise AttributeError('Fit model to data first')
        
if __name__ == '__main__':

    seed = 0
    np.random.seed(seed)
    # fake data
    T = 1000
    N = 100

    X = np.random.binomial(n=1, p=.7, size=(N, T))

    true_weights = np.random.normal(loc=0, scale=1, size=(1, N))
    true_bias = np.random.normal()

    y = np.random.binomial(1, p=1 / (1 + np.exp(- true_weights @ X + true_bias)))

    # glm = BernoulliGLMwReg(n_neurons_per_group=np.array([N]), reg_params=np.array([0.01]))
    # glm.random_init_params(X)
    # glm.fit(X, y, n_iter=200, lr=8*1e-4, decay=0.99)
    # print(glm.best_cost, glm.best_log_likelihood)

    # glm.weights = true_weights
    # glm.bias = true_bias
    # print(glm.calc_log_likelihood(X,y))
    
    glm = BernoulliGLMPyTorch(n_neurons_per_group=np.array([N])).to(device)
    glm.fit(X.T, y.T, n_iter=1000, lr=1e-3, verbose=1)
    
    with torch.no_grad():
        glm.linear.weight.data = torch.FloatTensor(true_weights).to(device)
        glm.linear.bias.data = torch.FloatTensor([true_bias]).to(device)
        print(glm.calc_log_likelihood(X.T, y.T))
        
    