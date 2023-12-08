import numpy as np
from loguru import logger
import brian2 as b2
import sys
from tpd import recorder as _recorder
import utils
from neurons import PAG
import utils

import matplotlib.pyplot as plt
import numpy as np

from fcutils.path import from_yaml

sys.path.append("./")
# from brian2_rnn.drnn import utils, visualize

class Experiment:
    def __init__(
        self,
        model_class,
        model_params_path,
        exp_params_path,
        name="base experiment class",
    ):
        """
            Base Experiment class
        """
        self.network = None
        self.name = name
        self.params = from_yaml(exp_params_path)

        # set random numbers generator
        self.set_random_seeds()

        # initiliaze network model & update params
        self.model = model_class(model_params_path)
        self.params = {**self.params, **self.model.params}

        # prepare experiment
        self.prepare_inputs()
    
    def set_random_seeds(self):
        if self.params["random_seed"] is not None:
            logger.debug(
                f'Setting random seed to: {self.params["random_seed"]}'
            )
            np.random.seed(self.params["random_seed"])
            b2.seed(self.params["random_seed"])
    
    def prepare_inputs(self):
        """
            Placeholders for functions of individual experiments
        """
        logger.debug(
            f'The experiment "{self.name}" does not have  prepare_inputs method'
        )

    def stimuli_func(self, t):
        raise NotImplementedError("Need to implement a stimuli function")

    def generate_network(self):
        # define network operations
        @b2.network_operation(dt=self.params["stimulus_dt"] * b2.ms)
        def stimulate_network(t):
            self.stimuli_func(t)

        # create network
        self.network = b2.Network(stimulate_network, *self.model.b2_objects,)

    def simulate(self,):
        """
            runs the simulation
        """
        logger.debug(f"Experiment params:\n{self.params}")
        logger.debug(
            "=" * 60
            + "\n"
            + " " * 15
            + f'Running simulation for {self.params["duration"]} s\n'
            + "=" * 60
            + f'\nExperiment name: "{self.name}"\nDescription: {self.description}\n'
            + "=" * 60
        )

        if self.network is None:
            self.generate_network()

        self.network.run(
            self.params["duration"] * 1000 * b2.ms,
            report=utils.simulation_report,
            report_period=5 * b2.second,
        )
        logger.info("simulation completed")
    
    def make_plots(self, list_of_plots = ['rasters', 'weight_matrices', 'single_PAG_neuron_V_trace']):
        # for plotting exptal traces, string should be in format: 'pre2post_inputs'

        # summary plots
        if 'rasters' in list_of_plots: utils.plot_rasters(self.model.neuron_populations, mode='all', path='./figs/'+self.name+'/')
        if 'voltages' in list_of_plots: utils.plot_voltages(self.model.neuron_populations, self.params)
        if 'single_PAG_neuron_V_trace' in list_of_plots: utils.plot_single_neuron_V_trace(self.model.PAG, path='./figs/'+self.name+'/')
        if 'weight_matrices' in list_of_plots: utils.plot_weight_matrices(self.model.neuron_populations, path='./figs/'+self.name+'/')
        input_fitting_list = [x for x in list_of_plots if 'inputs' in x]
        
        
        try:
            if self.params['synapse_to_fit'] != '': input_fitting_list = [self.params['synapse_to_fit'] + '_inputs' ]
        except:
            pass
        

        for input in input_fitting_list:
            pre = input.split('_')[0].split('2')[0]
            post = input.split('_')[0].split('2')[1]
            post_pop = [pop for pop in self.model.neuron_populations if pop.neurons.name.lower() == post]
            if len(post_pop)==1: utils.plot_single_pop(post_pop[0], self.params, plot_expt_inputs_from = pre)
    
    def save(self, recorder=None):
        """
            Save data and results
        """
        recorder = recorder if recorder is not None else _recorder

        # save spike times of each population of neurons
        for pop in self.model.neuron_populations:
            logger.debug(f"Saving spiking data for: {pop.name}")

            # get when each neuron spikes
            spikes_record = {}
            for neuron in range(pop.n):
                times = np.where(pop.spikes.i == neuron)[0]
                spikes_record[neuron] = list(pop.spikes.t[times] / b2.ms)

            # save in three formats so that Dario is happy, because he uses matlab
            recorder.add_data(
                spikes_record,
                f"{pop.name}_spike_times",
                fmt="json",
                description="Spike times (in ms) for each neuron (by index)",
            )

            recorder.add_data(
                spikes_record, f"{pop.name}_spike_times", fmt="csv",
            )

            recorder.add_data(
                {k: np.array(v) for k, v in spikes_record.items()},
                f"{pop.name}_spike_times",
                fmt="mat",
            )

            # save spikes connectivity matrix
            if pop.synapses is not None:
                weights = utils.get_synapses_connectivity_matrix(pop.synapses)

                recorder.add_data(
                    weights, f"{pop.name}_connectivity_matrix", fmt="npy",
                )

                recorder.add_data(
                    weights, f"{pop.name}_connectivity_matrix", fmt="mat",
                )

        # save each population of synapses
        for syn in self.model.synapses:
            # save spikes connectivity matrix
            weights = utils.get_synapses_connectivity_matrix(syn)

            recorder.add_data(
                weights, f"connectivity_{syn.name}", fmt="npy",
            )

            recorder.add_data(
                weights, f"connectivity_{syn.name}", fmt="mat",
            )


class SimulationExperiment(Experiment):
    STIM_STATUS = 'off'

    name = 'feed forward simulation experiment'
    description = 'simulating feed forward models into the PAG from ACC, VMH, IC, SC, PMd'

    def __init__(
            self, model_class, model_params_path, exp_params_path, name=None
    ):
        name = name or self.name
        super().__init__(
            model_class, model_params_path, exp_params_path, name=None
        )

        self.evoked_poisson_spiketrain = []
        for i, population in enumerate(self.model.neuron_populations):
            if population is None or isinstance(population, PAG):
                self.evoked_poisson_spiketrain.append(None)
            else:
                self.evoked_poisson_spiketrain.append(self._pregenerate_poisson_stimulation_times(population))
        
        self.stim_bin_i = 0

    def prepare_inputs(self):
        if "stimulus_indices" in self.params.keys():
            logger.debug(
                "Setting stimulus indices based on given parameters indices"
            )
            self.stim_target_idx = self.params["stimulus_indices"]
        else:
            # compute the simulus index
            self.stim_target_idx = []
            for population in self.model.neuron_populations:
                if population is None or isinstance(population, PAG):
                    self.stim_target_idx.append(None)
                else:
                    self.stim_target_idx.append(np.arange(population.n))

            logger.debug(
                f'STIMULUS start time: {self.params["stim_start_t"]} s - end time: {self.params["stim_end_t"]} s'
            )

    def stimuli_func(self, t):
        time = t.variable.get_value()[0] * b2.second
        t_start = self.params["stim_start_t"] * 1000 * b2.ms
        t_end = self.params["stim_end_t"] * 1000 * b2.ms
        stimulus_strength = self.params["stim_strength"] * b2.namp
        # train_strength = self.params["train_strength"] * b2.namp # no train spikes

        # start/stop stimulus spont
        if t >= t_start and t < t_end and self.params["use_pulse_spont"]: # no use_pulse here
            if self.STIM_STATUS == "off":
                logger.debug(f" === Stimulus ON at time: {time}")
                self.STIM_STATUS = "on"
            
            for i, population in enumerate(self.model.neuron_populations):
                if population is not None and not isinstance(population, PAG):
                    population.neurons.I_stim[
                        self.stim_target_idx[i]
                    ] = stimulus_strength * self.evoked_poisson_spiketrain[i][self.stim_target_idx[i], self.stim_bin_i]
            self.stim_bin_i +=1
    
    def _pregenerate_poisson_stimulation_times(self, pop):

        def homogeneous_poisson(rate, tmax, bin_size):
            ## https://github.com/btel/python-in-neuroscience-tutorials/blob/master/poisson_process.ipynb
            #  Bernoulli approximation of homogenous Poisson process. 
            # Note that for this to work, the bin has to be chosen small enough that at most only a single event can appear within each of them. 
            # Output is a 1D binary sequence, which is a binned representation of the train of spikes.
            
            # from units import Hz, ms, s

            rate = rate * b2.Hz    # spike rate
            bin_size = bin_size * b2.ms # bin size 
            tmax = tmax * b2.ms *1000

            nbins = np.floor(tmax/bin_size).astype(int) + 1
            prob_of_spike = rate * bin_size
            spikes = np.random.rand(nbins) < prob_of_spike
            return spikes * 1
        
        evoked_spikes = [homogeneous_poisson(self.params["lambda_p"], self.params["stim_end_t"] - self.params["stim_start_t"] , self.params["stimulus_dt"]) for i in range(pop.neurons._N)]

        return np.array(evoked_spikes)


class ExternalPulseExperiment(Experiment):
    STIM_STATUS = 'off'
    train_start = 0 * b2.second
    train_end = 0 * b2.second
    
    name = 'feed forward spontaneous simulation experiment with external stimuli'
    description = 'simulating feed forward models into the PAG from ACC, VMH, IC, SC, PMd, but with external pulses from half time onwards'
    
    def __init__(self, model_class, model_params_path, exp_params_path):
        super().__init__(
            model_class, model_params_path, exp_params_path, name="Pulses"
        )
        if self.params["use_pulse_spont"]:
            self.evoked_poisson_spiketrain = []
            for i, population in enumerate(self.model.neuron_populations):
                if population is None or isinstance(population, PAG):
                    self.evoked_poisson_spiketrain.append(None)
                else:
                    self.evoked_poisson_spiketrain.append(self._pregenerate_poisson_stimulation_times(population))
            self.stim_bin_i=0
    
    def stimuli_func(self, t):
        
        '''
        1)Simulate X seconds with Brian. In this case the X second would be split into two consecutive epochs: first part spontaneous activity,
        second part a series of train of pulses, simulating some stimuli coming into the neurons.
        You could do things like 100 1 sec long 10Hz pulse trains, with an ITI of 1 sec.
        For the pulses you can increase the spontaneous activity of the neurons for like 20 ms.
        2)Use the spontaneous activity part to generate a training set and fit the GLM with that.
        3) Use the pulse epoch of the presynaptic activity to predict the activity of the PAG neuron with the GLM you fitted in 2.
        '''
        time = t.variable.get_value()[0] * b2.second
        t_start = self.params["stim_start_t"] * 1000 * b2.ms
        t_end = self.params["stim_end_t"] * 1000 * b2.ms
        stimulus_strength = self.params["stim_strength"] * b2.namp
        train_strength = self.params["train_strength"] * b2.namp
        
        # start/stop stimulus
        if t >= t_start and t < t_end and self.params["use_pulse"]:
            if self.STIM_STATUS == "off":
                logger.debug(f" === Stimulus ON at time: {time}")
                self.STIM_STATUS = "on"
                
            for i, population in enumerate(self.model.neuron_populations):
                if population is not None and not isinstance(population, PAG):
                    population.neurons.I_stim = stimulus_strength * self.evoked_poisson_spiketrain[i][:, self.stim_bin_i]
            self.stim_bin_i +=1
    
        # start/stop stimulus spont
        if t >= t_start and t < t_end and self.params["use_pulse_spont"]:
            # self.stim_target_idx = [[list(np.arange(pop.neurons._N))] for pop in self.model.neuron_populations[:-1]]
            for i, population in enumerate(self.model.neuron_populations):
                if population is not None and not isinstance(population, PAG):
                    population.neurons.I_stim = stimulus_strength * self.evoked_poisson_spiketrain[i][:, self.stim_bin_i]
        
        # start/stop train
        if t >= t_start and t <= t_end and self.params['use_train']:
            train_t = self.params["step_every"] * 1000 * b2.ms
            train_duration = self.params["step_duration"] * 1000 * b2.ms
            
            if t - self.train_end >= train_t and self.STIM_STATUS == "off":
                logger.debug(
                    f" -- Delivering train stimulus at time: {time:.4f} with duration {train_duration}"
                )
                self.STIM_STATUS = 'on'
                self.train_start = time
                for i, population in enumerate(self.model.neuron_populations):
                    if population is not None and not isinstance(population, PAG):
                        population.neurons.I_stim = train_strength
                        
            elif t - self.train_start >= train_duration and self.STIM_STATUS == 'on':
                logger.debug(
                    f" -- Stopping distractor stimulus at time {time:.4f}"
                )
                self.train_end = time
                self.STIM_STATUS = 'off'
                for i, population in enumerate(self.model.neuron_populations):
                    if population is not None and not isinstance(population, PAG):
                        population.neurons.I_stim = 0.0 * b2.amp
                
                
    def _pregenerate_poisson_stimulation_times(self, population):

        def homogeneous_poisson(rate, tmax, bin_size):
            ## https://github.com/btel/python-in-neuroscience-tutorials/blob/master/poisson_process.ipynb
            #  Bernoulli approximation of homogenous Poisson process. 
            # Note that for this to work, the bin has to be chosen small enough that at most only a single event can appear within each of them. 
            # Output is a 1D binary sequence, which is a binned representation of the train of spikes.
            
            # from units import Hz, ms, s

            rate = rate * b2.Hz    # spike rate
            bin_size = bin_size * b2.ms # bin size 
            tmax = tmax * b2.ms *1000

            nbins = np.floor(tmax/bin_size).astype(int) + 1
            prob_of_spike = rate * bin_size
            spikes = np.random.rand(nbins) < prob_of_spike
            return spikes * 1
        
        evoked_spikes = [homogeneous_poisson(self.params["lambda_p"], self.params["stim_end_t"] - self.params["stim_start_t"] , self.params["stimulus_dt"]) for i in range(population.neurons._N)]

        return np.array(evoked_spikes)