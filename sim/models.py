from pathlib import Path
from tpd import recorder as _recorder
from brian2 import PoissonInput
import brian2 as b2
from loguru import logger
import numpy as np

import matplotlib.pyplot as plt
from brian2.utils.logger import BrianLogger

from fcutils.plot.figure import clean_axes

from fcutils.path import from_yaml
import sys

sys.path.append("./")

from neurons import *
from utils import plot_connectivity_matrices


class FeedForwardModel:
    '''
        Feedforward model.
    '''

    name = 'ff model'
    b2_objects = []

    def __init__(self, params, recorder=None):
        
        if isinstance(params, (Path, str)):
            params = from_yaml(params)
        self.params = params
        logger.debug(f'Creating FeedForwardModel with parameters: \n{params}')

        # logging
        if params["b2_DEBUG"]:
            BrianLogger.log_level_debug()

        recorder = recorder if recorder is not None else _recorder
        recorder.start(
            base_folder=Path(params["save_path"]),
            folder_name=params["save_name"],
            timestamp=False,
        )

        self.VMH = VMH(n=self.params['n_vmh'])                     # ventromedial nucleus of the hypothalamus
        self.ACC = ACC(n=self.params['n_acc'])                      # anterior cingulate cortex
        self.IC = IC(n=self.params['n_ic'])                       # inferior coliculus
        self.SC = SC(n=self.params['n_sc'])                          # superior coliculus
        self.PMD = PMD(n=self.params['n_pmd'])                         # premotor cortex
        self.INH = InhNeuron(n=self.params['n_inh'])

        self.PAG = PAG(n=self.params['n_pag'])

        # collect all neuron popolations into brain 2 objexts
        self.neuron_populations = [self.VMH, self.ACC, self.IC, self.SC, self.PMD, self.INH, self.PAG]
        for population in self.neuron_populations:
            if population is not None:
                self.b2_objects.extend(
                    [*population.objects]
                )

        self.get_poisson_inputs() # poisson inputs to all areas projecting to PAG

        self.generate_connections()

    def get_poisson_inputs(self):
        """
            Create populations of poisson spiking units to provide
            inputs to all neurons populations EXCEPT PAG
        """

        for population in self.neuron_populations:
            if not isinstance(population, PAG) and population is not None:

                self.b2_objects.extend([PoissonInput(
                    target=population.neurons,
                    target_var='s_spont',
                    N=self.params["n_external"], # may customise for each neuronal population
                    rate=self.params["external_firing_rate"][population.name.lower()] * b2.Hz,
                    weight=1.0)])
    
    def generate_connections(self):
        """
            Generates synapses for the network
        """
        logger.debug("Creating synapses")
        self.synapses = []
        
        # VMH -> PAG
        if self.VMH is not None and self.params['enabled_synapses']['vmh2pag']:
            self.syn_vmh2pag = self.VMH.make_synapses(
                name='vmh2pag',
                target=self.PAG,
                source='vmh',
                stype=self.params['stype']['vmh2pag'],
                plasticity='off', # set all plasticity to off
                p=self.params["synapses_probability"]["vmh2pag"],
                X=self.params["X"]["vmh2pag"],
                U=self.params["U"]["vmh2pag"],
                tau1_early=self.params["tau1_early"]["vmh2pag"],
                tau2_early=self.params["tau2_early"]["vmh2pag"],
                tau1_late=self.params["tau1_late"]["vmh2pag"],
                tau2_late=self.params["tau2_late"]["vmh2pag"],
                g_ratio = self.params["synapse_gmax_late"]["vmh2pag"]/ self.params["synapse_gmax_early"]["vmh2pag"],
                E_rev = self.params["E_rev"]['AMPA']
            )
            self.VMH.set_synapses_weights(
                weight = self.params["synapse_gmax_early"]["vmh2pag"] * self.params["synapse_gmax_scaling"]["vmh2pag"],
                std = self.params["synapse_gmax_noise_std"]["vmh2pag"],
                target_synapse_silencing_probability = self.params["target_synapse_silencing_probability"]["vmh2pag"],
                source_synapse_silencing_probability = self.params["source_synapse_silencing_probability"]["vmh2pag"],
            )
            
            self.synapses.append(self.syn_vmh2pag)
            
        # ACC -> PAG
        if self.ACC is not None and self.params['enabled_synapses']['acc2pag']:
            self.syn_acc2pag = self.ACC.make_synapses(
                name='acc2pag',
                target=self.PAG,
                source='acc',
                stype=self.params['stype']['acc2pag'],
                plasticity='off', # set all plasticity to off for now
                # g_early_plasticity_correction = self.params["U_experimental"]["acc2pag"]/self.params["U"]["acc2pag"],
                p=self.params["synapses_probability"]["acc2pag"],
                # tauf=self.params["tauf"]["acc2pag"],
                # taud=self.params["taud"]["acc2pag"],
                X=self.params["X"]["acc2pag"],
                U=self.params["U"]["acc2pag"],
                tau1_early=self.params["tau1_early"]["acc2pag"],
                tau2_early=self.params["tau2_early"]["acc2pag"],
                tau1_late=self.params["tau1_late"]["acc2pag"],
                tau2_late=self.params["tau2_late"]["acc2pag"],
                g_ratio = self.params["synapse_gmax_late"]["acc2pag"]/ self.params["synapse_gmax_early"]["acc2pag"],
                E_rev = self.params["E_rev"]['AMPA']
            )
            self.ACC.set_synapses_weights(
                weight = self.params["synapse_gmax_early"]["acc2pag"] * self.params["synapse_gmax_scaling"]["acc2pag"],
                std = self.params["synapse_gmax_noise_std"]["acc2pag"],
                target_synapse_silencing_probability = self.params["target_synapse_silencing_probability"]["acc2pag"],
                source_synapse_silencing_probability = self.params["source_synapse_silencing_probability"]["acc2pag"],
            )

            self.synapses.append(self.syn_acc2pag)
        
        # IC -> PAG
        if self.IC is not None and self.params['enabled_synapses']['ic2pag']:
            self.syn_ic2pag = self.IC.make_synapses(
                name='ic2pag',
                target=self.PAG,
                source='ic',
                stype=self.params['stype']['ic2pag'],
                plasticity='off', # set all plasticity to off for now
                # g_early_plasticity_correction = self.params["U_experimental"]["acc2pag"]/self.params["U"]["acc2pag"],
                p=self.params["synapses_probability"]["ic2pag"],
                # tauf=self.params["tauf"]["acc2pag"],
                # taud=self.params["taud"]["acc2pag"],
                X=self.params["X"]["ic2pag"],
                U=self.params["U"]["ic2pag"],
                tau1_early=self.params["tau1_early"]["ic2pag"],
                tau2_early=self.params["tau2_early"]["ic2pag"],
                tau1_late=self.params["tau1_late"]["ic2pag"],
                tau2_late=self.params["tau2_late"]["ic2pag"],
                g_ratio = self.params["synapse_gmax_late"]["ic2pag"]/ self.params["synapse_gmax_early"]["ic2pag"],
                E_rev = self.params["E_rev"]['AMPA']
            )
            self.IC.set_synapses_weights(
                weight = self.params["synapse_gmax_early"]["ic2pag"] * self.params["synapse_gmax_scaling"]["ic2pag"],
                std = self.params["synapse_gmax_noise_std"]["ic2pag"],
                target_synapse_silencing_probability = self.params["target_synapse_silencing_probability"]["ic2pag"],
                source_synapse_silencing_probability = self.params["source_synapse_silencing_probability"]["ic2pag"],
            )
            self.synapses.append(self.syn_ic2pag)

        # SC -> PAG
        if self.SC is not None and self.params['enabled_synapses']['sc2pag']:
            self.syn_sc2pag = self.SC.make_synapses(
                name='sc2pag',
                target=self.PAG,
                source='sc',
                stype=self.params['stype']['sc2pag'],
                plasticity='off', # set all plasticity to off for now
                # g_early_plasticity_correction = self.params["U_experimental"]["acc2pag"]/self.params["U"]["acc2pag"],
                p=self.params["synapses_probability"]["sc2pag"],
                # tauf=self.params["tauf"]["acc2pag"],
                # taud=self.params["taud"]["acc2pag"],
                X=self.params["X"]["sc2pag"],
                U=self.params["U"]["sc2pag"],
                tau1_early=self.params["tau1_early"]["sc2pag"],
                tau2_early=self.params["tau2_early"]["sc2pag"],
                tau1_late=self.params["tau1_late"]["sc2pag"],
                tau2_late=self.params["tau2_late"]["sc2pag"],
                g_ratio = self.params["synapse_gmax_late"]["sc2pag"]/ self.params["synapse_gmax_early"]["sc2pag"],
                E_rev = self.params["E_rev"]['AMPA']
            )
            self.SC.set_synapses_weights(
                weight = self.params["synapse_gmax_early"]["sc2pag"] * self.params["synapse_gmax_scaling"]["sc2pag"],
                std = self.params["synapse_gmax_noise_std"]["sc2pag"],
                target_synapse_silencing_probability = self.params["target_synapse_silencing_probability"]["sc2pag"],
                source_synapse_silencing_probability = self.params["source_synapse_silencing_probability"]["sc2pag"],
            )

            self.synapses.append(self.syn_sc2pag)

        # Inhibitory -> PAG
        if self.INH is not None and self.params['enabled_synapses']['inh2pag']:
            self.syn_inh2pag = self.INH.make_synapses(
                name='inh2pag',
                target=self.PAG,
                source='inh',
                stype=self.params['stype']['inh2pag'],
                plasticity='off', # set all plasticity to off for now
                # g_early_plasticity_correction = self.params["U_experimental"]["vmh2pag"]/self.params["U"]["vmh2pag"],
                p=self.params["synapses_probability"]["inh2pag"],
                # tauf=self.params["tauf"]["vmh2pag"],
                # taud=self.params["taud"]["vmh2pag"],
                X=self.params["X"]["inh2pag"],
                U=self.params["U"]["inh2pag"],
                tau1_early=self.params["tau1_early"]["inh2pag"],
                tau2_early=self.params["tau2_early"]["inh2pag"],
                tau1_late=self.params["tau1_late"]["inh2pag"],
                tau2_late=self.params["tau2_late"]["inh2pag"],
                g_ratio = self.params["synapse_gmax_late"]["inh2pag"]/ self.params["synapse_gmax_early"]["inh2pag"],
                E_rev = self.params["E_rev"]['AMPA']
            )
            self.INH.set_synapses_weights(
                weight = self.params["synapse_gmax_early"]["inh2pag"] * self.params["synapse_gmax_scaling"]["inh2pag"],
                std = self.params["synapse_gmax_noise_std"]["inh2pag"],
                target_synapse_silencing_probability = self.params["target_synapse_silencing_probability"]["inh2pag"],
                source_synapse_silencing_probability = self.params["source_synapse_silencing_probability"]["inh2pag"],
            )
            
            self.synapses.append(self.syn_inh2pag)
            
            # PMD -> PAG
        if self.PMD is not None and self.params['enabled_synapses']['pmd2pag']:
            self.syn_pmd2pag = self.PMD.make_synapses(
                name='pmd2pag',
                target=self.PAG,
                source='pmd',
                stype=self.params['stype']['pmd2pag'],
                plasticity='off', # set all plasticity to off for now
                # g_early_plasticity_correction = self.params["U_experimental"]["vmh2pag"]/self.params["U"]["vmh2pag"],
                p=self.params["synapses_probability"]["pmd2pag"],
                # tauf=self.params["tauf"]["vmh2pag"],
                # taud=self.params["taud"]["vmh2pag"],
                X=self.params["X"]["pmd2pag"],
                U=self.params["U"]["pmd2pag"],
                tau1_early=self.params["tau1_early"]["pmd2pag"],
                tau2_early=self.params["tau2_early"]["pmd2pag"],
                tau1_late=self.params["tau1_late"]["pmd2pag"],
                tau2_late=self.params["tau2_late"]["pmd2pag"],
                g_ratio = self.params["synapse_gmax_late"]["pmd2pag"]/ self.params["synapse_gmax_early"]["pmd2pag"],
                E_rev = self.params["E_rev"]['AMPA']
            )
            self.PMD.set_synapses_weights(
                weight = self.params["synapse_gmax_early"]["pmd2pag"] * self.params["synapse_gmax_scaling"]["pmd2pag"],
                std = self.params["synapse_gmax_noise_std"]["pmd2pag"],
                target_synapse_silencing_probability = self.params["target_synapse_silencing_probability"]["pmd2pag"],
                source_synapse_silencing_probability = self.params["source_synapse_silencing_probability"]["pmd2pag"],
            )
            
            self.synapses.append(self.syn_pmd2pag)

        self.b2_objects.extend(self.synapses)

    def visualize_synapses(self):
        pass


def unif_if_undef(param, _min=0, _max=0):
    return param if param is not None else np.random.uniform(_min, _max)

def normal_if_undef(param, mean, std, min_clip=-float('inf'), max_clip=float('inf')):
    return param if param is not None else np.clip(np.random.normal(mean, std), min_clip, max_clip)