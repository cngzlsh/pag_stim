from brian2 import NeuronGroup, SpikeMonitor, StateMonitor, Synapses
from loguru import logger
import brian2 as b2
import numpy as np
from scipy.stats import norm


"""
    The following neurons classes are from
    the working memory example in the "Neuronal Dynamics" book, 
    which is implementing Compte et atl 2000.

        Implementation of a working memory model.
        Literature:
        Compte, A., Brunel, N., Goldman-Rakic, P. S., & Wang, X. J. (2000). Synaptic mechanisms and
        network dynamics underlying spatial working memory in a cortical network model.
        Cerebral Cortex, 10(9), 910-923.

        Some parts of this implementation are inspired by material from
        *Stanford University, BIOE 332: Large-Scale Neural Modeling, Kwabena Boahen & Tatiana Engel, 2013*,
        online available.

        This file is part of the exercise code repository accompanying
        the book: Neuronal Dynamics (see http://neuronaldynamics.epfl.ch)
        located at http://github.com/EPFL-LCN/neuronaldynamics-exercises.

        This free software: you can redistribute it and/or modify it under
        the terms of the GNU General Public License 2.0 as published by the
        Free Software Foundation. You should have received a copy of the
        GNU General Public License along with the repository. If not,
        see http://www.gnu.org/licenses/.

        Should you reuse and publish the code for your own purposes,
        please cite the book or point to the webpage http://neuronaldynamics.epfl.ch.

        Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski.
        Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
        Cambridge University Press, 2014.

"""


class conductancebasedLIF:
    eqs = ""  # to be specified by subclasses

    # ----------------------------- plasticity params ---------------------------- #

    # ---------------------------------- params ---------------------------------- #
    reset = ""  # reset expression
    threshold = ""  # threshold expression

    threshold_v: float = 0 * b2.mV  # when voltage > th spike
    reset_v: float = 0 * b2.mV  # reset voltage
    refractory: int = 10 * b2.ms

    Cm: float = 0 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak: float = 0 * b2.nS  # leak conductance
    E_leak: float = -60 * b2.mV  # reversal potential

    G_spont =  1 * b2.nS # 0.381 * b2.nS #changed on 21 april to bring spont G in other synapse g range
    E_spont = 0.0 * b2.mV
    tau_spont = 0.9 * 2.0 * b2.ms
      
    # projections from the external population
    G_extern2inhib = 2.38 * b2.nS
    G_extern2excit = 3.1 * b2.nS

    # projectsions from the inhibitory populations
    weight_scaling_factor = 8
    G_inhib2inhib = 1.024 * b2.nS  # scaled by a weight scaling factor
    G_inhib2excit = 1.336 * b2.nS  # scaled by a weight scaling factor

    # projections from the excitatory population
    G_excit2excit = 0.381 * b2.nS  # scaled by a weight scaling factor
    G_excit2inhib = 0.292 * b2.nS  # scaled by a weight scaling factor

    # specify the AMPA synapses
    E_AMPA = 0.0 * b2.mV
    tau_AMPA = 0.9 * 2.0 * b2.ms

    # specify the GABA synapses
    E_GABA = -70.0 * b2.mV
    tau_GABA = 10.0 * b2.ms

    

    # ---------------------------- synaptic plasticity --------------------------- #

    def __init__(self, n=100):
        self.n = n
        self.synapses = None
        logger.info(
            f"Creating population: {self.name.upper()} with {n} neurons."
        )
        # create neurons
        self.neurons = NeuronGroup(
            n,
            model=self.eqs,
            name=self.name,
            threshold=self.threshold,
            reset=self.reset,
            refractory=self.refractory,
            method="euler",
            namespace=dict(
                G_extern2inhib=self.G_extern2inhib,
                G_extern2excit=self.G_extern2excit,
                G_inhib2inhib=self.G_inhib2inhib * self.weight_scaling_factor,
                G_inhib2excit=self.G_inhib2excit * self.weight_scaling_factor,
                G_excit2excit=self.G_excit2excit * self.weight_scaling_factor,
                G_excit2inhib=self.G_excit2inhib * self.weight_scaling_factor,
                threshold_v=self.threshold_v,
                reset_v=self.reset_v,
                refractory=self.refractory,
                Cm=self.Cm,
                G_leak=self.G_leak,
                E_leak=self.E_leak,
                E_AMPA=self.E_AMPA,
                tau_AMPA=self.tau_AMPA,
                E_GABA=self.E_GABA,
                tau_GABA=self.tau_GABA,
                G_spont=self.G_spont,
                E_spont=self.E_spont,
                tau_spont=self.tau_spont

            ),
        )
        # initialize random voltage, no inputs
        self.neurons.v = (
            np.random.uniform(
                self.reset_v / b2.mV, high=self.threshold_v / b2.mV, size=n
            )
            * b2.mV
        )

        self.neurons.v = self.E_leak

        

        

        try:
            self.neurons.I_stim = 0.0 * b2.namp
        except AttributeError:
            # inhibitory neurons don't have I_stim
            pass

        # monitor variables
        self.spikes = SpikeMonitor(
            self.neurons, name=self.name + "_spikes_monitor"
        )
        self.voltages = StateMonitor(
            self.neurons,
            "v",
            record=True,
            name=self.name + "_voltages_monitor",
        )

        self.objects = [self.neurons, self.spikes, self.voltages]

        self.log_params()

    def __repr__(self):
        return f"Population: {self.name}"

    def __str__(self):
        return self.name

    def make_synapses(
        self,
        name,
        target,
        source: str = "rsp",
        stype: str = "event-driven",
        plasticity: str = None,
        g_early_plasticity_correction: float = 1,
        tauf: float = 1,
        taud: float = 1,
        X: float = 1,
        U: float = 1,
        tau1_early: float = 1,
        tau2_early: float = 2,
        tau1_late: float = 1,
        tau2_late: float = 2,
        g_ratio = 0.1,
        E_rev: float = 0,
        p=1.0,
        method='exact',
        **kwargs,
    ):

        # define synapses model
        I_syn = f"I_{source}_post" 
        
        
            

        if plasticity == "off":

            if stype == 'event-driven':
                model = f"""w: siemens
                    dg_syn_early/dt = -g_syn_early/tau2_early : siemens (clock-driven)
                    {I_syn} = (g_syn_early) * (v_post-E_rev): amp (summed)
                """
                on_pre = "g_syn_early += w"

            if stype == 'biExp':

                model = f"""w: siemens
                    ds_syn_early/dt = -s_syn_early/tau2_early : siemens  (clock-driven)
                    dg_syn_early/dt = ((tau2_early / tau1_early) ** (tau1_early / (tau2_early - tau1_early))*s_syn_early - g_syn_early)/tau1_early : siemens (clock-driven)
                    
                    
                    ds_syn_late/dt = -s_syn_late/tau2_late : siemens  (clock-driven)
                    dg_syn_late/dt = ((tau2_late / tau1_late) ** (tau1_late / (tau2_late - tau1_late))*s_syn_late - g_syn_late)/tau1_late : siemens (clock-driven)
                    {I_syn} = (g_syn_early + g_syn_late) * (v_post-E_rev): amp (summed)
                    """
                on_pre = """s_syn_early += w


                            s_syn_late += w*g_ratio
                        """
            namespace = dict(tau1_early=tau1_early * b2.ms, tau2_early=tau2_early * b2.ms, tau1_late = tau1_late * b2.ms, tau2_late = tau2_late * b2.ms, E_rev=E_rev * b2.mV, g_ratio = g_ratio, method = method)
            setx, setu = False, False
        
        elif plasticity == "both":
            # see: https://deepnote.com/project/campindia2016-Df8W19g2TVmvTV6UtIGuaQ/%2Ftutorials%2Fshort%20term%20synaptic%20plasticity%2FShort%20Term%20Plasticity.ipynb/#00017-776e9c42-2614-44de-bf30-04b9ae8c5850

            if stype == 'event-driven':
                model = f"""w: siemens
                    dg_syn_early/dt = -g_syn_early/tau2_early : siemens
                    dx/dt= (1-x) / taud : 1 (event-driven)
                    du/dt= (U-u) / tauf : 1 (event-driven)
                    {I_syn} = (g_syn_early) * (v_post-E_rev): amp (summed)
                """
                on_pre = """                    
                        u += U*(1-u)
                        g_syn_early += w * u * x
                        x -= x*u
                """

            if stype == 'biExp':

                model = f"""w: siemens
                    ds_syn_early/dt = -s_syn_early/tau2_early : siemens  (clock-driven)
                    dg_syn_early/dt = ((tau2_early / tau1_early) ** (tau1_early / (tau2_early - tau1_early))*s_syn_early - g_syn_early)/tau1_early : siemens (clock-driven)
                    ds_syn_late/dt = -s_syn_late/tau2_late : siemens  (clock-driven)
                    dg_syn_late/dt = ((tau2_late / tau1_late) ** (tau1_late / (tau2_late - tau1_late))*s_syn_late - g_syn_late)/tau1_late : siemens (clock-driven)
                    {I_syn} = (g_syn_early + g_syn_late) * (v_post-E_rev): amp (summed)
                    dx/dt= (X-x) / taud : 1 (clock-driven)
                    du/dt= (U-u) / tauf : 1 (clock-driven)
                    """
                
                on_pre = """ 
                        u += U*(1-u)
                        s_syn_early += w *g_early_plasticity_correction * u * x  
                        x -= x*u
                        s_syn_late += w*g_ratio
                        """
            setx, setu = True, True
            namespace = dict(
                tau1_early=tau1_early * b2.ms, tau2_early=tau2_early * b2.ms, tau1_late = tau1_late * b2.ms, tau2_late = tau2_late * b2.ms, E_rev=E_rev * b2.mV, g_ratio = g_ratio, method = method,
                U=U, X=X, taud= taud * b2.ms, tauf = tauf * b2.ms, g_early_plasticity_correction=g_early_plasticity_correction)

        else:
            raise ValueError(
                f'Plasticity rule "{plasticity}" not recognized while creating {stype} synapses between: "{self.name}" and "{target.name}".'
            )


        logger.info(
            f'!!! Creating {stype} synapses, reversal {E_rev} mV, between: "{self.name}" and "{target.name}". Plasticity: "{plasticity}"\nOn_pre:{on_pre}'
        )

        synapses = Synapses(
            self.neurons,
            target.neurons,
            name=name,
            on_pre=on_pre,
            model=model,
            namespace=namespace,
            method=method,
            # **kwargs,
        )
        synapses.connect(p=p,)
        synapses._n_neurons = (self.n, target.n)
         
        synapses.g_syn_early = 0
        if stype == 'biExp':
            synapses.s_syn_early = 0
            synapses.s_syn_late = 0 
            synapses.g_syn_late = 0
        if setx:
            synapses.x = X
        if setu:
            synapses.u = U

        
        
        self.synapses = synapses
        return synapses

    def get_synapses_weights_kernel(
        self, kernel: str, kernel_width: float, connection_probability: float,
    ):
        """
            Computes a kernel (gaussian shaped, as a fn of distance from the diagonal).
            Width -> std of gaussian expressed in degrees
        """
        # N = self.synapses.N_incoming_post[0]
        # M = self.synapses.N_outgoing_pre[0]
        N, M = self.synapses._n_neurons
        if N != M:
            raise ValueError(
                "This only works for squared connectivity matrices!!"
            )

        # if connectivity had prob < 1 not all weights will be there
        if N * M != self.synapses.w.shape[0]:
            # create a mask to remove null ones
            W = np.full((N, M), np.nan)
            W[self.synapses.i, self.synapses.j] = 1
        else:
            W = np.ones((N, M))

        if kernel == "ones":
            x = np.ones((N, N)) * W
        else:
            factor = N / 360
            sigma = kernel_width * factor
            rv = norm(loc=0, scale=sigma)

            # compute distance from diagonal
            idx = np.arange(N).reshape(N, 1)
            distances = np.sqrt((idx - idx.T) ** 2)  # N x N

            # put through normal and normalize
            x = rv.pdf(distances)
            x /= np.max(x)

            # ensure weights are periodc
            n = len(x)
            nhalf = int(n / 2)
            k0 = x[0, 0:nhalf]
            k = np.hstack((k0[::-1], k0, k0[::-1], k0, k0[::-1], k0))

            for i in range(n):
                x[i, :] = k[n + nhalf - i : n + nhalf + n - i]

            # get diagonal/inverted diaonal
            if kernel == "diagonal":
                x = x * W
            elif kernel == "inverted_diagonal":
                x = (1 - x) * W
            else:
                raise ValueError(f'Kernel type "{kernel}" not recognized')

        # remove null connections (i.e. when neuron in B doesn't receive from neurons in A)
        n_skipped = int(round(N * (1 - connection_probability)))
        if n_skipped:
            selected_idx = np.arange(N)
            np.random.shuffle(selected_idx)
            selected_idx = selected_idx[:n_skipped]
            x[:, selected_idx] *= 0

        return x.flatten()

    def set_synapses_weights(
        self,
        weight: float,
        std: float,
        target_synapse_silencing_probability: float,
        source_synapse_silencing_probability: float,
    ):
        logger.debug(
            f"Created synapses: {self.synapses.name} with weights: {weight:.4f} and std: {std:.4f}"
        )

        # get the weights according to the connectivity rule
        if std == 0:
            weights = np.ones(self.synapses.w.shape) * weight
        else:
            # create weights matrix with noise.
            weights = np.random.normal(weight, std, size=self.synapses.w.shape)

        num_connections = len(self.synapses.w)
        _from = self.synapses.i
        _to = self.synapses.j
        _silenced_connection_count = 0

        # target synapsic silencing: PAG neuron receives no inputs from a source population at all
        silenced_targets = np.unique(
            np.arange(self.synapses.target.N) * np.random.binomial(n=1, p=target_synapse_silencing_probability, size=self.synapses.target.N)
            )
        for i, tgt in enumerate(_to):
            if tgt in silenced_targets:
                weights[i] = 0
                _silenced_connection_count += 1

        # source synaptic silencing
        silenced_sources = np.unique(
            np.arange(self.synapses.source.N) * np.random.binomial(n=1, p=source_synapse_silencing_probability, size=self.synapses.source.N)
            )
        for j, src in enumerate(_from):
            if src in silenced_sources:
                weights[j] = 0
                _silenced_connection_count += 1
                
        self.synapses.w = weights * b2.nS

        logger.debug(
            f"{_silenced_connection_count} connections dropped in total."
        )

    def log_params(self):
        """
            Saves cell parameters to logs
        """
        logger.debug(
            f"""
    CELL PARAMETERS: {self.name}
----------------------------------------------
    reset = {self.reset}
    threshold = {self.threshold}

    threshold_v = {self.threshold_v}
    reset_v = {self.reset_v}
    refractory = {self.refractory}

    Cm = {self.Cm}
    G_leak = {self.G_leak}
    E_leak = {self.E_leak}

    # projections from the external population
    G_extern2inhib = {self.G_extern2inhib}
    G_extern2excit = {self.G_extern2excit}

    # projectsions from the inhibitory populations
    weight_scaling_factor = {self.weight_scaling_factor}
    G_inhib2inhib = {self.G_inhib2inhib}
    G_inhib2excit = {self.G_inhib2excit}

    # projections from the excitatory population
    G_excit2excit = {self.G_excit2excit}
    G_excit2inhib = {self.G_excit2inhib}

    # specify the AMPA synapses
    E_AMPA = {self.E_AMPA}
    tau_AMPA = {self.tau_AMPA}

    # specify the GABA synapses
    E_GABA = {self.E_GABA}
    tau_GABA = {self.tau_GABA}

----------------------------------------------
----------------------------------------------
        """
        )
