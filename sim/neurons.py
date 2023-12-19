import brian2 as b2
import numpy as np

from lif import conductancebasedLIF
from brian2 import StateMonitor

'''
neuron group classes, same LIF neurons in brian2_nn/drnn/neurons
create 2 classes of NeuronGroup, currently using the same paramters as RSP
'''

# UPSTREAM NEURONS

class ACC(conductancebasedLIF):
    name = "ACC"
    eqs = """
        I_stim : amp
        dv/dt = (
        - G_leak * (v-E_leak)
        + I_stim
        - G_spont * s_spont * (v-E_spont)
        )/Cm : volt (unless refractory)
        ds_spont/dt = -s_spont/tau_spont : 1
    """
    reset = "v=reset_v;"  # reset expression
    threshold = "v>threshold_v"  # threshold expression

    # ---------------------------------- params ---------------------------------- #
    threshold_v = -50 * b2.mV  # when voltage > th: spike
    reset_v = -60 * b2.mV  # reset voltage
    refractory = 2 * b2.ms

    # cunductances etc
    Cm = 0.5 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak = 25.0 * b2.nS  # leak conductance
    E_leak = -70.0 * b2.mV  # reversal potential

    clamp = 'IC'

    def __init__(self, n=100):
        super().__init__(n=n)

        # create a monitor to keep track of the inputs
        self.inputs_monitor = StateMonitor(self.neurons, "I_stim", record=True)
        self.objects.append(self.inputs_monitor)


class VMH(conductancebasedLIF):
    name = "VMH"
    eqs = """
        I_stim : amp
        dv/dt = (
        - G_leak * (v-E_leak)
        + I_stim
        - G_spont * s_spont * (v-E_spont)
        )/Cm : volt (unless refractory)
        ds_spont/dt = -s_spont/tau_spont : 1
    """
    reset = "v=reset_v;"  # reset expression
    threshold = "v>threshold_v"  # threshold expression

    # ---------------------------------- params ---------------------------------- #
    threshold_v = -50 * b2.mV  # when voltage > th: spike
    reset_v = -60 * b2.mV  # reset voltage
    refractory = 2 * b2.ms

    # cunductances etc
    Cm = 0.5 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak = 25.0 * b2.nS  # leak conductance
    E_leak = -70.0 * b2.mV  # reversal potential

    clamp = 'IC'

    def __init__(self, n=100):
        super().__init__(n=n)

        # create a monitor to keep track of the inputs
        self.inputs_monitor = StateMonitor(self.neurons, "I_stim", record=True)
        self.objects.append(self.inputs_monitor)

class IC(conductancebasedLIF):
    name = "IC"
    eqs = """
        I_stim : amp
        dv/dt = (
        - G_leak * (v-E_leak)
        + I_stim
        - G_spont * s_spont * (v-E_spont)
        )/Cm : volt (unless refractory)
        ds_spont/dt = -s_spont/tau_spont : 1
    """
    reset = "v=reset_v;"  # reset expression
    threshold = "v>threshold_v"  # threshold expression

    # ---------------------------------- params ---------------------------------- #
    threshold_v = -50 * b2.mV  # when voltage > th: spike
    reset_v = -60 * b2.mV  # reset voltage
    refractory = 2 * b2.ms

    # cunductances etc
    Cm = 0.5 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak = 25.0 * b2.nS  # leak conductance
    E_leak = -70.0 * b2.mV  # reversal potential

    clamp = 'IC'

    def __init__(self, n=100):
        super().__init__(n=n)

        # create a monitor to keep track of the inputs
        self.inputs_monitor = StateMonitor(self.neurons, "I_stim", record=True)
        self.objects.append(self.inputs_monitor)

class SC(conductancebasedLIF):
    name = "SC"
    eqs = """
        I_stim : amp
        dv/dt = (
        - G_leak * (v-E_leak)
        + I_stim
        - G_spont * s_spont * (v-E_spont)
        )/Cm : volt (unless refractory)
        ds_spont/dt = -s_spont/tau_spont : 1
    """
    reset = "v=reset_v;"  # reset expression
    threshold = "v>threshold_v"  # threshold expression

    # ---------------------------------- params ---------------------------------- #
    threshold_v = -50 * b2.mV  # when voltage > th: spike
    reset_v = -60 * b2.mV  # reset voltage
    refractory = 2 * b2.ms

    # cunductances etc
    Cm = 0.5 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak = 25.0 * b2.nS  # leak conductance
    E_leak = -70.0 * b2.mV  # reversal potential

    clamp = 'IC'

    def __init__(self, n=100):
        super().__init__(n=n)

        # create a monitor to keep track of the inputs
        self.inputs_monitor = StateMonitor(self.neurons, "I_stim", record=True)
        self.objects.append(self.inputs_monitor)

class PMD(conductancebasedLIF):
    name = "PMD"
    eqs = """
        I_stim : amp
        dv/dt = (
        - G_leak * (v-E_leak)
        + I_stim
        - G_spont * s_spont * (v-E_spont)
        )/Cm : volt (unless refractory)
        ds_spont/dt = -s_spont/tau_spont : 1
    """
    reset = "v=reset_v;"  # reset expression
    threshold = "v>threshold_v"  # threshold expression

    # ---------------------------------- params ---------------------------------- #
    threshold_v = -50 * b2.mV  # when voltage > th: spike
    reset_v = -60 * b2.mV  # reset voltage
    refractory = 2 * b2.ms

    # cunductances etc
    Cm = 0.5 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak = 25.0 * b2.nS  # leak conductance
    E_leak = -70.0 * b2.mV  # reversal potential

    clamp = 'IC'

    def __init__(self, n=100):
        super().__init__(n=n)

        # create a monitor to keep track of the inputs
        self.inputs_monitor = StateMonitor(self.neurons, "I_stim", record=True)
        self.objects.append(self.inputs_monitor)

class InhNeuron(conductancebasedLIF):
    name = "InhNeuron"
    eqs = """
        I_stim : amp
        dv/dt = (
        - G_leak * (v-E_leak)
        + I_stim
        - G_spont * s_spont * (v-E_spont)
        )/Cm : volt (unless refractory)
        ds_spont/dt = -s_spont/tau_spont : 1
    """
    reset = "v=reset_v;"  # reset expression
    threshold = "v>threshold_v"  # threshold expression

    # ---------------------------------- params ---------------------------------- #
    threshold_v = -50 * b2.mV  # when voltage > th: spike
    reset_v = -60 * b2.mV  # reset voltage
    refractory = 2 * b2.ms

    # cunductances etc
    Cm = 0.5 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak = 25.0 * b2.nS  # leak conductance
    E_leak = -70.0 * b2.mV  # reversal potential

    clamp = 'IC'

    def __init__(self, n=100):
        super().__init__(n=n)

        # create a monitor to keep track of the inputs
        self.inputs_monitor = StateMonitor(self.neurons, "I_stim", record=True)
        self.objects.append(self.inputs_monitor)
    
# DOWNSTREAM NEURONS

class PAG(conductancebasedLIF):
    name = "PAG"
    eqs = """
        I_acc : amp
        I_vmh : amp
        I_ic : amp
        I_sc : amp
        I_pmd : amp
        dv/dt = (
        - G_leak * (v-E_leak)
        - I_acc
        - I_vmh
        - I_ic
        - I_sc
        - I_pmd
        - G_spont * s_spont * (v-E_spont)
        )/Cm : volt (unless refractory)
        ds_spont/dt = -s_spont/tau_spont : 1
    """
    reset = "v=reset_v;"  # reset expression
    threshold = "v>threshold_v"  # threshold expression

    # ---------------------------------- params ---------------------------------- #
    threshold_v = -37.665 * b2.mV  # when voltage > th: spike
    reset_v = -62.341 * b2.mV  # reset voltage
    refractory = 2 * b2.ms

    # cunductances etc
    Cm = 0.0468 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak = 17.858 * b2.nS  # leak conductance
    E_leak = -60.0 * b2.mV  # reversal potential

    clamp = 'IC'

    def __init__(self, n=100, clamp='IC'):

        self.clamp = clamp

        super().__init__(n=n)

        # create a monitor to keep track of the inputs
        self.inputs_acc_monitor = StateMonitor(self.neurons, "I_acc", record=True)
        self.objects.append(self.inputs_acc_monitor)
        self.inputs_vmh_monitor = StateMonitor(self.neurons, "I_vmh", record=True)
        self.objects.append(self.inputs_vmh_monitor)
        self.inputs_ic_monitor = StateMonitor(self.neurons, "I_ic", record=True)
        self.objects.append(self.inputs_ic_monitor)
        self.inputs_sc_monitor = StateMonitor(self.neurons, "I_sc", record=True)
        self.objects.append(self.inputs_sc_monitor)
        self.inputs_pmd_monitor = StateMonitor(self.neurons, "I_pmd", record=True)
        self.objects.append(self.inputs_pmd_monitor)