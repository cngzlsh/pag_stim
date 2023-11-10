from loguru import logger
import matplotlib.pyplot as plt
from myterial import indigo, salmon, teal_light
from brian2 import ms, mV, pA
import numpy as np
import brian2 as b2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from neurons import PAG

from fcutils.plot.figure import clean_axes

plot_every_n_neurons = 1
dropbox_folder = "./dropbox_folder"

def plot_rasters(pops, mode='all'):
    """
        Plots rasters and average trace for each population
    """

    f, axes = plt.subplots(nrows=len(pops), figsize=(12, 9), sharex=True)
    f._save_name = "spikes_rasters"
    f.suptitle("Spikes rasters")
    clean_axes(f)

    # spikes rasters
    colors = ('b', 'c', 'g', 'm', 'y', 'r')
    for ax, pop, color in zip(axes, pops, colors):
        if mode == 'all':
            ax.scatter(pop.spikes.t / ms, pop.spikes.i, s=5, color=color)
        elif mode == 'projection only':
            if isinstance(pop, PAG):
                ax.scatter(pop.spikes.t / ms, pop.spikes.i, s=5, color=color)
            else:
                zero_weights_conn_idx = np.where(pop.synapses.w==0)[0]
                projecting_src_neuron_idx = []
                for src_idx in np.arange(pop.synapses.source.N):
                    # check if all weights are zero
                    is_projecting = False
                    all_conn_from_src_idx = np.where(pop.synapses.i == src_idx)[0]
                    for conn in all_conn_from_src_idx:
                        if conn not in zero_weights_conn_idx:
                            is_projecting = True
                    if is_projecting:
                        projecting_src_neuron_idx.append(src_idx)
                
                logger.debug(f'Found {len(projecting_src_neuron_idx)} out of {pop.n} {pop.name} neurons projecting to PAG neurons.')
                logger.debug(f'Mean connection weight: {np.mean(pop.synapses.w[np.nonzero(pop.synapses.w)])}, std: {np.std(pop.synapses.w[np.nonzero(pop.synapses.w)])}')
                spike_t, src_t = [], []
                # only plot these indices
                for spike_idx, src_idx in enumerate(pop.spikes.i):
                    if src_idx in projecting_src_neuron_idx:
                        spike_t.append(pop.spikes.t[spike_idx])
                        src_t.append(pop.spikes.i[spike_idx])

                ax.scatter(spike_t / ms, src_t, s=5, color=color)
        

    # plot inputs to population
    if pops[0].inputs_monitor is not None:
        inputs = pops[0].inputs_monitor.I_stim[:] / b2.namp
        inputs[inputs == 0.0] = np.nan
        axes[0].imshow(
            inputs,
            cmap="Reds",
            zorder=200,
            extent=[0, pops[0].voltages.t[-1] / ms, 0, pops[0].n],
            aspect="auto",
            vmin=0,
            origin="lower",
            alpha=0.25,
        )

    # set axes
    for i, pop in enumerate(pops):
        axes[i].set(
            ylim=[0, pop.n],
            xlim=[0, pop.voltages.t[-1] / ms],
            ylabel=f"{pop.name} neurons",
            yticks=[0, pop.n / 2, pop.n],
            yticklabels=["0", f"{pop.n/2}", f"{pop.n}"],
        )
    plt.savefig('./figs/rasters.png')

def plot_weight_matrices(pops):

    f, axes = plt.subplots(ncols=len(pops)-1, figsize=(12, 9), sharey=True)
    f._save_name = "spikes_rasters"
    f.suptitle("Weight matrices")
    clean_axes(f)
    for ax, pop in zip(axes, pops):
        if isinstance(pop, PAG):
            break
        mat = np.zeros((pop.synapses.source.N, pop.synapses.target.N))
        for s, w in enumerate(pop.synapses.w):
            mat[pop.synapses.i[s], pop.synapses.j[s]] = w
        ax.matshow(mat, cmap='Blues')
    
    for i, pop in enumerate(pops[:-1]):
        axes[i].set(
        xlim=[0, pop.synapses.target.N],
        ylim=[0, pop.n],
        xlabel=f"{pop.synapses.target.name} neuron index",
        ylabel=f"{pop.name} neuron index",
    )
    plt.savefig('./figs/weight_matrices.png')


def plot_voltages(pops, expt_params, plot_synaptic_current='', plot_expt_inputs_from = ''):
    """
        Plots neuron voltage values for eaach population (subset of neurons)
    """
    f, axes = plt.subplots(nrows=len(pops), figsize=(12, 9), sharex=True)
    f._save_name = "voltage_traces"
    f.suptitle("Voltage traces")
    clean_axes(f)
    

    colors = (teal_light, salmon, indigo)
    for ax_voltages, pop, color in zip(axes, pops, colors):
        pop_name = pop.neurons.name

        if pop.clamp =='VC': 
            ax_currents = ax_voltages.twinx()
            ax_currents.set(ylabel="synaptic current (pA)"   )

        voltages = pop.voltages.v[::plot_every_n_neurons, :]

        #plot mean simulated voltage 
        mean_voltage = np.mean(voltages, axis =0)

        ax_voltages.plot(
            pop.voltages.t / ms,
            mean_voltage / mV,
            lw=2,
            alpha=0.9,
            color=color,
        )

        #plot exptal input trace 
        if plot_expt_inputs_from != '': 
            ax_expt = ax_voltages
            if pop.clamp =='VC': ax_expt = ax_currents
            plot_experimental_traces(ax_expt, pop, expt_params, plot_expt_inputs_from = plot_expt_inputs_from)            

        #get simulated synaptic current 
        if pop.clamp =='VC': 
            if 'inputs_rsp_monitor' in dir(pop): i_syn = pop.inputs_rsp_monitor.I_rsp[::plot_every_n_neurons, :]

        # if plot_synaptic_current != '':
            # if plot_synaptic_current == 'vgat' and 'inputs_vgat_monitor' in dir(pop): i_syn = pop.inputs_vgat_monitor.I_vgat[::plot_every_n_neurons, :]
            # if plot_synaptic_current == 'vglut' and 'inputs_vglut_monitor' in dir(pop): i_syn = pop.inputs_vglut_monitor.I_vglut[::plot_every_n_neurons, :]

        for i in range(voltages.shape[0]):
            #plot simulated voltage 
            ax_voltages.plot(
                pop.voltages.t / ms,
                voltages[i, :] / mV,
                lw=2,
                alpha=0.5,
                color=color,
            )

            #plot simulated synaptic current 
            try:
                if pop.clamp =='VC': # plot_synaptic_current == 'rsp' or (plot_synaptic_current == 'vgat' and pop_name == 'VGLUT'):    
                    ax_currents.plot(
                        pop.voltages.t / ms,
                        i_syn[i, :] / pA,
                        lw=2, 
                        ls ='--',
                        alpha=0.3,
                        color=color,
                    )
            except NameError:
                pass

            ax_voltages.axhline(pop.threshold_v / mV, lw=2, color="k", ls="--")
            ax_voltages.axhline(pop.reset_v / mV, lw=2, color="k")

    # set axes
    for i, pop in enumerate(pops):
        axes[i].set(
        xlim=[0, pop.voltages.t[-1] / ms], ylabel=f"{pop.name} (mV)",
    )


    # axes[0].set(
    #     xlim=[0, pops[0].voltages.t[-1] / ms], ylabel="RSP (mV)",
    # )
    # axes[1].set(
    #     xlim=[0, pops[1].voltages.t[-1] / ms], ylabel="VGLUT (mV)",
    # )

    # if len(pops) > 2:
    #     axes[2].set(
    #         ylabel="VGAT (mV)",
    #         xlim=[0, pops[2].voltages.t[-1] / ms],
    #         xlabel="time (ms)",
    #     )

def plot_single_neuron_V_trace(pop, idx='random'):
    if idx == 'random':
        idx = np.random.randint(pop.n)
    pop_name = pop.neurons.name

    plt.figure(figsize=(12, 3))
    voltages = pop.voltages.v[idx, :] 
    plt.plot(
        pop.voltages.t / ms,
        voltages / mV,
        lw=2,
        alpha=0.5,
        color='teal'
    )
    plt.axhline(pop.threshold_v / mV, lw=2, color="k", ls="--")
    plt.axhline(pop.reset_v / mV, lw=2, color="k")
    plt.title(f'Voltage trace for {pop_name} neuron {idx}')
    plt.savefig('./figs/pag_vtrace.png')
    # to be completed

'''
The following plotting functions are implemented by Dario and are unchanged.
'''
def plot_single_pop(pop, expt_params, plot_expt_inputs_from = 'rsp'):
    pop_name = pop.neurons.name
    f, ax = plt.subplots(figsize=(12, 9))
    #ax_currents = ax_voltages.twinx()
    f._save_name = "synapse_fitting_" + plot_expt_inputs_from + '2' + pop_name
    f.suptitle("synapse_fitting_" + plot_expt_inputs_from + '2' + pop_name)
    clean_axes(f)

    color = (teal_light, salmon, indigo)[['RSP', 'VGLUT', 'VGAT'].index(pop_name)]

    if pop.clamp == 'VC' and 'inputs_rsp_monitor' in dir(pop): #because rsp co-opted to do vgat2vglut synapse
        sim_traces = pop.inputs_rsp_monitor.I_rsp[::plot_every_n_neurons, :] /pA
        ylabel = plot_expt_inputs_from + ' synaptic currents in ' + pop_name + ' (pA)'
    else: 
        sim_traces = pop.voltages.v[::plot_every_n_neurons, :] /mV  
        ylabel = pop_name + ' membrane potential (mV)'

    for i in range(sim_traces.shape[0]):
        ax.plot(
            pop.voltages.t / ms,
            sim_traces[i, :],
            lw=2,
            alpha=0.5,
            color=color,
        )

    #plot mean simulated trace 
    mean_sim = np.mean(sim_traces, axis =0)

    ax.plot(
        pop.voltages.t / ms,
        mean_sim,
        lw=2,
        alpha=0.5,
        color='k',
        label = 'mean simulated trace'
    )
    
    #plot extpal trace
    if plot_expt_inputs_from != '':    plot_experimental_traces(ax, pop, expt_params, plot_expt_inputs_from = plot_expt_inputs_from)  

    

    ax.set(
        xlim=[0, pop.voltages.t[-1] / ms], ylabel=ylabel,
    )

    ax.legend(loc=1)

def plot_experimental_traces(ax, pop, expt_params, plot_expt_inputs_from = 'rsp'):
    pop_name = pop.neurons.name
    color = (teal_light, salmon, indigo)[['RSP', 'VGLUT', 'VGAT'].index(pop_name)]

    dt = 0.04

    if plot_expt_inputs_from == 'rsp':
        if pop_name != 'RSP':
            target_trace = np.load(r"{dropbox_folder}/2019_patching/1910_patching/avg_{cell_type}_train.npy".format(dropbox_folder=dropbox_folder, cell_type=pop_name.lower()))
            time_shift = 11.06-0.1

    elif plot_expt_inputs_from == 'vgat':
        if pop_name == 'VGLUT':
            target_trace = np.load(r"{dropbox_folder}/rsc_dual/curated/all_rspVgat2vglut_traces_train_avg.npy".format(dropbox_folder=dropbox_folder))
            time_shift = 11.86     
        if pop_name == 'VGAT':
            target_trace = np.load(r"{dropbox_folder}/vgat2vgat/vgat2vgat_pulse.npy".format(dropbox_folder=dropbox_folder))
            time_shift = 11.86 + 3.7      

    elif plot_expt_inputs_from == 'vglut':
        if pop_name == 'VGLUT':
            target_trace = np.load(r"{dropbox_folder}/SC-SC/vglut2vglut_IC_train_avg.npy".format(dropbox_folder=dropbox_folder))
            time_shift = 11.86 - 2      
        if pop_name == 'VGAT':
            target_trace = np.load(r"{dropbox_folder}/sSCvglut_dSC/vglut-/vglut2vgat_train_VC.npy".format(dropbox_folder=dropbox_folder))
            target_trace2 = np.load(r"{dropbox_folder}/sSCvglut_dSC/vglut-/vglut2vgat_pulse_VC.npy".format(dropbox_folder=dropbox_folder))
            relative_time_shift = 2.3 -2.05 - 0.25
            target_trace2 = target_trace2[int(relative_time_shift/dt):]
            target_trace2 = np.pad(target_trace2, (0, len(target_trace) - len(target_trace2)), constant_values = np.nan)
            #target_trace2 = target_trace2/ np.nanmax(target_trace2[int(60/dt):int(70/dt)]) * np.nanmax(target_trace[int(60/dt):int(70/dt)])
            target_trace = np.stack((target_trace, target_trace2)).T

            time_shift = 13.6    

    if 'target_trace' in locals():
        if not pop.clamp == 'VC': target_trace += pop.E_leak / mV #offset exptal trace to resting Vm, except for vgat2vlut which only has VC
        
        t = np.linspace(0, len(target_trace)*dt, len(target_trace)) - time_shift + expt_params['stim_start_t']*1000 
        ax.plot(t,target_trace,                 
            lw=2,
            ls =':',
            alpha=1,
            # color='k',
            label = 'expt trace')   

def simulation_report(elapsed, fraction, _, total):
    sym_time = total * fraction
    logger.debug(
        f"Completed {fraction*100:.2f}% ({sym_time:.2f}/{total:.2f} s) | in {elapsed:.2f} seconds |"
    )

def get_synapses_connectivity_matrix(synapses):
    N = synapses._n_neurons[0]

    try:
        n = len(synapses.w)
    except AttributeError:
        weights = np.full(synapses._n_neurons, np.nan)
        weights[synapses.i, synapses.j] = 1
    else:
        if N ** 2 == n:
            weights = np.array(synapses.w).reshape(N, N)
        else:
            weights = np.full(synapses._n_neurons, np.nan)
            weights[synapses.i, synapses.j] = synapses.w

    return weights



def plot_connectivity_matrices(synapses, ax, scale=2, every=1):
    weights = get_synapses_connectivity_matrix(synapses)

    img = ax.imshow(
        weights, origin="lower", interpolation=None, vmin=0, cmap="bwr"
    )
    ax.set(title=synapses.name, xlabel="target neuron", ylabel="source neuron")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.figure.colorbar(img, cax=cax)
