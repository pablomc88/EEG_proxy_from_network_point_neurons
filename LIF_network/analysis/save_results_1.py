# -*- coding: utf-8 -*-

###############################################################################
## Simulate network_1 and save results to file.                              ##
##                                                                           ##
## Author: Pablo Martinez Canada (pablo.martinez@iit.it)                     ##
## Date: 15/02/2021                                                          ##
###############################################################################


import nest
import numpy as np
import pandas as pd
import scipy.stats
import sys,os
import matplotlib.pyplot as plt
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import network_1
import tools

# Remove stored data of recorders
def cleandir():
    dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'results'))
    for f in os.listdir(dir):
        if 'dat' in f or 'gdf' in f:
            os.remove(os.path.join(dir, f))

#! ===========
#! Parameters
#! ===========

size_factor = 1.0 # downscale the size of the network
weight_factor = 1.0 # increase the synaptic weights to compensate it

Network_params = {
    "N_exc": int(4000.0/size_factor),
    "N_inh": int(1000.0/size_factor),
    "P": 0.2,
    "extent": 1.0,
    "exc_exc_recurrent": 0.178*weight_factor,
    "exc_inh_recurrent": 0.233*weight_factor,
    "inh_inh_recurrent": -2.70*weight_factor,
    "inh_exc_recurrent": -2.01*weight_factor,
    "th_exc_external": 0.234*weight_factor,
    "th_inh_external": 0.317*weight_factor,
    "cc_exc_external": 0.187*weight_factor,
    "cc_inh_external": 0.254*weight_factor
}

excitatory_cell_params  = {
    "V_th": -52.0 ,
    "V_reset": -59.0 ,
    "t_ref": 2.0 ,
    "g_L": 25.0 ,
    "C_m": 500.0 ,
    "E_ex": 0.0 ,
    "E_in": -80.0 ,
    "E_L": -70.0 ,
    "tau_rise_AMPA": 0.4 ,
    "tau_decay_AMPA": 2.0 ,
    "tau_rise_GABA_A": 0.25 ,
    "tau_decay_GABA_A": 5.0 ,
    "tau_m": 20.0,
    "I_e": 0.0
}

inhibitory_cell_params  = {
    "V_th": -52.0 ,
    "V_reset": -59.0 ,
    "t_ref": 1.0 ,
    "g_L": 20.0 ,
    "C_m": 200.0 ,
    "E_ex": 0.0 ,
    "E_in": -80.0 ,
    "E_L": -70.0 ,
    "tau_rise_AMPA": 0.2 ,
    "tau_decay_AMPA": 1.0 ,
    "tau_rise_GABA_A": 0.25 ,
    "tau_decay_GABA_A": 5.0 ,
    "tau_m": 10.0,
    "I_e": 0.0
}

Neuron_params = [excitatory_cell_params,inhibitory_cell_params]

Simulation_params = {
    "experiment_id": "test",
    "simtime": 1000.0,
    "simstep": 1.0,
    "siminterval": 1000.0,
    "trials": 1,
    "num_threads": 8,
    "toMemory" : True,
    "decimate": False,
    "computeMP": False
}

External_input_params = {
    "v_0": 1.5,
    "A_ext": 0.0 ,
    "f_ext": 0.0 ,
    "OU_sigma": 0.4 ,
    "OU_tau": 16.0
}

Analysis_params = {
    "To_be_measured": ['V_m','g_ex','g_in'],
    "Compute_PSP": False,
    "PSP_number_cells": 0,
    "cells_to_analyze": [i for i in range(10)]
}

#! ===========
#! Simulation
#! ===========

# External input rates
# ext_rates = np.arange(1.5,10.5,0.5) # (spikes/s)
ext_rates = [1.5]

# create directory for output
if not os.path.isdir('../results'):
    os.mkdir('../results')

if not os.path.isdir('../results/'+Simulation_params["experiment_id"]):
    os.mkdir('../results/'+Simulation_params["experiment_id"])

for v0 in ext_rates:
    print("Ext. rate = %s sp/s" % v0)
    External_input_params["v_0"] = v0

    for trial in range(Simulation_params["trials"]):
        print("Trial = %s" % trial)

        # Membrane potentials
        if Simulation_params["computeMP"]:
            potentials = [ [np.array([]) for k in range(Network_params["N_exc"])],
                           [np.array([]) for k in range(Network_params["N_inh"])] ]

        # AMPA and GABA currents of exc. cells
        AMPA_current = np.array([])
        GABA_current = np.array([])

        # LFPs
        LFP = [np.array([]) for k in range(7)]

        # PSTHs of exc. cells
        PSTH_bin_size = 1.0 # ms
        all_PSTHs_exc = np.zeros((Network_params["N_exc"],int(Simulation_params["simtime"]/PSTH_bin_size)))
        all_PSTHs_inh = np.zeros((Network_params["N_inh"],int(Simulation_params["simtime"]/PSTH_bin_size)))

        # Spike times
        all_spikes = [np.array([]) for k in range(3*(Network_params["N_exc"]+Network_params["N_inh"]))]

        # Time array
        t_sim = np.array([])

        # To save data
        filename = 'trial_'+str(trial)+'_rate_'+str(v0)

        # Create the network
        net = network_1.network(Network_params,Neuron_params,Simulation_params,
            External_input_params, Analysis_params)
        net.create_network()

        # Simulate the network in intervals of duration "siminterval"
        for time_interval in np.arange(0,Simulation_params["simtime"],Simulation_params["siminterval"]):
            # Clean dir for every time interval
            cleandir()

            # Simulate
            [simtime,data_v,data_s,senders_v,senders_s,pop_ex,pop_in,pop_thalamo,pop_cc,
            pop_parrot_th,pop_parrot_cc, mult_exc,mult_inh] = net.simulate_network(Simulation_params["siminterval"])

            # Load data from files
            print("Loading results...")
            if Simulation_params["toMemory"] == False:
                [data_v,data_s,senders_v,senders_s] = tools.loadRec(
                                                Analysis_params["To_be_measured"],
                                                Simulation_params["num_threads"])

            # Membrane potentials
            if trial==0 and Simulation_params["computeMP"]:
                print("\nComputing MPs...")
                for k,i in enumerate(pop_ex):
                    pos_exc = np.where(senders_v[0]==i)
                    potentials[0][k] = np.concatenate((potentials[0][k],(data_v[0][0]['V_m'])[pos_exc]))

                for k,i in enumerate(pop_in):
                    pos_inh = np.where(senders_v[1]==i)
                    potentials[1][k] = np.concatenate((potentials[1][k],(data_v[1][0]['V_m'])[pos_inh]))

            # AMPA and GABA currents of exc. cells
            print("\nComputing AMPA and GABA currents...")

            # Save data_v to Pandas DataFrame
            data_v_pd = pd.DataFrame(data_v[0][0])
            data_v_pd.set_index(senders_v[0], inplace=True)

            # Sort by time and index
            data_v_pd['index'] = data_v_pd.index
            data_v_pd.sort_values(['index', 'times'],inplace=True)

            # Create arrays
            first_trace = (data_v[0][0]['V_m'])[np.where(senders_v[0]==pop_ex[0])[0]]
            AMPA_trace = np.zeros(len(first_trace))
            GABA_trace = np.zeros(len(first_trace))

            for k,i in enumerate(pop_ex):
                sys.stdout.write("\r" + "Processing cell %s " % k)
                sys.stdout.flush()
                # Faster way of indexing
                rr = slice(k*len(first_trace),(k+1)*len(first_trace),1)

                AMPA_trace += -(data_v_pd[rr]['g_ex'].values) *\
                ((data_v_pd[rr]['V_m'].values) - excitatory_cell_params["E_ex"])
                GABA_trace += -(data_v_pd[rr]['g_in'].values) *\
                ((data_v_pd[rr]['V_m'].values) - excitatory_cell_params["E_in"])

                # # Old approach
                # sender_pos = np.where(senders_v[0]==i)[0]
                #
                # AMPA_trace += -(data_v[0][0]['g_ex'])[sender_pos] *\
                # ((data_v[0][0]['V_m'])[sender_pos] - excitatory_cell_params["E_ex"])
                # GABA_trace += -(data_v[0][0]['g_in'])[sender_pos] *\
                # ((data_v[0][0]['V_m'])[sender_pos] - excitatory_cell_params["E_in"])

            AMPA_current = np.concatenate((AMPA_current,AMPA_trace))
            GABA_current = np.concatenate((GABA_current,GABA_trace))

            # Firing rate of exc. cells
            print("\nComputing Firing rates and LFPs...")
            sel_cells_exc = [pop_ex[i] for i in Analysis_params["cells_to_analyze"]]
            sel_cells_inh = [pop_in[i] for i in Analysis_params["cells_to_analyze"]]

            [PSTH_exc,selected_PSTHs_exc,avg_PSTH_exc,mean_FR_exc,
            avg_ISI_exc] = tools.PSTH(Simulation_params["simtime"],data_s[0],senders_s[0],
                                      pop_ex,sel_cells_exc,PSTH_bin_size)

            [PSTH_inh,selected_PSTHs_inh,avg_PSTH_inh,mean_FR_inh,
            avg_ISI_inh] = tools.PSTH(Simulation_params["simtime"],data_s[1],senders_s[1],
                                    pop_in,sel_cells_inh,PSTH_bin_size)

            for k in range(len(PSTH_exc)):
                all_PSTHs_exc[k] += PSTH_exc[k]

            for k in range(len(PSTH_inh)):
                all_PSTHs_inh[k] += PSTH_inh[k]

            # Select the current time interval since the PSTH is computed
            # for the whole simulation time
            PSTH_range = [int(time_interval/PSTH_bin_size),int((time_interval+\
                          Simulation_params["siminterval"])/PSTH_bin_size)]

            new_PSTH = []
            for k in range(len(PSTH_exc)):
                new_PSTH.append(PSTH_exc[k][PSTH_range[0]:PSTH_range[1]])

            # LFP
            LFP_trace = tools.LFP(Simulation_params["siminterval"],Simulation_params["simstep"],
                    data_v,senders_v,new_PSTH,pop_ex,excitatory_cell_params["E_ex"],
                    excitatory_cell_params["E_in"],excitatory_cell_params["g_L"],
                    PSTH_bin_size,data_v_pd)

            # # Old approach
            # LFP_trace = tools.LFP(Simulation_params["siminterval"],Simulation_params["simstep"],
            #         data_v,senders_v,new_PSTH,pop_ex,excitatory_cell_params["E_ex"],
            #         excitatory_cell_params["E_in"],excitatory_cell_params["g_L"],
            #         PSTH_bin_size)

            for k in range(7):
                LFP[k] = np.concatenate((LFP[k],LFP_trace[k]))

            # Time array
            t_sim = np.concatenate((t_sim,
                    (data_v[0][0]['times'])[np.where(senders_v[0]==pop_ex[0])]))

            # Collect spike times produced by all cells in all populations. Spikes
            # of exc. cells will be located at [0 ... N_exc-1], spikes of inh. cells
            # at [N_exc ... (N_exc + N_inh - 1)], spikes from the external inputs at
            # [N_exc + N_inh) ... (N_exc + N_inh + N_th + N_cc - 1)]
            print("\nComputing spike times...")

            k = 0
            for cell in pop_ex:
                sender_positions = np.where(senders_s[0]==cell)
                spike_times = (data_s[0][0]['times'])[sender_positions[0]]
                all_spikes[k]= np.concatenate((all_spikes[k],spike_times))
                k+=1

            for cell in pop_in:
                sender_positions = np.where(senders_s[1]==cell)
                spike_times = (data_s[1][0]['times'])[sender_positions[0]]
                all_spikes[k]= np.concatenate((all_spikes[k],spike_times))
                k+=1

            for cell in pop_parrot_th:
                sender_positions = np.where(senders_s[2]==cell)
                spike_times = (data_s[2][0]['times'])[sender_positions[0]]
                all_spikes[k]= np.concatenate((all_spikes[k],spike_times))
                k+=1

            for cell in pop_parrot_cc:
                sender_positions = np.where(senders_s[3]==cell)
                spike_times = (data_s[3][0]['times'])[sender_positions[0]]
                all_spikes[k]= np.concatenate((all_spikes[k],spike_times))
                k+=1

            # Create the connection matrix. Each line 'k' of the file has the
            # IDs of the presynaptic connections to the cell 'k'.
            if time_interval == 0:
                connection_matrix = [[] for k in range(len(pop_ex)+len(pop_in))]
                print("Computing connection matrix...")

                # Search for recurrent connections
                for n in pop_ex+pop_in:
                    conns = nest.GetConnections(target = [n])
                    st = nest.GetStatus(conns)

                    if n in pop_ex:
                        ind = pop_ex.index(n)
                    else:
                        ind = pop_in.index(n)+len(pop_ex)

                    for st_con in st:
                        if st_con['source'] in pop_ex:
                            connection_matrix[ind].append(pop_ex.index(st_con['source']))
                        if st_con['source'] in pop_in:
                            connection_matrix[ind].append(pop_in.index(st_con['source'])+\
                            len(pop_ex))

                # Add connections from external inputs
                for j in range(len(pop_parrot_th)):
                    connection_matrix[j].append(j+len(pop_ex)+len(pop_in))
                    connection_matrix[j].append(j+len(pop_ex)+len(pop_in)+len(pop_parrot_th))

        print("Saving data...")
        # Save spikes
        tools.saveData(Simulation_params["experiment_id"],filename,".spikes",all_spikes)

        # Save connection matrix
        tools.saveData(Simulation_params["experiment_id"],filename,".connections",connection_matrix)

        # Save membrane potentials
        if Simulation_params["computeMP"]:
            tools.saveData(Simulation_params["experiment_id"],filename,".MP_exc",potentials[0])
            tools.saveData(Simulation_params["experiment_id"],filename,".MP_inh",potentials[1])

        # Save AMPA and GABA currents
        if Simulation_params["decimate"]:
            tools.saveData(Simulation_params["experiment_id"],filename,".AMPA",
            tools.decimate(AMPA_current,10))

            tools.saveData(Simulation_params["experiment_id"],filename,".GABA",
            tools.decimate(GABA_current,10))
        else:
            tools.saveData(Simulation_params["experiment_id"],filename,".AMPA",
            AMPA_current)

            tools.saveData(Simulation_params["experiment_id"],filename,".GABA",
            GABA_current)

        # Normalize LFPs
        # Discard first 500 ms for comp. mean and s. deviation
        start_time_pos = int(500.0/Simulation_params["simstep"])
        for k in range(7):
            LFP[k] = (LFP[k] - np.mean(LFP[k][start_time_pos:])) / np.std(LFP[k][start_time_pos:])

        # Save LFPs
        if Simulation_params["decimate"]:
            tools.saveData(Simulation_params["experiment_id"],filename,".LFP",
            tools.decimate(LFP,10))
        else:
            tools.saveData(Simulation_params["experiment_id"],filename,".LFP",
            LFP)

        # Save time array
        tools.saveData(Simulation_params["experiment_id"],filename,".times",[t_sim])

        # Save time step
        tools.saveData(Simulation_params["experiment_id"],filename,".dt",Simulation_params["simstep"])

        # Analysis
        print("Analysis of results...")

        # Pairwise correlation of exc. cells
        number_of_samples = 1000
        bin_size = 2.0 # ms
        binned_spikes = []
        sel_pop = np.random.randint(len(pop_ex),size = number_of_samples)
        for k in sel_pop:
            binned_spikes.append(np.histogram(all_spikes[k],bins=int(Simulation_params["simtime"]/bin_size),
                                range=[0.,Simulation_params["simtime"]])[0])
        cc_exc, hh, bb = tools.pairwiseCorrelation(binned_spikes)

        print("Mean pairwise correlation of exc. cells = %s"%cc_exc)

        # Pairwise correlation of inh. cells
        binned_spikes = []
        sel_pop = np.random.randint(len(pop_in),size = number_of_samples)
        for k in sel_pop:
            binned_spikes.append(np.histogram(all_spikes[k+len(pop_ex)],bins=int(Simulation_params["simtime"]/bin_size),
                                range=[0.,Simulation_params["simtime"]])[0])
        cc_inh, hh, bb = tools.pairwiseCorrelation(binned_spikes)

        print("Mean pairwise correlation of inh. cells = %s"%cc_inh)

        # Coeff. of variation of the ISI of exc. cells
        ISI = np.array([])
        for spike_times in all_spikes[0:len(pop_ex)]:
            d = np.diff(spike_times)
            if len(d) > 2: # at least 3 spikes
                ISI=np.concatenate((ISI,d))

        CV_ISI_exc = scipy.stats.variation(ISI)
        print("Coeff. of variation of ISI of exc. cells = %s" % CV_ISI_exc)

        # Coeff. of variation of the ISI of inh. cells
        ISI = np.array([])
        for spike_times in all_spikes[len(pop_ex):len(pop_ex)+len(pop_in)]:
            d = np.diff(spike_times)
            if len(d) > 2: # at least 3 spikes
                ISI=np.concatenate((ISI,d))

        CV_ISI_inh = scipy.stats.variation(ISI)
        print("Coeff. of variation of ISI of inh. cells = %s" % CV_ISI_inh)

        # Mean firing rate of exc. cells
        avg_PSTH= np.sum(all_PSTHs_exc,axis = 0)/len(pop_ex)
        Mean_FR_exc = np.mean(avg_PSTH[int(500.0/PSTH_bin_size):])
        print("Mean firing rate of exc. cells = %s" % Mean_FR_exc)

        # Mean firing rate of inh. cells
        avg_PSTH= np.sum(all_PSTHs_inh,axis = 0)/len(pop_in)
        Mean_FR_inh = np.mean(avg_PSTH[int(500.0/PSTH_bin_size):])
        print("Mean firing rate of inh. cells = %s" % Mean_FR_inh)

        # Save results of the analysis of exc. cells
        try:
            analysis = pickle.load(open('../results/'+Simulation_params["experiment_id"]+\
                                        '/'+filename+".analysis", "rb"))
            analysis.append([cc_exc,CV_ISI_exc,Mean_FR_exc])
            pickle.dump(analysis, open('../results/'+Simulation_params["experiment_id"]+'/'+\
                                         filename+".analysis", "wb"))
        except (OSError, IOError) as e:
            pickle.dump([ [cc_exc,CV_ISI_exc,Mean_FR_exc] ], open('../results/'+Simulation_params["experiment_id"]+\
                                        '/'+filename+".analysis", "wb"))

        # Save results of the analysis of inh. cells
        try:
            analysis = pickle.load(open('../results/'+Simulation_params["experiment_id"]+\
                                        '/'+filename+".analysis_IN", "rb"))
            analysis.append([cc_inh,CV_ISI_inh,Mean_FR_inh])
            pickle.dump(analysis, open('../results/'+Simulation_params["experiment_id"]+'/'+\
                                         filename+".analysis_IN", "wb"))
        except (OSError, IOError) as e:
            pickle.dump([ [cc_inh,CV_ISI_inh,Mean_FR_inh] ], open('../results/'+Simulation_params["experiment_id"]+\
                                        '/'+filename+".analysis_IN", "wb"))

        # Raster plots of spikes
        fig = plt.figure(figsize=[8,6], dpi=300)
        Vax = fig.add_axes([0.15,0.15,0.8,0.8],frameon=True)
        for k in range(len(pop_ex) + len(pop_in)):
            if k < len(pop_ex):
                Vax.scatter(all_spikes[k],k*np.ones(len(all_spikes[k])),s=0.2,color='b',label = 'Exc.')
            else:
                Vax.scatter(all_spikes[k],k*np.ones(len(all_spikes[k])),s=0.2,color = 'r',label = 'Inh.')

        Vax.set_xlim([0.0,Simulation_params['simtime']])
        Vax.set_xlabel('Time (ms)')
        Vax.set_ylabel('Cell ID')

        # Dummy plot for the legend
        Vax = fig.add_axes([0.8,0.15,0.1,0.7],frameon=False)
        for k in range(2):
            if k==0:
                Vax.plot([],[],'o',color='b',label = 'Exc.')
            else:
                Vax.plot([],[],'o',color = 'r',label = 'Inh.')

        Vax.set_yticks([])
        Vax.set_xticks([])
        Vax.legend(loc='upper right')

        # # Raster plots of spikes (NEST)
        # nest.raster_plot.from_device(mult_exc,hist=True,title="Exc.")
        # nest.raster_plot.from_device(mult_inh,hist=True,title="Inh.")

        plt.show()
