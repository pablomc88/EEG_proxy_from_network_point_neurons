# -*- coding: utf-8 -*-

###############################################################################
## Computation of the average coefficient of determination (R^2) on selected ##
## datasets.                                                                 ##
## Author: Pablo Martinez Canada (pablo.martinez@iit.it)                     ##
## Date: 05/02/2021                                                          ##
###############################################################################

import matplotlib.pylab as plt
from matplotlib import rcParams
import numpy as np
import os,sys
import pickle
import tools

# Properties of plots
DPI = 300
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 10

# Firing rates of the external input (v0)
# fr = np.arange(1.5,30.5,0.5)
fr = [1.5]

# IDs of the datasets
datasets_LIF = ["test"]
datasets_3D = ["test"]

# Spatially extended network
compute_local_proxy = False # Compute local proxies (True) or global proxy (False)
subnetwork = 1 # Select S1 (0), S2 (1), S3 (2) or S4 (3)

# Total number of datasets
number_datasets = len(datasets_LIF)

# Create arrays with all combinations
ext_rates = np.array([])
experiment_id_LIF = np.array([])
experiment_id_3D = np.array([])
trials = np.array([],dtype=int)

for k in range(number_datasets):
    ext_rates = np.concatenate((ext_rates,fr))
    experiment_id_LIF = np.concatenate((experiment_id_LIF,[datasets_LIF[k] for i in range(len(fr))]))
    experiment_id_3D = np.concatenate((experiment_id_3D,[datasets_3D[k] for i in range(len(fr))]))
    trials = np.concatenate((trials,[1 for i in range(len(fr))])) # We set 1 trial by default

# Parameters of the ERWS-1 and ERWS-2 proxies
# Causal
# ERWS_1_params = [0. , 3.1, 0.1]
# ERWS_2_params = [0. ,  0. ,  0. , -1.5,  0.2,  4. ,  0.5,  0.5,  0. ]
# Non-causal
ERWS_1_params = [-0.9,  2.3,  0.3]
ERWS_2_params = [-0.6,  0.1, -0.4, -1.9,  0.6,  3.0 ,  1.4,  1.7,  0.2]

# R2 of the proxies
R2 = []

# To plot time traces of proxies vs EEGs
fig1 = plt.figure(figsize=[7,6], dpi=DPI)
row = 0
col = 0

# To plot R^2
fig2 = plt.figure(figsize=[4,3], dpi=DPI)

# Loop through datasets
for j,(exp_id_LIF,exp_id_3D,v0,n_trials) in enumerate(zip(experiment_id_LIF,experiment_id_3D,ext_rates,trials)):
    print("Exp. LIF: %s, Exp. 3D: %s, v0 = %s, trials = %s" % (exp_id_LIF,exp_id_3D,v0,n_trials))

    # Loop through number of trials
    for trial in range(n_trials):
        print("Trial = %s" % trial)

        # Load data
        filename = 'trial_'+str(trial)+'_rate_'+str(v0)
        try:
            with open( '../multicompartment_network/results/'+exp_id_3D+"/"+filename+"_simulation_params","rb") as f:
                # Load simulation parameters of the 3D network
                sim_data_3D = pickle.load(open( '../multicompartment_network/results/'+exp_id_3D+"/"+\
                                                filename+"_simulation_params","rb") ,encoding='latin1' )
                X = np.array(sim_data_3D["X"])
                X_electrode = np.array(sim_data_3D["X_electrode"])
                Z = np.array(sim_data_3D["Z"])
                Z_electrode = np.array(sim_data_3D["Z_electrode"])
                simulation_params = sim_data_3D["simulation_params"]
                cell_params_ex = sim_data_3D["cell_params_ex"]
                cell_params_in = sim_data_3D["cell_params_in"]

                # Load simulation results of the 3D network
                [summed_dipole,summed_EEG_top,tvec] = tools.loadResults(exp_id_3D,filename,
                simulation_params["population_sizes"],Z,simulation_params["tstop"],
                simulation_params["dt"],simulation_params["individual_EEG"])

                # Load simulation results of the LIF network
                if compute_local_proxy:
                    AMPA_LIF = tools.loadData(exp_id_LIF,filename,'.AMPA'+"_sub_"+str(subnetwork))
                    GABA_LIF = tools.loadData(exp_id_LIF,filename,'.GABA'+"_sub_"+str(subnetwork))
                    LFP_LIF = tools.loadData(exp_id_LIF,filename,'.LFP'+"_sub_"+str(subnetwork))
                else:
                    AMPA_LIF = tools.loadData(exp_id_LIF,filename,'.AMPA')
                    GABA_LIF = tools.loadData(exp_id_LIF,filename,'.GABA')
                    LFP_LIF = tools.loadData(exp_id_LIF,filename,'.LFP')
                times_LIF = tools.loadData(exp_id_LIF,filename,'.times')
                dt_LIF = tools.loadData(exp_id_LIF,filename,'.dt')

                # Startup time to analyze results at 100 ms
                start_time_pos_3D = int(np.where(tvec >= 100.)[0][0]) + 1 # Add 1 because Neuron sim. starts at 0 ms
                                                                          # and Nest sim. starts at dt ms
                start_time_pos_LIF = int(np.where(times_LIF[0] >= 100.)[0][0])

                # The 3D signal is the EEG signal, not the LFP
                new_Z = [[0]]
                z_pos = 0
                summed_signal = np.array([summed_EEG_top])
                baseline = -1

                # ERWS-1 and ERWS-2 proxies
                [ERWS_1,R2_ERWS_1,R2_ERWS_1_mean,R2_ERWS_1_max,best_parameters_ERWS_1,
                avg_parameters_ERWS_1] = tools.weightedSum(new_Z, AMPA_LIF,GABA_LIF,summed_signal,
                                                          start_time_pos_3D,start_time_pos_LIF,
                                                          dt_LIF,simulation_params["dt"],baseline,False,
                                                          ERWS_1_params)

                [ERWS_2,R2_ERWS_2,R2_ERWS_2_mean,R2_ERWS_2_max,best_parameters_ERWS_2,
                avg_parameters_ERWS_2] = tools.weightedSum(new_Z, AMPA_LIF,GABA_LIF,summed_signal,
                                                          start_time_pos_3D,start_time_pos_LIF,
                                                          dt_LIF,simulation_params["dt"],baseline,False,
                                                          ERWS_2_params,v0)

                # Compute the other proxies
                delays,R2_LIF,R2_LIF_mean,R2_LIF_max,cc,data_LIF_norm,data_3D_norm = tools.getMetrics(
                                                new_Z,LFP_LIF,summed_signal,start_time_pos_3D,
                                                start_time_pos_LIF,simulation_params["dt"],baseline,False)

                # Add performance of ERWS-1 and ERWS-2 to performance of the other proxies
                R2_LIF_mean = np.append(R2_LIF_mean,R2_ERWS_1_mean)
                R2_LIF_mean = np.append(R2_LIF_mean,R2_ERWS_2_mean)
                R2_LIF_max = np.append(R2_LIF_max,R2_ERWS_1_max)
                R2_LIF_max = np.append(R2_LIF_max,R2_ERWS_2_max)

                # Collect performance for each sample
                R2.append(R2_LIF_mean)

                # Plot ERWS-1 and ERWS-2 vs EEG
                start_time_pos = start_time_pos_LIF
                end_time_pos = int(np.where(times_LIF[0] >= 800.)[0][0])

                if(row < 3 and col < 3 and np.random.rand(1) > 0.05):
                    Vax = fig1.add_axes([0.1+col*0.3,0.1+row*0.3,0.25,0.25],frameon=True)
                    Vax.plot(times_LIF[0][start_time_pos:end_time_pos],
                            ERWS_1[0][0:end_time_pos-start_time_pos],
                            'b',label = 'ERWS1')
                    Vax.plot(times_LIF[0][start_time_pos:end_time_pos],
                            ERWS_2[0][0:end_time_pos-start_time_pos],
                            'r',label = 'ERWS2')
                    Vax.plot(times_LIF[0][start_time_pos:end_time_pos],
                            data_3D_norm[0][0][0:end_time_pos-start_time_pos],
                            '--k',label = 'EEG')

                    if col == 0 and row == 0:
                        Vax.set_xlabel('Time (ms)')
                        Vax.legend(loc = 'lower right')
                    else:
                        Vax.set_xticks([])

                    if (col < 2):
                        col+=1
                    else:
                        col = 0
                        row+=1

        except Exception as e:
            print(e)

# Print average R^2
R2 = np.array(R2)
print("Mean R^2:")
print(np.mean(R2,axis = 0))

# Plot R^2
labels = ['FR','AMPA','GABA',r'$V_m$',r'$\sum I$',r'$\sum |I|$','LRWS','ERWS1','ERWS2']
colors = ['b','pink','c','orange','g','r','sienna','cornflowerblue','yellowgreen']
Vax = fig2.add_axes([0.15,0.15,0.7,0.75],frameon=True)
Vax.bar(np.arange(len(np.mean(R2,axis = 0))),np.mean(R2,axis = 0),1,color=colors)
Vax.plot([0,9],[0.9,0.9],'k--')
Vax.set_xticks([])
Vax.set_xticklabels([])
Vax.text(-2.0,1.0,r'$R^{2}$')

# Legend
Vax = fig2.add_axes([0.85,0.1,0.15,0.6],frameon=False)
Vax.set_xticks([])
Vax.set_yticks([])
Vax.set_ylim([0,1])
Vax.set_xlim([0,1])
for rr in range(len(np.mean(R2,axis = 0))):
    Vax.plot([],[],color = colors[rr],label = labels[rr])

font_legend = {'fontsize':8}
Vax.legend(loc='center',frameon = False)
Vax.legend(prop={'size':font_legend['fontsize']})

plt.show()
