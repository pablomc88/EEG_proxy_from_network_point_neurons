# -*- coding: utf-8 -*-

###############################################################################
## Simulation of network_1.                                                  ##
##                                                                           ##
## Author: Pablo Martinez Canada (pablo.martinez@iit.it)                     ##
## Date: 15/02/2021                                                          ##
###############################################################################

from mpi4py import MPI
import numpy as np
import os,sys
import LFPy
import neuron
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import network_1

# Initialize the MPI interface
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Electrode positions
# Grid electrode
# X, Z = np.mgrid[0:2250:250, -400:425:25]
# X_electrode = X.flatten()
# Z_electrode = Z.flatten()

# Point electrode
X = []
Z = [np.arange(-400,425,25)]
Z_electrode = Z[0]
X_electrode = np.zeros(len(Z_electrode))

simulation_params = {
   "experiment_id_3D": "test",
   "experiment_id_LIF": "test",
    'tstop' : 1000.0,
    "dt": 1.0,
    "trials": 1,
    "population_sizes": [4000, 1000],
    "xyz_rotations": [dict(x=4.729, y=-3.166, z = 0.0), # Markram
                      dict(x=4.729, y=-3.166, z = 0.0)],
    # "xyz_rotations": [dict(x=np.pi/2.0, y=np.pi, z = 0.0), # Allen
    #                 dict(x=4.729, y=-3.166, z = 0.0)],
    "z_cell_pos": -150.0,
    "weight_factor": 1.0,
    "radius": 500.0,
    "AMPA_syn_position": [-10**6,10**6],
    "GABA_syn_position": [-10**6,0],
    "Z_size":np.array(Z).size,
    "Z_electrode": Z_electrode,
    "X_electrode": X_electrode,
    "individual_EEG": True,
    "decimate": False,
    "record_all": False
}

cell_params_ex = {
    # 'morphology' : 'Cux2-CreERT2_Ai14-207761.04.02.01_506798042_m.asc', # Allen I.
    # 'morphology' : 'dend-C250500A-P3_axon-C260897C-P2_-_Clone_9.asc', # Markram 9
    'morphology' : 'dend-C260897C-P3_axon-C220797A-P3_-_Clone_0.asc', # Markram 0
    # 'template': 'Neocortical_Microcircuitry_Template_Allen', # Allen I.
    'template': 'Neocortical_Microcircuitry_Template',
    'nsegs_method': 'none',
    # 'nsegs_method': 'lambda_f', # Allen I.
    'passive_parameters':{'g_pas' : 1./30000.,'e_pas' : -70.0},
    'dt' : simulation_params['dt'],
    'tstart' : 0.0,
    'tstop' : simulation_params['tstop'],
    'v_init' : -70.,
}

cell_params_in = {
    'morphology' : 'C250500A-I4_-_Scale_x1.000_y1.025_z1.000_-_Clone_0.asc',
    'template': 'Neocortical_Microcircuitry_Template',
    'nsegs_method': 'none',
    'passive_parameters':{'g_pas' : 1./20000.,'e_pas' : -70.0},
    'dt' : simulation_params['dt'],
    'tstart' : 0.0,
    'tstop' : simulation_params['tstop'],
    'v_init' : -70.0
}

cell_params = [cell_params_ex,cell_params_in]

#! ===========
#! Simulation
#! ===========

# External input rates
# ext_rates = np.arange(1.5,30.5,1.0) # (spikes/s)
ext_rates = [1.5] # (spikes/s)

# create directory for output
if RANK==0:
    if not os.path.isdir('../results/'):
        os.mkdir('../results/')

    if not os.path.isdir('../results/'+simulation_params["experiment_id_3D"]):
        os.mkdir('../results/'+simulation_params["experiment_id_3D"])
    # # Remove stored data otherwise
    # else:
    #     dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'results'))
    #     for f in os.listdir(dir):
    #         os.remove(os.path.join(dir, f))

# Resync MPI threads
COMM.Barrier()

# Simulate for each external rate and each trial
for v0 in ext_rates:
    if RANK==0:
        print("Ext. rate = %s" % v0)

    for trial in range(simulation_params["trials"]):
        if RANK==0:
            print("Trial = %s" % trial)

        # Simulation
        net = network_1.network(simulation_params,cell_params)
        filename = 'trial_'+str(trial)+'_rate_'+str(v0)

        net.simulate_network(filename,COMM,SIZE,RANK)

        # Resync MPI threads
        COMM.Barrier()

        # Remove presynaptic spikes to release hard disk space
        # if RANK==0:
        #     os.remove('../../LIF_network/results/'+\
        #               simulation_params["experiment_id_LIF"]+"/"+filename+".spikes")

        # Save simulation params to file
        if RANK==0:
            results_dict = {
                "X": X,
                "X_electrode": X_electrode,
                "Z": Z,
                "Z_electrode": Z_electrode,
                "simulation_params": simulation_params,
                "cell_params_ex": cell_params_ex,
                "cell_params_in": cell_params_in
            }

            pickle.dump(results_dict,
                        open('../results/'+simulation_params["experiment_id_3D"]+\
                             "/"+filename+"_simulation_params","wb"))
