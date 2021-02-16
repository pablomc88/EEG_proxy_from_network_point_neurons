# -*- coding: utf-8 -*-

###############################################################################
## Hybrid modeling scheme according to [1,2]. This network creates 2         ##
## populations (excitatory and inhibitory) of mutually unconnected           ##
## multicompartment neurons. Spike times generated by the LIF model network  ##
## serve as input spikes for each multicompartment neuron.                   ##
##                                                                           ##
## [1] Mazzoni, A., Lindén, H., Cuntz, H., Lansner, A., Panzeri, S., &       ##
# Einevoll, G. T.(2015). Computing the local field potential (LFP) from      ##
## integrate-and-fire network models. PLoS computational biology, 11(12),    ##
## e1004584.                                                                 ##
##                                                                           ##
## [2] Hagen, E., Dahmen, D., Stavrinou, M. L., Lindén, H., Tetzlaff, T.,    ##
## van Albada,S. J., ... & Einevoll, G. T. (2016). Hybrid scheme for         ##
## modeling local field potentials from point-neuron networks. Cerebral      ##
## Cortex, 1-36.                                                             ##
##                                                                           ##
## Author: Pablo Martinez Canada (pablo.martinez@iit.it)                     ##
## Date: 15/02/2021                                                          ##
###############################################################################

import numpy as np
import os,sys
import LFPy
import neuron
from mpi4py import MPI
import pickle
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '/analysis'))
import tools

class network(object):

    def __init__(self,simulation_params,cell_params):

        #! ===========
        #! Parameters
        #! ===========

        # Cell parameters
        self.cell_params_ex = {
            'morphology'    : os.path.join('..','morphologies',
                                 cell_params[0]['morphology']),
            'templatefile'  : os.path.join('..','morphologies',
                                            cell_params[0]['template']+'.hoc'),
            'templatename'  : cell_params[0]['template'],
            'templateargs'  : '../morphologies/'+cell_params[0]['morphology'],
            'nsegs_method' : cell_params[0]['nsegs_method'],
            # 'lambda_f' : 100.,  # frequency where length constants are computed
            'pt3d' : True,
            'passive' : True,
            'Ra' : 100.,
            'cm' : 1.0,
            'passive_parameters' : {
                            'g_pas' : cell_params[0]['passive_parameters']['g_pas'],
                            'e_pas' : cell_params[0]['passive_parameters']['e_pas']
                            },
            'dt' : cell_params[0]['dt'],
            'tstart' : cell_params[0]['tstart'],
            'tstop' : cell_params[0]['tstop'],
            'v_init' : cell_params[0]['v_init']
        }

        self.cell_params_in = {
            'morphology'    : os.path.join('..','morphologies',
                                 cell_params[1]['morphology']),
            'templatefile'  : os.path.join('..','morphologies',
                                            cell_params[1]['template']+'.hoc'),
            'templatename'  : cell_params[1]['template'],
            'templateargs'  : '../morphologies/'+cell_params[1]['morphology'],
            'nsegs_method' : cell_params[1]['nsegs_method'],
            # 'lambda_f' : 100.,  # frequency where length constants are computed
            'pt3d' : True,
            'passive' : True,
            'Ra' : 100.,
            'cm' : 1.0,
            'passive_parameters' : {
                            'g_pas' : cell_params[1]['passive_parameters']['g_pas'],
                            'e_pas' : cell_params[1]['passive_parameters']['e_pas']
                            },
            'dt' : cell_params[1]['dt'],
            'tstart' : cell_params[1]['tstart'],
            'tstop' : cell_params[1]['tstop'],
            'v_init' : cell_params[1]['v_init']
        }

        # Simulation parameters
        self.simulation_params = {
            "experiment_id_LIF": simulation_params['experiment_id_LIF'],
            "experiment_id_3D": simulation_params['experiment_id_3D'],
            "population_sizes": [simulation_params["population_sizes"][0], # Exc./Inh.
                                simulation_params["population_sizes"][1]],
            "z_cell_pos": simulation_params['z_cell_pos'],
            "xyz_rotations": simulation_params["xyz_rotations"], # cell rotations
            # Increase the synaptic weights when the LIF network is downscaled
            "weight_factor": simulation_params["weight_factor"],
            "radius": simulation_params["radius"], # Cylinder radius (um)
            # Z location of GABA synapses on excitatory cells
            "GABA_syn_position": {'z_min': simulation_params["GABA_syn_position"][0],
                                     'z_max': simulation_params["GABA_syn_position"][1]},
            # Z location of AMPA synapses on excitatory cells
            "AMPA_syn_position": {'z_min': simulation_params["AMPA_syn_position"][0],
                                     'z_max': simulation_params["AMPA_syn_position"][1]},
            # Array of Z locations for the electrode measures
            "Z_electrode": simulation_params["Z_electrode"],
            "Z_size": simulation_params["Z_size"],
            # Array of X locations for the electrode measures
            "X_electrode": simulation_params["X_electrode"],
            # Choose to compute the EEG for each cell or to compute the CDM for
            # each cell, sum them all at the end of simulation and then compute
            # the EEG
            "individual_EEG": simulation_params["individual_EEG"],
            # Downsample the simulation results before saving to file
            "decimate": simulation_params["decimate"],
            # Save membrane potentials, spikes and synapses to file (True/False)
            "record_all" :simulation_params["record_all"]
        }

    #! ===================================
    #! Configure and simulate the network
    #! ===================================

    def simulate_network(self,filename,COMM,SIZE,RANK):

        filename = filename # to load data from the LIF network

        # XY cell positions: randomly placed within a circle
        if RANK == 0:
            x_cell_pos = [[],[]]
            y_cell_pos = [[],[]]

            for cell in range(sum(self.simulation_params["population_sizes"])):
                r = np.random.rand()*self.simulation_params["radius"]
                angle = np.random.rand()*(2*np.pi)
                x = r * np.cos(angle)
                y = r * np.sin(angle)

                if cell < self.simulation_params["population_sizes"][0]:
                    x_cell_pos[0].append(x)
                    y_cell_pos[0].append(y)
                else:
                    x_cell_pos[1].append(x)
                    y_cell_pos[1].append(y)

        else:
            x_cell_pos = None
            y_cell_pos = None

        x_cell_pos = COMM.bcast(x_cell_pos, root=0)
        y_cell_pos = COMM.bcast(y_cell_pos, root=0)

        # Resync MPI threads
        COMM.Barrier()

        # Z positions
        z_cell_pos = [self.simulation_params["z_cell_pos"],
                      self.simulation_params["z_cell_pos"]]

        # XYZ cell rotations
        xyz_rotations = [self.simulation_params["xyz_rotations"][0],
                         self.simulation_params["xyz_rotations"][1]]

        # Synapse parameters
        synapse_parameters = {}

        # Recurrent connections
        synapse_parameters['exc_exc']={
            'e' : 0.0,                   # reversal potential (mV)
            'tau1' : 0.4,                # rise time constant (ms)
            'tau2' : 2.0,                # decay time constant (ms)
            'weight' : self.simulation_params["weight_factor"]*\
                       0.178*10**(-3),   # syn. weight (uS)
            'position_parameters' : {
                'z_min': self.simulation_params["AMPA_syn_position"]["z_min"],
                'z_max': self.simulation_params["AMPA_syn_position"]["z_max"]}
        }

        synapse_parameters['inh_exc']={
            'e' : -80.,
            'tau1' : 0.25,
            'tau2' : 5.0,
            'weight' : self.simulation_params["weight_factor"]*\
                       2.01*10**(-3),
            'position_parameters' : {
                'z_min': self.simulation_params["GABA_syn_position"]["z_min"],
                'z_max': self.simulation_params["GABA_syn_position"]["z_max"]}
        }

        synapse_parameters['exc_inh']={
            'e' : 0.,
            'tau1' : 0.2,
            'tau2' : 1.0,
            'weight' : self.simulation_params["weight_factor"]*\
                       0.233*10**(-3),
            'position_parameters' : {
                'z_min': -10**6,
                'z_max':  10**6}
        }

        synapse_parameters['inh_inh']={
            'e' : -80.,
            'tau1' : 0.25,
            'tau2' : 5.0,
            'weight' : self.simulation_params["weight_factor"]*\
                       2.70*10**(-3),
            'position_parameters' : {
                'z_min': -10**6,
                'z_max':  10**6}
        }

        # External inputs
        synapse_parameters['th_exc']={
            'e' : 0.0,
            'tau1' : 0.4,
            'tau2' : 2.0,
            'weight' : self.simulation_params["weight_factor"]*\
                       0.234*10**(-3),
            'position_parameters' : {
                'z_min': self.simulation_params["AMPA_syn_position"]["z_min"],
                'z_max': self.simulation_params["AMPA_syn_position"]["z_max"]}
        }

        synapse_parameters['th_inh']={
            'e' : 0.0,
            'tau1' : 0.2,
            'tau2' : 1.0,
            'weight' : self.simulation_params["weight_factor"]*\
                       0.317*10**(-3),
            'position_parameters' : {
                'z_min': -10**6,
                'z_max':  10**6}
        }

        synapse_parameters['cc_exc']={
            'e' : 0.0,
            'tau1' : 0.4,
            'tau2' : 2.0,
            'weight' : self.simulation_params["weight_factor"]*\
                       0.187*10**(-3),
            'position_parameters' : {
                'z_min': self.simulation_params["AMPA_syn_position"]["z_min"],
                'z_max': self.simulation_params["AMPA_syn_position"]["z_max"]}
        }

        synapse_parameters['cc_inh']={
            'e' : 0.0,
            'tau1' : 0.2,
            'tau2' : 1.0,
            'weight' : self.simulation_params["weight_factor"]*\
                       0.254*10**(-3),
            'position_parameters' : {
                'z_min': -10**6,
                'z_max':  10**6}
        }

        # Define electrode parameters
        Z = self.simulation_params["Z_electrode"]
        X = self.simulation_params["X_electrode"]

        electrode_parameters = {
            'sigma' : 0.3,      # extracellular conductivity
            'z' : Z,
            'x' : X,
            'y' : np.zeros(Z.size),
        }

        # EEG: four_sphere parameters
        # Dimensions that approximate those of a rat head model
        radii = [9000.,9500.,10000.,10500.]
        sigmas = [0.3, 1.5, 0.015, 0.3]
        rad_tol = 1e-2

        # Summed LFPs
        if RANK==0:
            summed_LFP = np.zeros(  (self.simulation_params["Z_size"],
                                    int(self.cell_params_ex["tstop"] /\
                                    self.cell_params_ex["dt"] + 1)) )

            if self.simulation_params["individual_EEG"]:
                summed_EEG_top = np.zeros(int(self.cell_params_ex["tstop"] /\
                                        self.cell_params_ex["dt"] + 1))
            else:
                summed_dipole = np.zeros((int(self.cell_params_ex["tstop"] /\
                                        self.cell_params_ex["dt"] + 1),3))

        # Load presynaptic spike times and connection matrix of the LIF network
        spike_times = tools.loadLIFData(self.simulation_params["experiment_id_LIF"],
                                           filename,'.spikes')
        connection_matrix = tools.loadLIFData(self.simulation_params["experiment_id_LIF"],
                                           filename,'.connections')

        # Start timer
        if RANK==0:
            start_c = time.time()

        # Iterate over cells in populations
        for j,pop_size in enumerate(self.simulation_params["population_sizes"]):
            for cell_id in range(pop_size):
                sys.stdout.write("\r" + "Simulating cell %s " % (cell_id+\
                                j*self.simulation_params["population_sizes"][0]))
                sys.stdout.flush()

                if cell_id % SIZE == RANK:
                    # Initialize cell instance, using the LFPy.TemplateCell class
                    if j==0:
                        post_cell = LFPy.TemplateCell(**self.cell_params_ex)
                        # Position and rotation of the cell
                        post_cell.set_rotation(**xyz_rotations[0])
                        post_cell.set_pos(x = x_cell_pos[0][cell_id],
                                          y = y_cell_pos[0][cell_id],
                                          z = z_cell_pos[0])
                    else:
                        post_cell = LFPy.TemplateCell(**self.cell_params_in)
                        # Position and rotation of the cell
                        post_cell.set_rotation(**xyz_rotations[1])
                        post_cell.set_pos(x = x_cell_pos[1][cell_id],
                                          y = y_cell_pos[1][cell_id],
                                          z = z_cell_pos[1])

                    # Search for presynaptic connections
                    for conn,pre_cell in enumerate(connection_matrix[cell_id+\
                            j*self.simulation_params["population_sizes"][0]]):
                        # Recurrent: Exc. -> Exc.
                        if j==0 and pre_cell < self.simulation_params["population_sizes"][0]:
                            dict_syn = synapse_parameters['exc_exc']
                        # Recurrent: Exc. -> Inh.
                        elif j==1 and pre_cell < self.simulation_params["population_sizes"][0]:
                            dict_syn = synapse_parameters['exc_inh']
                        # Recurrent: Inh. -> Exc.
                        elif j==0 and pre_cell >= self.simulation_params["population_sizes"][0] and\
                        pre_cell < sum(self.simulation_params["population_sizes"]):
                            dict_syn = synapse_parameters['inh_exc']
                        # Recurrent: Inh. -> Inh.
                        elif j==1 and pre_cell >= self.simulation_params["population_sizes"][0] and\
                        pre_cell < sum(self.simulation_params["population_sizes"]):
                            dict_syn = synapse_parameters['inh_inh']
                        # External: Th. -> Exc.
                        elif j==0 and pre_cell >= sum(self.simulation_params["population_sizes"]) and\
                        pre_cell < 2*sum(self.simulation_params["population_sizes"]):
                            dict_syn = synapse_parameters['th_exc']
                        # External: Th. -> Inh.
                        elif j==1 and pre_cell >= sum(self.simulation_params["population_sizes"]) and\
                        pre_cell < 2*sum(self.simulation_params["population_sizes"]):
                            dict_syn = synapse_parameters['th_inh']
                        # External: CC. -> Exc.
                        elif j==0 and pre_cell >= 2*sum(self.simulation_params["population_sizes"]):
                            dict_syn = synapse_parameters['cc_exc']
                        # External: CC. -> Inh.
                        elif j==1 and pre_cell >= 2*sum(self.simulation_params["population_sizes"]):
                            dict_syn = synapse_parameters['cc_inh']

                        # Segment indices to locate the connection
                        pos = dict_syn["position_parameters"]
                        idx = post_cell.get_rand_idx_area_norm(section=['dend','apic','soma'],nidx=1,**pos)

                        # Make a synapse and set the spike times
                        for i in idx:
                            syn = LFPy.Synapse(cell=post_cell, idx=i, syntype='Exp2Syn',
                                          weight=dict_syn['weight'],delay=1.0,
                                          **dict(tau1=dict_syn['tau1'], tau2=dict_syn['tau2'],
                                           e=dict_syn['e']))
                            syn.set_spike_times(np.array(spike_times[int(pre_cell)]))

                    # Create spike-detecting NetCon object attached to the cell's soma
                    # midpoint
                    if self.simulation_params["record_all"]:
                        for sec in post_cell.somalist:
                            post_cell.netconlist.append(neuron.h.NetCon(sec(0.5)._ref_v,
                                                        None,sec=sec))
                            post_cell.netconlist[-1].threshold = -52.0 # as in the LIF net.
                            post_cell.netconlist[-1].weight[0] = 0.0
                            post_cell.netconlist[-1].delay = 0.0

                    # record spike events
                    if self.simulation_params["record_all"]:
                        spikes = neuron.h.Vector()
                        post_cell.netconlist[-1].record(spikes)

                    # Simulate
                    post_cell.simulate(rec_imem=True,
                                       rec_current_dipole_moment=True)

                    # Compute dipole
                    P = post_cell.current_dipole_moment

                    # Compute EEG
                    eeg_top = []
                    if self.simulation_params["individual_EEG"]:
                        somapos = np.array([x_cell_pos[j][cell_id],
                                            y_cell_pos[j][cell_id],
                                            8500])
                        r_soma_syns = [post_cell.get_intersegment_vector(idx0=0,
                                      idx1=i) for i in post_cell.synidx]
                        r_mid = np.average(r_soma_syns, axis=0)
                        r_mid = somapos + r_mid/2.

                        # Change position of the EEG electrode
                        # print("Warning: The EEG electrode is not at the top!!!")
                        # theta_r = np.pi * 0.5
                        # phi_angle_r = 0.0
                        # x_eeg = (radii[3] - rad_tol) * np.sin(theta_r) * np.cos(phi_angle_r)
                        # y_eeg = (radii[3] - rad_tol) * np.sin(theta_r) * np.sin(phi_angle_r)
                        # z_eeg = (radii[3] - rad_tol) * np.cos(theta_r)
                        # eeg_coords = np.vstack((x_eeg, y_eeg, z_eeg)).T

                        # EEG electrode at the top
                        eeg_coords = np.array([[0., 0., radii[3] - rad_tol]])

                        # Four-Sphere method
                        four_sphere_top = LFPy.FourSphereVolumeConductor(radii,
                                                        sigmas, eeg_coords)
                        pot_db_4s_top = four_sphere_top.calc_potential(P, r_mid)
                        eeg_top = (np.array(pot_db_4s_top) * 1e6)[0]
                        P = []

                    # Set up the extracellular electrode
                    grid_electrode = LFPy.RecExtElectrode(post_cell,
                                                           **electrode_parameters)
                    # Compute LFP
                    grid_electrode.calc_lfp()

                    # send LFP/EEG of this cell to RANK 0
                    if RANK != 0:
                        if self.simulation_params["individual_EEG"]:
                            COMM.send([grid_electrode.LFP,eeg_top], dest=0)
                        else:
                            COMM.send([grid_electrode.LFP,P], dest=0)
                    else:
                        summed_LFP += grid_electrode.LFP
                        if self.simulation_params["individual_EEG"]:
                            if len(np.argwhere(np.isnan(eeg_top))) == 0:
                                summed_EEG_top += eeg_top
                        else:
                            summed_dipole += P

                    # Synapses
                    if self.simulation_params["record_all"]:
                        syns = []
                        for s in post_cell.synapses:
                            syns.append([s.x,s.y,s.z,s.kwargs['e']])

                # collect single LFP/EEG contributions on RANK 0
                if RANK == 0:
                    if cell_id % SIZE != RANK:
                        data = COMM.recv(source = cell_id % SIZE)
                        summed_LFP += data[0]

                        if self.simulation_params["individual_EEG"]:
                            if len(np.argwhere(np.isnan(data[1]))) == 0:
                                summed_EEG_top += data[1]
                        else:
                            summed_dipole += data[1]

                # Save results to file
                if cell_id % SIZE == RANK:
                    if self.simulation_params["record_all"]:
                        if cell_id == 0:
                            # Create dict
                            results_dict = {
                            "somav":post_cell.somav,
                            "spikes":spikes,
                            "syns":syns,
                            "xyz_rotations":xyz_rotations,
                            "x_cell_pos":x_cell_pos,
                            "y_cell_pos":y_cell_pos,
                            "z_cell_pos":z_cell_pos,
                            "cell_params_ex":self.cell_params_ex,
                            "cell_params_in": self.cell_params_in,
                            "radii":radii,
                            "sigmas":sigmas,
                            "rad_tol":rad_tol
                            }
                        else:
                            # Create dict
                            results_dict = {
                            "somav":post_cell.somav,
                            "spikes":spikes,
                            "syns":syns,
                            }

                        pickle.dump(results_dict,
                            open('../results/'+self.simulation_params["experiment_id_3D"]+"/"+\
                            filename+str(cell_id+j*self.simulation_params["population_sizes"][0]),
                            "wb"))

                    else:
                        if cell_id == 0:
                            # Create dict
                            results_dict = {
                            "cell_params_ex":self.cell_params_ex,
                            "cell_params_in": self.cell_params_in,
                            "radii":radii,
                            "sigmas":sigmas,
                            "rad_tol":rad_tol
                            }

                            pickle.dump(results_dict,
                                open('../results/'+self.simulation_params["experiment_id_3D"]+"/"+\
                                filename+str(cell_id+j*self.simulation_params["population_sizes"][0]),
                                "wb"))

        # Resync MPI threads
        COMM.Barrier()

        if RANK == 0:
            # Save time vector
            pickle.dump(post_cell.tvec,
                        open('../results/'+self.simulation_params["experiment_id_3D"]+"/"+\
                             filename+"tvec","wb"))
            # Save LFP,EEG,CDM
            if self.simulation_params["decimate"]:
                data_tofile = tools.decimate(summed_LFP,10)
            else:
                data_tofile = summed_LFP

            pickle.dump(data_tofile,
                        open('../results/'+self.simulation_params["experiment_id_3D"]+"/"+\
                             filename+"_LFP","wb"))

            if self.simulation_params["individual_EEG"]:
                if self.simulation_params["decimate"]:
                    data_tofile = tools.decimate(summed_EEG_top,10)
                else:
                    data_tofile = summed_EEG_top

                pickle.dump(data_tofile,
                            open('../results/'+self.simulation_params["experiment_id_3D"]+"/"+\
                                 filename+"_EEG","wb"))
            else:
                if self.simulation_params["decimate"]:
                    data_tofile = tools.decimate(np.transpose(np.array(summed_dipole)),10)
                else:
                    data_tofile = np.transpose(np.array(summed_dipole))

                pickle.dump(data_tofile,
                            open('../results/'+self.simulation_params["experiment_id_3D"]+"/"+\
                                 filename+"_CDM","wb"))

            # Print computation time
            end_c = time.time()
            print("\n\ntime elapsed: %s min" % str((end_c - start_c)/60.0))

        # Cleanup of object references. It will allow the script
        # to be run in successive fashion
        post_cell = None
        grid_electrode = None
        syn = None
        four_sphere_top = None
        neuron.h('forall delete_section()')