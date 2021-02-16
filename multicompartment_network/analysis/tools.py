# -*- coding: utf-8 -*-

###############################################################################
## Functions to load data used in the multicompartment model network.        ##
##                                                                           ##
## Author: Pablo Martinez Canada (pablo.martinez@iit.it)                     ##
## Date: 15/02/2021                                                          ##
###############################################################################

import sys,os
import numpy as np
import scipy.stats
import scipy.signal
import pickle

# Load LIF model network data
def loadLIFData(experiment_id,filename,extension):

    data = pickle.load(open( '../../LIF_network/results/'+\
                            experiment_id+"/"+filename+extension,"rb"),encoding='latin1' )

    return data

# Load simulation results of the multicompartment model network
def loadmultData(experiment_id,filename,population_sizes,Z,tstop,dt,individual_EEG,
                record_all):

    print("\nLoading results...")

    # Membrane potentials
    potential_vectors = []
    # Spike times
    spike_vectors = []
    # to plot synapses
    synapses = [[] for n in range(sum(population_sizes))]

    # Summed LFPs
    summed_LFP = np.zeros((Z.size,int(tstop / dt + 1)))

    # Summed current dipole moments and EEG
    summed_EEG_top = np.zeros(int(tstop /dt + 1))
    if individual_EEG == False:
        summed_dipole = np.zeros((int(tstop /dt + 1),3))

    # LFP
    summed_LFP = pickle.load(open( '../results/'+experiment_id+"/"+filename+"_LFP","rb"),encoding='latin1' )
    # CDM / EEG
    if individual_EEG:
        summed_EEG_top = pickle.load(open( '../results/'+experiment_id+"/"+filename+"_EEG","rb"),encoding='latin1' )
        summed_dipole = []
    else:
        summed_dipole = np.array(pickle.load(open(
                        '../results/'+experiment_id+"/"+filename+"_CDM","rb"),encoding='latin1'))
        # Backward compatibility
        if len(summed_dipole[0])>3:
            summed_dipole = np.transpose(summed_dipole)

    # tvec
    tvec = pickle.load(open( '../results/'+experiment_id+"/"+filename+"tvec","rb") ,encoding='latin1')

    # iterate over the populations
    if record_all:
        for cell_id in range(sum(population_sizes)):
            sys.stdout.write("\r" + "Loading cell %s " % cell_id)
            sys.stdout.flush()

            data = pickle.load(open(  '../results/'+experiment_id+"/"+filename+str(cell_id),"rb"),encoding='latin1' )
            potential_vectors.append(data["somav"])
            spike_vectors.append(data["spikes"])
            synapses[cell_id] = data["syns"]

            if cell_id == 0:
                common_data = data
                # Common parameters
                xyz_rotations = common_data["xyz_rotations"]
                x_cell_pos = common_data["x_cell_pos"]
                y_cell_pos = common_data["y_cell_pos"]
                z_cell_pos = common_data["z_cell_pos"]
                cell_params_ex = common_data["cell_params_ex"]
                cell_params_in = common_data["cell_params_in"]
    else:
        common_data = pickle.load(open(  '../results/'+experiment_id+"/"+filename+str(0),"rb") ,encoding='latin1')
        # Common parameters
        xyz_rotations = []
        x_cell_pos = []
        y_cell_pos = []
        z_cell_pos = []
        cell_params_ex = common_data["cell_params_ex"]
        cell_params_in = common_data["cell_params_in"]

    # EEG: four_sphere parameters
    radii = common_data["radii"]
    sigmas = common_data["sigmas"]
    rad_tol = common_data["rad_tol"]

    # Sum all CDMs and compute the EEG
    if individual_EEG==False:
        print("\nComputing EEG...")

        r_mid = np.array([0., 0., 8500])
        eeg_coords_top = np.array([[0., 0., radii[3] - rad_tol]])
        four_sphere_top = LFPy.FourSphereVolumeConductor(radii,
                                        sigmas, eeg_coords_top)
        pot_db_4s_top = four_sphere_top.calc_potential(summed_dipole, r_mid)
        summed_EEG_top = (np.array(pot_db_4s_top) * 1e6)[0]

    # Return results
    return [potential_vectors,spike_vectors,summed_LFP,summed_dipole,
            summed_EEG_top,synapses,tvec,xyz_rotations,x_cell_pos,
            y_cell_pos,z_cell_pos,cell_params_ex,cell_params_in]

# Decimate data
def decimate(x, q=10, n=4, k=0.8, filterfun=scipy.signal.cheby1):
    """
    scipy.signal.decimate like downsampling using filtfilt instead of lfilter,
    and filter coeffs from butterworth or chebyshev type 1.

    Parameters
    ----------
    x : ndarray
        Array to be downsampled along last axis.
    q : int
        Downsampling factor.
    n : int
        Filter order.
    k : float
        Aliasing filter critical frequency Wn will be set as Wn=k/q.
    filterfun : function
        `scipy.signal.filter_design.cheby1` or
        `scipy.signal.filter_design.butter` function

    Returns
    -------
    ndarray
        Downsampled signal.

    """
    if not isinstance(q, int):
        raise TypeError("q must be an integer")

    if n is None:
        n = 1

    if filterfun == scipy.signal.butter:
        b, a = filterfun(n, k / q)
    elif filterfun == scipy.signal.cheby1:
        b, a = filterfun(n, 0.05, k / q)
    else:
        raise Exception('only scipy.signal.butter or scipy.signal.cheby1 supported')

    try:
        y = scipy.signal.filtfilt(b, a, x)
    except: # Multidim array can only be processed at once for scipy >= 0.9.0
        y = []
        for data in x:
            y.append(scipy.signal.filtfilt(b, a, data))
        y = np.array(y)

    try:
        return y[:, ::q]
    except:
        return y[::q]
