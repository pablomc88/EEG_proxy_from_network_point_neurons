# -*- coding: utf-8 -*-

###############################################################################
## Additional functions to load data and compute R^2.                        ##
## Author: Pablo Martinez Canada (pablo.martinez@iit.it)                     ##
## Date: 05/02/2021                                                          ##
###############################################################################

import sys,os
import numpy as np
import scipy.stats
import scipy.signal
import pickle

# Load LIF data
def loadData(experiment_id,filename,extension):

    data = pickle.load(open( '../LIF_network/results/'+\
                            experiment_id+"/"+filename+extension,"rb"),encoding='latin1' )

    return data

# Load simulation results of the multicompartment network
def loadResults(experiment_id,filename,population_sizes,Z,tstop,dt,individual_EEG):

    print("\nLoading results...")

    # Summed current dipole moments and EEG
    summed_EEG_top = np.zeros(int(tstop /dt + 1))
    if individual_EEG == False:
        summed_dipole = np.zeros((int(tstop /dt + 1),3))

    # CDM / EEG
    if individual_EEG:
        summed_EEG_top = pickle.load(open( '../multicompartment_network/results/'+experiment_id+"/"+filename+"_EEG","rb"),encoding='latin1' )
        summed_dipole = []
    else:
        summed_dipole = np.array(pickle.load(open(
                        '../multicompartment_network/results/'+experiment_id+"/"+filename+"_CDM","rb"),encoding='latin1'))
        # Backward compatibility
        if len(summed_dipole[0])>3:
            summed_dipole = np.transpose(summed_dipole)

    # tvec
    tvec = pickle.load(open( '../multicompartment_network/results/'+experiment_id+"/"+filename+"tvec","rb") ,encoding='latin1')

    # Sum all CDMs and compute the EEG
    if individual_EEG==False:
        import LFPy
        print("\nComputing EEG...")

        # Simulation parameters
        sim_params = pickle.load(open(  '../multicompartment_network/results/'+\
                                        experiment_id+"/"+filename+str(0),"rb") ,encoding='latin1')

        # EEG: four_sphere parameters
        radii = sim_params["radii"]
        sigmas = sim_params["sigmas"]
        rad_tol = sim_params["rad_tol"]

        r_mid = np.array([0., 0., 8500])
        eeg_coords_top = np.array([[0., 0., radii[3] - rad_tol]])
        four_sphere_top = LFPy.FourSphereVolumeConductor(radii,
                                        sigmas, eeg_coords_top)
        pot_db_4s_top = four_sphere_top.calc_potential(summed_dipole, r_mid)
        summed_EEG_top = (np.array(pot_db_4s_top) * 1e6)[0]

    # Return results
    return [summed_dipole,summed_EEG_top,tvec]

# Adapt the 3D and LIF signals to have the same length and start time
def adaptSignals(pos,data_3D,data_LIF,start_time_pos_3D,start_time_pos_LIF,dt_3D,baseline):
    # Normalize the 3D signal
    signal_3D = data_3D[pos,start_time_pos_3D:]
    # First 500 ms are not used for normalization
    signal_3D_norm = (signal_3D - np.mean(signal_3D[int(500./dt_3D):]))/\
                                  np.std(signal_3D[int(500./dt_3D):])

    # The LIF signal is already normalized
    signal_LIF_norm = data_LIF[start_time_pos_LIF:]

    # The baseline is defined as the sign of the max value. For the EEG, the
    # baseline is -1. For the LFP is 0.
    if baseline > -1:
        posmax = np.argmax(np.abs(signal_3D))
        baseline = np.sign(signal_3D[posmax])

    # Shorten the longer signal
    if len(signal_3D_norm) >= len(signal_LIF_norm):
        # Adapt the signal of the 3D network
        len_new_signal = len(signal_3D_norm)- (len(signal_3D_norm) % len(signal_LIF_norm))
        signal_3D_norm = signal_3D_norm[0:len_new_signal:int(len_new_signal/len(signal_LIF_norm))]
    else:
        # Adapt the signal of the LIF network
        len_new_signal = len(signal_LIF_norm)- (len(signal_LIF_norm) % len(signal_3D_norm))
        signal_LIF_norm = signal_LIF_norm[0:len_new_signal:int(len_new_signal/len(signal_3D_norm))]

    return (signal_3D_norm,baseline * signal_LIF_norm,baseline)

# Cross-correlation, best delay of time-shifting the LFP/EEG and coefficient of determination
def getMetrics(Z,data_LIF,data_3D,start_time_pos_3D,start_time_pos_LIF,dt_3D,baseline=-1,
                timeShift = False):

    delays = np.zeros( (len(Z[0]),len(data_LIF)) )
    R2 = np.zeros( (len(Z[0]),len(data_LIF)) )
    cc = []
    data_LIF_norm = []
    data_3D_norm = []

    for i,z_i in enumerate(Z[0]):
        cc_z = []
        d_LIF = []
        d_3D = []
        for j,signal_proxy in enumerate(data_LIF):
            # Normalize signals
            signal_3D_norm, signal_LIF_norm,baseline = adaptSignals(i,data_3D,signal_proxy,
            start_time_pos_3D,start_time_pos_LIF,dt_3D,baseline)

            # Cross-correlation in time between the signal and the different proxies
            corr = np.correlate(signal_3D_norm,signal_LIF_norm,"full")
            cc_z.append(corr)
            # Find the maximum of the cc which will be the optimal delay
            best_delay = int(np.argmax(corr)-len(corr)/2)
            delays[i,j] = best_delay
            # Coefficient of determination
            if timeShift:
                new_signal_proxy = np.roll(signal_LIF_norm,best_delay)
            else:
                new_signal_proxy = signal_LIF_norm

            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
                                                    signal_3D_norm,new_signal_proxy)
            R2[i,j] = r_value**2
            # Normalized data
            d_3D.append(signal_3D_norm)
            d_LIF.append(new_signal_proxy)

        cc.append(cc_z)
        data_LIF_norm.append(d_LIF)
        data_3D_norm.append(d_3D)

    return delays,R2,np.sum(R2,axis = 0)/len(Z[0]),np.max(R2,axis=0),cc,data_LIF_norm,data_3D_norm

# Linear combination of the LIF synaptic currents
def sumCurrents(parameters,data):
    # Extended network
    if data[3]:
        delta_ext = 0.25 # mm
        total_current = 0.0
        # Sum of currents for all radial distances
        for segment in range(len(data[0])):
            new_AMPA = np.roll(data[0][segment],int(parameters[0]/data[2]))
            new_GABA = np.roll(data[1][segment],int(parameters[1]/data[2]))
            aux_current = new_AMPA -parameters[2]*new_GABA
            aux_current_norm = (aux_current - np.mean(aux_current))/np.std(aux_current)
            total_current += aux_current_norm * np.exp(-segment*delta_ext*parameters[3])

    # Standard network
    else:
        new_AMPA = np.roll(data[0],int(parameters[0]/data[2]))
        new_GABA = np.roll(data[1],int(parameters[1]/data[2]))
        total_current = new_AMPA -parameters[2]*new_GABA
    return (total_current - np.mean(total_current))/np.std(total_current)

# Recompute the proxy parameters by using a power function of the firing rate
def powParams(coeff,firing_rate):
    # tau_AMPA = coeff[0] * np.exp(-coeff[1] * firing_rate) + coeff[2]
    # tau_GABA = coeff[3] * np.exp(-coeff[4] * firing_rate) + coeff[5]
    # alpha = coeff[6] * np.exp(-coeff[7] * firing_rate) + coeff[8]
    tau_AMPA = coeff[0] * np.power(firing_rate,-coeff[1]) + coeff[2]
    tau_GABA = coeff[3] * np.power(firing_rate,-coeff[4]) + coeff[5]
    alpha = coeff[6] * np.power(firing_rate,-coeff[7]) + coeff[8]
    return [tau_AMPA,tau_GABA,alpha]

# Error function between the WS proxy and the LFP/EEG
def ErrorFunc(parameters,*x):
    return np.sum((sumCurrents(parameters,x[1:])-x[0])**2)

# By default this function returns the WS proxy. If a fixed set of parameters is passed,
# it returns the ERWS-1, or the ERWS-2 when the firing rate is positive.
def weightedSum(Z,AMPA_LIF,GABA_LIF,data_3D,start_time_pos_3D,start_time_pos_LIF,
                dt_LIF,dt_3D,baseline=-1,extended_network=False,fixed_vals = [0],
                firing_rate=-1.0,compute_BIC=False):
    WS = []
    R2 = np.zeros( (len(Z[0]),1) )
    best_parameters = []

    for i,z_i in enumerate(Z[0]):
        if extended_network:
            AMPA_array = []
            GABA_array = []
            for segment in range(len(AMPA_LIF)):
                # Normalize signals
                signal_3D_norm, signal_AMPA_norm,baseline = adaptSignals(i,data_3D,AMPA_LIF[segment],
                start_time_pos_3D,start_time_pos_LIF,dt_3D,baseline)
                signal_3D_norm, signal_GABA_norm,baseline = adaptSignals(i,data_3D,GABA_LIF[segment],
                start_time_pos_3D,start_time_pos_LIF,dt_3D,baseline)

                AMPA_array.append(signal_AMPA_norm)
                GABA_array.append(signal_GABA_norm)

            data = (signal_3D_norm, AMPA_array,GABA_array,dt_LIF,extended_network)
        else:
            # Normalize signals
            signal_3D_norm, signal_AMPA_norm,baseline = adaptSignals(i,data_3D,AMPA_LIF,
            start_time_pos_3D,start_time_pos_LIF,dt_3D,baseline)
            signal_3D_norm, signal_GABA_norm,baseline = adaptSignals(i,data_3D,GABA_LIF,
            start_time_pos_3D,start_time_pos_LIF,dt_3D,baseline)

            data = (signal_3D_norm, signal_AMPA_norm,signal_GABA_norm,dt_LIF,extended_network)

        if len(fixed_vals) > 1:
            if firing_rate < 0:
                best_vals = fixed_vals
            else:
                best_vals = powParams(fixed_vals,firing_rate)
        else:
            print("Optimizing parameters of the WS proxy...")

            # Fit the LFP/EEG by brute force:
            # Coarse exploration of the search space
            delta_coarse = 0.5
            if extended_network:
                intervals = (slice(-10,10,delta_coarse),
                             slice(-10,10,delta_coarse),
                             slice(-10,10,delta_coarse),
                             slice(0,5,delta_coarse))
            else:
                intervals = (slice(-10,10,delta_coarse),
                              slice(-10,10,delta_coarse),
                              slice(-10,10,delta_coarse))

            best_vals_first_search=brute(ErrorFunc,intervals,args=data,finish=None)

            # Exploitation of the best region found before
            if extended_network:
                rrange = ( slice(best_vals_first_search[0]-0.5,best_vals_first_search[0]+0.5,0.1),
                          slice(best_vals_first_search[1]-0.5,best_vals_first_search[1]+0.5,0.1),
                          slice(best_vals_first_search[2]-0.5,best_vals_first_search[2]+0.5,0.1),
                          slice(best_vals_first_search[3]-0.5,best_vals_first_search[3]+0.5,0.1))
            else:
                rrange = ( slice(best_vals_first_search[0]-0.5,best_vals_first_search[0]+0.5,0.1),
                          slice(best_vals_first_search[1]-0.5,best_vals_first_search[1]+0.5,0.1),
                          slice(best_vals_first_search[2]-0.5,best_vals_first_search[2]+0.5,0.1))

            best_vals=brute(ErrorFunc,rrange,args=data,finish=None)

            print ( "z_i = ",z_i," , Best parameters = ",best_vals )

        WS.append(sumCurrents(best_vals,data[1:]))
        best_parameters.append(best_vals)

        # Bayesian Information Criterion (BIC)
        if compute_BIC:
            RSS = ErrorFunc(best_vals,*data)
            n = len(data[0])

            if len(fixed_vals) > 1:
                if firing_rate < 0:
                    K = 3
                else:
                    K = 9
            else:
                K = 3

            BIC = n*np.log(RSS/n) + K*np.log(n)
            R2[i] = BIC

        # Coefficient of determination
        else:
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
                                        signal_3D_norm,sumCurrents(best_vals,data[1:]))
            R2[i] = r_value**2

    return (WS,R2,np.sum(R2,axis = 0)/len(Z[0]),np.max(R2,axis=0),best_parameters,
            np.mean(best_parameters,axis = 0))
