# -*- coding: utf-8 -*-

import sys,os
import numpy as np
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import math

# Properties of plots
DPI = 300 # journals typically requires a minimum of 300 dpi
font_1 = {'fontname':'Arial','fontsize':8}
font_2 = {'fontname':'Arial','fontsize':10}
font_3 = {'fontname':'Arial','fontsize':12}

# Update (safely) dictionaries
def updateDicts(dict1, dict2):
    assert(isinstance(dict1, dict))
    assert(isinstance(dict2, dict))

    tmp = dict1.copy()
    tmp.update(dict2)
    return tmp

# Create a 'generic' plot
def plot(fig_size,number_rows,number_cols,xlabels,ylabels,titles,
x_data,y_data,y_range,data_labels,line_type,legends,plot_types = []):
    fig = plt.figure(figsize=fig_size, dpi=DPI)

    factor_cols = 0.25 - number_cols*0.05
    factor_rows = 0.25 - number_rows*0.05

    box_width = (1.0 - number_cols*factor_cols - 0.05)/number_cols
    box_height = (1.0 - number_rows*factor_rows - 0.1)/number_rows

    j = 0
    for r in range(int(number_rows)-1,-1,-1):
        for c in range(int(number_cols)):
            Vax = fig.add_axes([factor_cols+c*(box_width+factor_cols),
            factor_rows+r*(box_height+factor_rows),
            box_width,box_height],aspect='auto', frameon=True)

            Vax.spines["top"].set_visible(False)
            Vax.spines["right"].set_visible(False)
            Vax.set_xlabel(xlabels[j],**font_2)
            Vax.set_ylabel(ylabels[j],**font_2)
            Vax.set_title(titles[j],**font_3)

            labels_x = np.round(np.linspace(x_data[j][0][0], x_data[j][0][-1],4),2)
            Vax.set_xticks(labels_x)
            Vax.set_xticklabels(labels_x,**font_2)

            labels_y = np.round(np.linspace(y_range[j][0],y_range[j][1],4),2)
            Vax.set_yticks(labels_y)
            Vax.set_yticklabels(labels_y,**font_2)

            for i in range(len(x_data[j])):
                if len(line_type[j]) > 0 and len(data_labels[j]) > 0:
                    if len(plot_types) == 0:
                        Vax.plot(x_data[j][i],y_data[j][i],line_type[j][i],label=data_labels[j][i])
                    else:
                        if plot_types[j] == 'semilogy':
                            Vax.semilogy(x_data[j][i],y_data[j][i],line_type[j][i],label=data_labels[j][i])
                        elif plot_types[j] == 'semilogx':
                            Vax.semilogx(x_data[j][i],y_data[j][i],line_type[j][i],label=data_labels[j][i])
                        elif plot_types[j] == 'loglog':
                            Vax.loglog(x_data[j][i],y_data[j][i],line_type[j][i],label=data_labels[j][i])
                        else:
                            Vax.plot(x_data[j][i],y_data[j][i],line_type[j][i],label=data_labels[j][i])

                else:
                    if len(plot_types) == 0:
                        Vax.plot(x_data[j][i],y_data[j][i])
                    else:
                        if plot_types[j] == 'semilogy':
                            Vax.semilogy(x_data[j][i],y_data[j][i])
                        elif plot_types[j] == 'semilogx':
                            Vax.semilogx(x_data[j][i],y_data[j][i])
                        elif plot_types[j] == 'loglog':
                            Vax.loglog(x_data[j][i],y_data[j][i])
                        else:
                            Vax.plot(x_data[j][i],y_data[j][i])

            if legends[j] == True:
                Vax.legend(prop={'family':font_2['fontname'],'size':font_2['fontsize']})

            Vax.set_ylim([y_range[j][0],y_range[j][1]])

            j+=1

# Post-Stimulus Time Histogram (PSTH) and Inter-Spike Interval (ISI)
def PSTH(sim_time,data,senders,population,selected_cells,bin_size):
    # PSTHs of all cells in the population
    PSTH = np.zeros((len(population),int(sim_time/bin_size)))
    # PSTHs of a subset of cells within the population
    selected_PSTHs = []
    # Averaged PSTH of all cells
    avg_PSTH = np.zeros(int(sim_time/bin_size))
    # Average coefficient of variation of the Inter-Spike Interval (ISI)
    avg_ISI = []

    j = 0
    for cell in population:
        sender_positions = np.where(senders==cell)
        spike_times = (data[0]['times'])[sender_positions[0]]

        # Count spikes in each bin
        for t in np.arange(0,sim_time,bin_size):
            spikes = spike_times[np.where((spike_times >= t) & (spike_times < t+\
            bin_size))[0]]
            PSTH[j,int(t/bin_size)] += len(spikes)

        # Convert to firing rate
        PSTH[j,:]*=(1000.0/bin_size)
        avg_PSTH+=PSTH[j,:]

        # ISI
        d = np.diff(spike_times)
        if len(d) > 5: # at least 6 spikes to get a reliable measure
            # Coeff. of variation of the ISI
            avg_ISI.append(scipy.stats.variation(d))
        j+=1

        if cell in selected_cells:
            selected_PSTHs.append(PSTH[j,:])

    avg_PSTH/= len(population)

    # Mean firing rate (first 500 ms of the simulations are not included)
    mean_FR = np.mean(avg_PSTH[int(500.0/bin_size):])

    # Check for very traces with very low spiking activity
    if len(avg_ISI)==0:
        avg_ISI.append([0.0])

    return [PSTH,selected_PSTHs,avg_PSTH,mean_FR,np.mean(avg_ISI)]

# LFP proxies
def LFP(sim_time,sim_step,data,senders,PSTH,population,E_ex,E_in,g_L,PSTH_bin,
        pandasdf=[]):
    first_trace = (data[0][0]['V_m'])[np.where(senders[0]==population[0])[0]]

    LFP = []
    AMPA_current = np.zeros(len(first_trace))
    GABA_current = np.zeros(len(first_trace))
    Vm = np.zeros(len(first_trace))

    if len(pandasdf) < 1:
        for cell in population:
            sender_pos = np.where(senders[0]==cell)[0]

            AMPA_current += -(data[0][0]['g_ex'])[sender_pos] *\
            ((data[0][0]['V_m'])[sender_pos] - E_ex)
            GABA_current += -(data[0][0]['g_in'])[sender_pos] *\
            ((data[0][0]['V_m'])[sender_pos] - E_in)
            Vm += (data[0][0]['V_m'])[sender_pos]

    # Using Pandas dataframe
    else:
        for k,i in enumerate(population):
            sys.stdout.write("\r" + "Processing cell %s " % k)
            sys.stdout.flush()
            rr = slice(k*len(first_trace),(k+1)*len(first_trace),1)

            AMPA_current += -(pandasdf[rr]['g_ex'].values) *\
            ((pandasdf[rr]['V_m'].values) - E_ex)
            GABA_current += -(pandasdf[rr]['g_in'].values) *\
            ((pandasdf[rr]['V_m'].values) - E_in)
            Vm += pandasdf[rr]['V_m'].values

    # Sum of firing rates
    summed_PSTH = np.sum(PSTH,axis=0)
    scaled_PSTH = []
    for t in np.arange(0.0,sim_time-1.0,sim_step):
        # moving average (5 ms)
        prev = 0
        delta = int(5./PSTH_bin)
        if int(t/PSTH_bin) >= delta:
            for k in np.arange(int(t/PSTH_bin) - delta,int(t/PSTH_bin),1):
                prev += summed_PSTH[k]

        scaled_PSTH.append((summed_PSTH[int(t/PSTH_bin)] + prev)/(delta+1.))

    LFP.append(scaled_PSTH)
    # Sum of AMPA PSCs
    LFP.append(AMPA_current)
    # Sum of GABA PSCs (change sign)
    LFP.append(-GABA_current)
    # Mean of membrane potential (change sign)
    LFP.append(-Vm)
    # Sum of AMPA and GABA PSCs (change sign)
    I1 = AMPA_current+GABA_current
    LFP.append(-I1)
    # Sum of absolute values of AMPA and GABA PSCs
    I2 = np.abs(AMPA_current) + np.abs(GABA_current)
    LFP.append(I2)
    # Reference Weighted Sum (RWS)
    delay = int(6.0 / sim_step)
    AMPA_delayed = np.concatenate((np.zeros(delay),AMPA_current))
    I3 = AMPA_delayed[:len(GABA_current)]-1.65*GABA_current
    LFP.append(I3)

    return LFP

# Estimation of power spectral density using Welch’s method. We divided the data into
# 8 overlapping segments, with 50% overlap (default)
def powerSpectrum(signal,sim_time,sim_step):
    return scipy.signal.welch(signal,fs = 1000./\
    sim_step, nperseg = (sim_time/(8.0 * sim_step)))

# Mean value and histogram of the correlation coefficient between pairs of neurons
def pairwiseCorrelation(x,bin = 200,hist_range = [-1.0,1.0]):
    cc = []
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                c,p_value = scipy.stats.pearsonr(x[i], x[j])
                if math.isnan(c) == False:
                    cc.append(c)

    hist, bin_edges = np.histogram(cc,bins=bin,range=hist_range)
    return np.mean(cc),hist,bin_edges

# For the computation of the PSPs and PSCs we subtract the MP of the original
# neurons from the one of the cloned neurons and, by doing a spike triggered
# average over time and selected neurons, we obtain the average effective PSP.
def postSynaptic(simtime,simstep,senders_v,data_v,pop_ex,pop_cloned_AMPA,
pop_cloned_GABA,E_ex,E_in):
    PSP_AMPA = np.zeros(int(100.0/simstep))
    PSP_GABA = np.zeros(int(100.0/simstep))
    PSC_AMPA = np.zeros(int(100.0/simstep))
    PSC_GABA = np.zeros(int(100.0/simstep))

    for n in range(len(pop_cloned_AMPA)):
        pos_exc = np.where(senders_v[0]==pop_ex[n])
        pos_cloned_AMPA = np.where(senders_v[2]==pop_cloned_AMPA[n])
        pos_cloned_GABA = np.where(senders_v[3]==pop_cloned_GABA[n])

        V_dif_AMPA = (data_v[2][0]['V_m'])[pos_cloned_AMPA]-\
        (data_v[0][0]['V_m'])[pos_exc]
        V_dif_GABA = (data_v[3][0]['V_m'])[pos_cloned_GABA]-\
        (data_v[0][0]['V_m'])[pos_exc]

        current_ex_AMPA = -(data_v[0][0]['g_ex'])[pos_exc] *\
        ((data_v[0][0]['V_m'])[pos_exc] - E_ex)

        current_ex_GABA = -(data_v[0][0]['g_in'])[pos_exc] *\
        ((data_v[0][0]['V_m'])[pos_exc] - E_in)

        current_cloned_AMPA = -(data_v[2][0]['g_ex'])[pos_cloned_AMPA] *\
        ((data_v[2][0]['V_m'])[pos_cloned_AMPA] - E_ex)

        current_cloned_GABA = -(data_v[3][0]['g_in'])[pos_cloned_GABA] *\
        ((data_v[3][0]['V_m'])[pos_cloned_GABA] - E_in)

        current_dif_AMPA = current_ex_AMPA - current_cloned_AMPA
        current_dif_GABA = current_ex_GABA - current_cloned_GABA

        i=0
        # first 500 ms of the simulations are not included
        for j in np.arange(500.0,simtime-100.0,100.0):
            PSP_AMPA += V_dif_AMPA[int(j/simstep):
            int(j/simstep)+len(PSP_AMPA)]
            PSP_GABA += V_dif_GABA[int(j/simstep):
            int(j/simstep)+len(PSP_GABA)]
            PSC_AMPA += current_dif_AMPA[int(j/simstep):
            int(j/simstep)+len(PSC_AMPA)]
            PSC_GABA += current_dif_GABA[int(j/simstep):
            int(j/simstep)+len(PSC_GABA)]
            i+=1

    PSP_AMPA/=(i*len(pop_cloned_AMPA))
    PSP_GABA/=(i*len(pop_cloned_GABA))
    PSC_AMPA/=(i*len(pop_cloned_AMPA))
    PSC_GABA/=(i*len(pop_cloned_GABA))

    return [PSP_AMPA,PSP_GABA,PSC_AMPA,PSC_GABA]

# Load recorded data of multimeters from file
def loadRec(magnitudes_to_record,num_threads):

    # Global arrays for all populations of neurons. Same format as the one used
    # in network.py
    data_v = []
    data_s = []

    # Specific arrays for each population
    RecordingNodes = {} # (V_m,g_ex,g_in)
    RecordingNodes['senders'] = []
    RecordingNodes['times'] = []
    for m in magnitudes_to_record:
        RecordingNodes[m] = []

    SpikesRecorders = {} # spikes
    SpikesRecorders['senders'] = []
    SpikesRecorders['times'] = []

    # Data of each multimeter is splitted into num_threads files
    thread_counter = 0

    # Load all files
    dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'results'))
    ldir = os.listdir(dir)
    ldir.sort()

    for filename in ldir:
        # Read file
        if 'dat' in filename or 'gdf' in filename:
            sys.stdout.write("\r" + "Loading file %s " % filename)
            sys.stdout.flush()
            text_file = open('../results/'+filename, "r")
            lines = [line.rstrip('\n') for line in text_file]
            for n in range(len(lines)):
                h = lines[int(n)].split("\t")
                e = []
                for element in h[0:len(h)-1]:
                    e.append(float(element))
                # Check the type of recorder
                if("RecordingNode" in filename):
                    RecordingNodes['senders'].append(int(e[0]))
                    RecordingNodes['times'].append(e[1])
                    for j,m in enumerate(magnitudes_to_record):
                        RecordingNodes[m].append(e[j+2])
                else:
                    SpikesRecorders['senders'].append(int(e[0]))
                    SpikesRecorders['times'].append(e[1])

            # new array
            if thread_counter >= num_threads-1:
                thread_counter = 0

                # Converto to numpy arrays
                RecordingNodes['senders'] = np.array(RecordingNodes['senders'])
                RecordingNodes['times'] = np.array(RecordingNodes['times'])
                for j,m in enumerate(magnitudes_to_record):
                    RecordingNodes[m]=np.array(RecordingNodes[m])

                SpikesRecorders['senders'] = np.array(SpikesRecorders['senders'])
                SpikesRecorders['times'] = np.array(SpikesRecorders['times'])

                # append to global arrays
                if("RecordingNode" in filename):
                    # Same format as used in NEST
                    data_v.append((RecordingNodes,))
                else:
                    data_s.append((SpikesRecorders,))

                RecordingNodes = {} # (V_m,g_ex,g_in)
                RecordingNodes['senders'] = []
                RecordingNodes['times'] = []
                for m in magnitudes_to_record:
                    RecordingNodes[m] = []

                SpikesRecorders = {} # spikes
                SpikesRecorders['senders'] = []
                SpikesRecorders['times'] = []

            else:
                thread_counter+=1

            text_file.close()

    # Senders (same format as the one used in network.py)
    senders_v = []
    for i in range(len(data_v)):
        senders_v.append(data_v[i][0]['senders'],)

    senders_s = []
    for i in range(len(data_s)):
        senders_s.append(data_s[i][0]['senders'],)

    return [data_v,data_s,senders_v,senders_s]

# Load data from file
def loadData(experiment_id,filename,extension):

    data = pickle.load(open( '../results/'+experiment_id+'/'+filename+extension,"rb") )

    # data = []
    # lines = [line.rstrip('\n') for line in open('../results/'+\
    # filename+extension, "r")]
    #
    # for n in range(len(lines)):
    #     h = lines[int(n)].split(',')
    #     e = []
    #     for element in h[0:len(h)-1]:
    #         e.append(float(element))
    #     data.append( e )

    return data

# Save data to file
def saveData(experiment_id,filename,extension,data):

    pickle.dump(data,open('../results/'+experiment_id+'/'+filename+extension, "wb" ) )

    # text_file = open('../results/'+ filename+extension, "w")
    #
    # for line in range(len(data)):
    #     for ch in data[line]:
    #         text_file.write(str(ch))
    #         text_file.write(",")
    #     text_file.write(os.linesep)
    #
    # text_file.close()

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
