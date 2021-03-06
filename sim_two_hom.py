#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:49:33 2016

@author: mbeiran
"""
import os, sys
import numpy as np
import f_network as fn
import f_noise as fs
import matplotlib.pyplot as plt
plt.close('all')
plt.rc('text', usetex=True)
plt.rcParams['text.usetex'] = True 
plt.rcParams.update({'figure.autolayout': True})
plt.rc('font', family='serif', size = 16)
#%%
#==============================================================================
#   Parameters 
#==============================================================================
print('Starting')
param = int(sys.argv[1]) # From 0 to 24, the noise/heterogeneity level. It indicates the index that should be used of the array DaS
sigma = float(sys.argv[2]) # Standard deviation of the signal
muA = float(sys.argv[3])  # Mean input to the neurons in the network. Choose 1.3 or 1.05 
fmax = float(sys.argv[4]) # Cut-off frequency of the signal
thres = 1.0
res = 0.0
tref = 0.1 #units of tau
tau = 1 #ms
to_ms = 20.0

Ds = 0.5*sigma**2
#Tools
T = 35.0
dt = 5e-3 #0.005
time = np.arange(0,T, dt)
N = len(time)


df = 1.0/T 
f = np.fft.fftfreq(len(time))
f = f

mus = np.linspace(0.0, 30.0, 3000)

seed = 696
np.random.seed(seed)


Nneur = 300
#5e-4#0.005 #units of tau (20ms) All times are in units of tau
Das = [np.logspace(-3.5,1.2,40)[::2]]
nN = 5
DAS = np.zeros(nN)
for i in range(nN):
    if i ==0:
        DAS[nN-1-i] = Das[0]*Das[0]/Das[1]
    else:
        DAS[nN-1-i] = DAS[nN-i]*Das[0]/Das[1]
        
Das = [DAS[param]]
dt_spk = dt#0.005
t_stim = np.arange(0.0,T, dt_spk)
trials_s= 80
mus_hom = np.ones(Nneur*trials_s)*muA
Da_het = 0.0
    
trials = 300
Da = Das[0]
    
for Da in Das:
    #==============================================================================
    #       PDF of mus and P_ISI of homogeneous
    #==============================================================================
    #P_mu, tme, pISI, muS, P1_mu, TTs, PItheor= fn.get_hetmus_nonoise(mus, muA, Da, tref, tol=1e-10, dt=1e-4)
    #==============================================================================
    #               Choose random mus for heterogeneous 
    #==============================================================================
    x0 = 0.0*np.zeros((Nneur))
        
    Cross_hom = 0.0
    pSD_stim_hom = 0.0
    pSD_out_hom = 0.0
    
    FCross_hom = 0.0
    FpSD_stim_hom = 0.0
    FpSD_out_hom = 0.0
    ISIs = []
    #%%
    for t in range(trials):
        #print('trial '+str(t))
        Cross_hom = 0.0*1j
        pSD_stim_hom = 0.0
        pSD_out_hom = 0.0
        
        OUT = []
        Stim = fs.gen_band_limited(T, dt, fmax)

            
        spks_hom, t_spk, y_model, stim1, t_v, initial_cond, initial_cond1 =\
        fn.euler_maru_withnoise(fn.ffun2, Ds, Da, mus_hom, dt, dt_spk,\
        T, trials_s*Nneur, fmax, T_prerun=10.0, t_ref = tref,example=True,verbose=False, Stim=Stim)
        stim_het = np.interp(t_spk, t_v, stim1)
        
        for t_s in range(trials_s):
            
            stim_het = np.interp(t_spk, t_v, stim1)
            out_het = np.mean(spks_hom[:,t_s*Nneur:(t_s+1)*Nneur], axis=-1)/dt_spk
            if t_s == trials_s-11:
                OUT = out_het
            elif t_s>trials_s-11:
                OUT = np.vstack([OUT, out_het])
                if t<40:
                    ISIs.extend(fn.giveISIs(spks_hom[:,t_s*Nneur:(t_s+1)*Nneur], t_spk[1]-t_spk[0]))
            FTfreq, cross_het = fn.im_crossPSD(stim_het, out_het, t_spk)    
            FTfreq, PSD_out_het = fn.PSD( out_het, t_spk)
            FTfreq, PSD_stim_het = fn.PSD(stim_het,t_spk) 
            
            Cross_hom += cross_het/(1.0*trials_s)
            pSD_stim_hom += PSD_stim_het/(1.0*trials_s)
            pSD_out_hom += PSD_out_het/(1.0*trials_s)
    
#        string = folder_name+str(Da)+'_trial_'+str(t)
#        np.save(string,1)
        
        FCross_hom += Cross_hom/(1.0*trials)
        FpSD_stim_hom += pSD_stim_hom/(1.0*trials)
        FpSD_out_hom += pSD_out_hom/(1.0*trials)
    
    hist, bin_edges = np.histogram(ISIs, 100, normed=True)
#    string = folder_name+'output_hom_Da_'+str(Da)+'_10lasttrials'
#    np.save(string, OUT)
#    string = folder_name+'spks'
#    np.save(string, spks_hom[:, t_s*Nneur:(t_s+1)*Nneur])
#    string = folder_name+'time_hom_Da_'+str(Da)
#    np.save(string, t_spk)
#    string = folder_name+'stim_hom'+str(Da)
#    np.save(string, stim_het)
#
#    np.save(folder_name+'freq_hom_Da_'+str(Da), FTfreq)
#    np.save(folder_name+'Cross_spectrum_hom_Da_'+str(Da), FCross_hom)
#    np.save(folder_name+'PSD_stim_het_Da_'+str(Da), FpSD_stim_hom)
#    np.save(folder_name+'PSD_out_het_Da_'+str(Da), FpSD_out_hom)
#    np.save(folder_name+'ISI_density'+str(Da), hist)
#    np.save(folder_name+'ISI_support'+str(Da), bin_edges)
