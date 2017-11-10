# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 15:32:48 2016

@author: mbeiran
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 10:53:47 2016

@author: mbeiran
"""

import os
import numpy as np
import f_network as fn
import f_noise as fs
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rcParams['text.usetex'] = True 
plt.rc('font',  size = 28)
#%%
sigma = 0.3
muA = 1.3
fmax = 15.0

# this is an inset axes over the main axes

LLL = ['hom', 'het']
II = 0
for  het_or_hom in LLL:
    fig = plt.figure(figsize = [4.5*1.5,4*1.5])    
    if het_or_hom == 'hom':
        ff = '/home/mbeiran/Documents/MasterThesis/Programming/Newhom_Sigma'+str(sigma)+'_muA_'+str(muA)+'_fmax_'+str(fmax)+'/'
    elif het_or_hom == 'het':
        ff = '/home/mbeiran/Documents/MasterThesis/Programming/Newhet_Sigma'+str(sigma)+'_muA_'+str(muA)+'_fmax_'+str(fmax)+'/'
    
    folder_imagesP = ff+'Plots_png/'
    if not os.path.exists(folder_imagesP):
        os.mkdir(folder_imagesP)
    folder_imagesE = ff+'Plots_eps/'
    if not os.path.exists(folder_imagesE):
        os.mkdir(folder_imagesE)
    tring = 'Sigma_'
    
    
    #==============================================================================
    #   Parameters 
    #==============================================================================
    thres = 1.0
    res = 0.0
    tref = 0.1 #units of tau
    tau = 1 #ms
    to_ms = 20.0
    
    Ds = 0.005
    #Tools
    T = 30.0
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
    dt_spk = dt#0.005
    t_stim = np.arange(0.0,T, dt_spk)
    mus_hom = np.ones(Nneur)*muA
    Da_het = 0.0
        
    trials = 1000
    
    cod_frac = np.zeros(25)
    CoH = np.zeros((2187, 25))
    PhCoH = np.zeros((2187, 25))
    
    DaS = np.zeros(25)
    DaS[5:] = np.logspace(-3.5,1.2,40)[::2]
    
    Das = [np.logspace(-3.5,1.2,40)[::2]]
    nN = 5
    DAS = np.zeros(nN)
    for i in range(nN):
        if i ==0:
            DAS[nN-1-i] = Das[0][0]*Das[0][0]/Das[0][1]
        else:
            DAS[nN-1-i] = DAS[nN-i]*Das[0][0]/Das[0][1]
    DaS[0:5] = DAS

    
    for i, param in enumerate(np.arange(25)):
        Das = [DaS[param],]
        Da = Das[0]
        folder_name = ff+'Da_'+str(Da)+'/'
        
        #==============================================================================
        #   Let's start loco 
        #==============================================================================
        
        # Insert inside loop. 1000 trials
        #str_t = folder_name+'output_Da_'+str(Da)+'_trial_'+str(t)+'.npy'
        #np.save(str_t,OUT)
        
    
        if het_or_hom == 'hom':    
            str1 = folder_name+'time_hom_Da_'+str(Da)+'.npy'
            str2 = folder_name+'stim_hom'+str(Da)+'.npy'
            str3 = folder_name+'freq_hom_Da_'+str(Da)+'.npy'
            str4 = folder_name+'Cross_spectrum_hom_Da_'+str(Da)+'.npy'
            str5 = folder_name+'PSD_stim_het_Da_'+str(Da)+'.npy'
            str6 = folder_name+'PSD_out_het_Da_'+str(Da)+'.npy'
            str7 = folder_name+'ISIs'+str(Da)+'.npy'
    
        elif het_or_hom == 'het':  
            str1 = folder_name+'time_het_Da_'+str(Da)+'.npy'
            str2 = folder_name+'stim_het'+str(Da)+'.npy'
            str3 = folder_name+'freq_het_Da_'+str(Da)+'.npy'
            str4 = folder_name+'Cross_spectrum_het_Da_'+str(Da)+'.npy'
            str5 = folder_name+'PSD_stim_het_Da_'+str(Da)+'.npy'
            str6 = folder_name+'PSD_out_het_Da_'+str(Da)+'.npy'
            str7 = folder_name+'ISIs'+str(Da)+'.npy'
        
    tab, tab1 = fn.get_nice_colors()
    Das = np.logspace(-3.5,1.2,40)[::2]
    II +=1    
    a = fig.add_subplot(111)
    majory = 2.0
    minory = 50
    majorx = 2.
    minorx= 50
    fn.get_rightframe(a, majory=majory, minory = minory, majorx=majorx, minorx=minorx, xlabel = "ISI", ylabel=r"$\rho(ISI)$", fontsize=25, labelsize= 25)
    nums = [4, 13, 24]
    if het_or_hom=='hom':
        lb_text = r'\underline{Noise intensity}'
    else:
        lb_text = r'\underline{Heterogeneity level}'
        
    strss = ['Low', 'Medium', 'Strong']
    if het_or_hom=='hom':
        binnums = [70, 199, 350]
    else:
        binnums = [70, 150, 220]
    for i,n in enumerate(nums):
        print(DaS[n])
        folder_name = ff+'Da_'+str(DaS[n])+'/'    
        #str3 = folder_name+'spks.npy'
        #spks = np.load(str3)
        str7 = folder_name+'ISIs'+str(DaS[n])+'.npy'
        ISIss = np.load(str7)
        #ISIs = fn.giveISIs(spks, dt_spk)
        his, hbins = np.histogram(ISIss, bins = binnums[i], density = True)
        #hbins = np.load(folder_name+'ISI_support'+str(DaS[n])+'.npy')
        #his = np.load(folder_name+'ISI_density'+str(DaS[n])+'.npy')
        a.plot(hbins[:-1], his, linewidth=4.0, color=tab[2*i], label=strss[i])
        print(len(hbins))
        theor_dens, t, dd = fn.get_densISI(DaS[n], muA, thres = 1.0, res = 0.0, tref = 0.1, T = 12, dt = 0.01, tol = 1e-7)
        a.plot(t, theor_dens, '--k', dashes = (5,1.9), linewidth=3.0)
    a.set_xlabel("interspike interval", fontsize=25)
    a.set_ylabel(r"ISI density", fontsize=25)
    a.set_xlim([0.01,4.6])
    a.set_ylim([0.0, 6.0])
 
             
    a.set_xticklabels('')
    a.set_yticklabels('')
    l = plt.legend(fontsize = 21., loc = 5, title=lb_text, frameon=False)
    a.get_legend().get_title().set_fontsize('22')

    
    folder = '/home/mbeiran/Documents/MasterThesis/Programming/'
    a.get_xaxis().set_tick_params(direction='out', width=2, length=10)
    a.get_yaxis().set_tick_params(direction='out', width=2, length=10)
    plt.tight_layout()
    plt.savefig(folder+'figs_2_b4'+het_or_hom+'.eps', dpi=600)
