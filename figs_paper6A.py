# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 11:01:58 2017

@author: mbeiran
"""


import os
import numpy as np
import f_network as fn
import matplotlib.pyplot as plt
plt.close('all')
plt.rc('text', usetex=True)
plt.rcParams['text.usetex'] = True 
#plt.rcParams.update({'figure.autolayout': True})
plt.rc('font', family='serif', size = 16)
#%%
muA = 1.3
sigma = 0.2
fmax = 15.0
haze = 0

    
Nneur = 300

het_or_hom = 'hom'
theory = True
nums = [0, 6, 15]
if het_or_hom == 'hom':
    if sigma == 0.1:
        if muA == 1.3:
            if fmax == 2.0:
                nums = [6, 20, 24]
            elif fmax == 15.0:
                nums = [0, 5, 15]
        elif muA == 1.05:
            if fmax == 15.0:
                nums = [6, 15, 23]
    elif sigma == 0.5:
        if muA == 1.3:
            if fmax == 2.0:        
                nums = [6, 14, 23]
            elif fmax == 15.0:
                nums = [3, 11, 17]
        elif muA == 1.05:
            if fmax == 15.0:
                nums = [6, 15, 23]
    elif sigma == 0.05:
        if muA == 1.3:
            if fmax == 2.0:        
                nums = [6, 17, 24]
            elif fmax == 15.0:
                nums = [6, 12, 22]
        elif muA == 1.05:
            if fmax == 15.0:
                nums = [6, 13, 23]
    elif sigma == 0.01:
        nums = [0, 6, 12]
    elif sigma == 1.0:
        nums = [5, 13, 22]
    elif sigma == 0.2:
        nums = [0, 7, 14]        
elif het_or_hom=='het':
    if sigma == 0.1:
        if muA == 1.3:
            if fmax == 2.0:
                nums = [6, 20, 24]
            elif fmax == 15.0:
                nums = [6, 15, 23]
        elif muA == 1.05:
            if fmax == 15.0:
                nums = [6, 18, 24]
    elif sigma == 0.5:
        if muA == 1.3:
            if fmax == 2.0:        
                nums = [6, 14, 23]
            elif fmax == 15.0:
                nums = [6, 15, 23]
        elif muA == 1.05:
            if fmax == 15.0:
                nums = [6, 14, 23]
    elif sigma == 0.05:
        if muA == 1.3:
            if fmax == 2.0:        
                nums = [6, 17, 24]
            elif fmax == 15.0:
                nums = [6, 18, 23]
        elif muA == 1.05:
            if fmax == 15.0:
                nums = [6, 18, 24]
    elif sigma == 0.01:
        nums = [2, 11, 20]
    elif sigma == 0.005:
        nums = [2, 11, 19]
    elif sigma == 0.2:
        nums = [3, 11, 22]
    elif sigma == 0.0075:
        nums = [0, 7, 20]    
    elif sigma == 0.3:
        nums = [4, 13, 23]
if Nneur<300:
    nums = [0, 7, 18] #0.05

if het_or_hom == 'hom':
    if Nneur == 300:
        ff = '/home/mbeiran/Documents/MasterThesis/Programming/Newhom_Sigma'+str(sigma)+'_muA_'+str(muA)+'_fmax_'+str(fmax)+'/'
    else:
        ff = '/home/mbeiran/Documents/MasterThesis/Programming/Newhom_Sigma'+str(sigma)+'_muA_'+str(muA)+'_fmax_'+str(fmax)+'_Nneur_'+str(Nneur)+'/'
    
elif het_or_hom == 'het':
    
    if Nneur == 300:
        ff = '/home/mbeiran/Documents/MasterThesis/Programming/Newhet_Sigma'+str(sigma)+'_muA_'+str(muA)+'_fmax_'+str(fmax)+'/'
    else:
        ff = '/home/mbeiran/Documents/MasterThesis/Programming/Newhet_Sigma'+str(sigma)+'_muA_'+str(muA)+'_fmax_'+str(fmax)+'_Nneur_'+str(Nneur)+'/'
#%%
folder_imagesP = '/home/mbeiran/Documents/MasterThesis/Programming/'
folder_imagesE = '/home/mbeiran/Documents/MasterThesis/Programming/'
tring = 'Sigma_'


if het_or_hom == 'hom':
    np.save('/home/mbeiran/Documents/MasterThesis/Programming/Newhom_Sigma'+str(sigma)+'_muA_'+str(muA)+'_fmax_'+str(fmax)+'/Sigma'+str(sigma)+'_muA_'+str(muA)+'_fmax_'+str(fmax), 1)
    np.save('/home/mbeiran/Documents/MasterThesis/Programming/Newhom_Sigma'+str(sigma)+'_muA_'+str(muA)+'_fmax_'+str(fmax)+'/Plots_eps/Sigma'+str(sigma)+'_muA_'+str(muA)+'_fmax_'+str(fmax), 1)
    np.save('/home/mbeiran/Documents/MasterThesis/Programming/Newhom_Sigma'+str(sigma)+'_muA_'+str(muA)+'_fmax_'+str(fmax)+'/Plots_png/Sigma'+str(sigma)+'_muA_'+str(muA)+'_fmax_'+str(fmax), 1)

#==============================================================================
#   Parameters 
#==============================================================================
thres = 1.0
res = 0.0
tref = 0.1 #units of tau
tau = 1 #ms
to_ms = 20.0

Ds = sigma**2/(4*fmax)
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


#5e-4#0.005 #units of tau (20ms) All times are in units of tau
dt_spk = dt#0.005
t_stim = np.arange(0.0,T, dt_spk)
mus_hom = np.ones(Nneur)*muA
Da_het = 0.0
    
trials = 1000
if Nneur==300:
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

    paramN = 25
else:
    DaS = np.logspace(-3.5,1.2,40)[::2]
    paramN = 20
    
cod_frac = np.zeros(paramN)
parametroraro = 2187
if muA == 1.05:
    parametroraro = 3499
CoH = np.zeros((parametroraro, paramN))
PhCoH = np.zeros((parametroraro, paramN))

    
for i, param in enumerate(np.arange(paramN)):
    Das = DaS[param]
    Da = Das
    folder_name = ff+'Da_'+str(Da)+'/'
    
    #==============================================================================
    #   Let's start loco 
    #==============================================================================
    
    # Insert inside loop. 1000 trials
    #str_t = folder_name+'output_Da_'+str(Da)+'_trial_'+str(t)+'.npy'
    #np.save(str_t,OUT)
    

    if het_or_hom == 'hom' or Nneur<300:    
        str1 = folder_name+'time_hom_Da_'+str(Da)+'.npy'
        str2 = folder_name+'stim_hom'+str(Da)+'.npy'
        str3 = folder_name+'freq_hom_Da_'+str(Da)+'.npy'
        str4 = folder_name+'Cross_spectrum_hom_Da_'+str(Da)+'.npy'
        str5 = folder_name+'PSD_stim_het_Da_'+str(Da)+'.npy'
        str6 = folder_name+'PSD_out_het_Da_'+str(Da)+'.npy'

    elif het_or_hom == 'het'and Nneur==300:  
        str1 = folder_name+'time_het_Da_'+str(Da)+'.npy'
        str2 = folder_name+'stim_het'+str(Da)+'.npy'
        str3 = folder_name+'freq_het_Da_'+str(Da)+'.npy'
        str4 = folder_name+'Cross_spectrum_het_Da_'+str(Da)+'.npy'
        str5 = folder_name+'PSD_stim_het_Da_'+str(Da)+'.npy'
        str6 = folder_name+'PSD_out_het_Da_'+str(Da)+'.npy'
    
    try:
        t_spk = np.load(str1)
        stim_het = np.load(str2)
        freq = np.load(str3)
        Sxs = np.load(str4)
        Sss = np.load(str5)
        Sxx = np.load(str6)
    except IOError:
        print(i, ' not loaded')
        epsS = np.sqrt(np.sum(Sss*(1-Coh))/np.sum(Sss))
        cod_frac[i] =epsS
        continue
        
    df = freq[1]-freq[0]
    #%%
    
    Coh = np.abs(Sxs)**2/(Sss*Sxx)
    Phcoh = np.angle(Sxs)
    epsS = np.sqrt(np.sum(Sss*(1-Coh))/np.sum(Sss))
    cod_frac[i] =epsS
    if i==0:
        CoH = np.zeros((len(Coh),paramN))
        PhCoH = np.zeros((len(Coh),paramN))
        CrosS = np.zeros((len(Sxs),paramN))
        Spec = np.zeros((len(Sxx),paramN))
    CoH[:, i] = Coh    
    PhCoH[:,i] = Phcoh
    CrosS[:,i] = Sxs
    Spec[:,i] = Sxx

#%%
tab, tab1 = fn.get_nice_colors()
Das = np.logspace(-3.5,1.2,40)[::2]

#%%
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(10,8))
#ax = fig.add_subplot(311)
majory = 1.0
minory = 0.5
majorx = 5
minorx= 1
yla = 'Coherence'
#Nums = [0, 6, 20]
calculate_theory = True
lbs = ['Weak noise', 'Medium noise', 'Strong noise']
locs = [[0.63,0.22], [0.59, 0.68], 1]
for i,n in enumerate(nums):
    fn.get_rightframe(ax[i], majory=majory, minory = minory, majorx=majorx, minorx=minorx, xlabel = '', ylabel='')
    #ax[i].plot(freq, CoH[:,n], color=tab[2*i], linewidth = 4, label=lbs[i])# label=r"$D_{hom}=$"+str(DaS[n]/(10**np.floor(np.log10(DaS[n]))))[0:3]+'e'+str(int(np.log10(DaS[n]))))
    D = DaS[n] 
    if het_or_hom == 'hom' and theory:
        mask = (freq>0)*(freq<fmax)
        args={'D':D+sigma**2/(4.0*fmax), 'Ds':sigma**2/(4.0*fmax),  'mu':muA }
        try:        
            chi = np.load(ff+'chi_Da_'+str(D)+'.npy') 
            psd = np.load(ff+'spectrum_Da_'+str(D)+'.npy')
            theor = np.load(ff+'Coherence_Da_'+str(D)+'.npy') 
            FF = np.load(ff+'f_mask.npy')
                    
            #ax[i].plot(FF[1:], theor[1:], color=tab[2*i], linewidth =2.5)
            ax[i].plot(FF[1:], theor[1:],  color='k', linewidth =3.0)
        except IOError:
            print('Calculating chi')
            Ds = sigma**2/(4.0*fmax)
            args={'D':D+Ds, 'Ds':Ds,  'mu':muA }
            time = np.arange(0,T, dt)
            f = np.fft.fftfreq(len(time), dt)
            mask = (f>0)*(f<fmax)
            if calculate_theory:
                coh, chi, psd = fn.coherence_thesis(f[mask], thres, res, tref, Nneur, sigma, fmax, verbose=False,  **args)
                np.save(ff+'Coherence_Da_'+str(D), coh)
                np.save(ff+'spectrum_Da_'+str(D), psd)
                np.save(ff+'chi_Da_'+str(D), np.abs(chi))  
                np.save(ff+'f', f[mask])
                #ax[i].plot(f[mask][1:], coh[1:], color=tab[2*i], linewidth =2.5)
                ax[i].plot(f[mask][1:], coh[1:], '--', color='k', linewidth =3.0)
                continue
        ax[i].set_ylim([0, 1.02])
        ax[i].legend(fontsize=26,loc=locs[i])
plt.xlim([0, 10.1])
plt.xlabel('Frequency')
fig.text(-0.04, 0.5, 'Coherence', va='center', rotation='vertical', fontsize=40)

plt.tight_layout()
plt.savefig(folder_imagesE+'figs_3_9_1.eps', dpi=600)

#%%
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False, figsize=(10,8))
#ax = fig.add_subplot(311)
majory = 0.5
minory = 0.1
majorx = 5
minorx= 1
yla = 'Coherence'
#Nums = [0, 6, 20]
calculate_theory = True
lbs = ['Weak noise', 'Medium noise', 'Strong noise']
locs = [[0.63,0.22], [0.59, 0.68], 1]
for i,n in enumerate(nums):
    if i == 0:
        majory = 0.5;
        minory = 0.25;
    elif i == 1:
        majory = 0.2;
        minory = 0.1;
    elif i == 2:
        majory = 0.005;
        minory = 0.0025;
        
    fn.get_rightframe(ax[i], majory=majory, minory = minory, majorx=majorx, minorx=minorx, xlabel = '', ylabel='')
    #ax[i].plot(freq, Spec[:,n], color=tab[2*i], linewidth = 4, label=lbs[i])# label=r"$D_{hom}=$"+str(DaS[n]/(10**np.floor(np.log10(DaS[n]))))[0:3]+'e'+str(int(np.log10(DaS[n]))))
    D = DaS[n] 
    mask = (freq>0)*(freq<fmax)
    args={'D':D+sigma**2/(4.0*fmax), 'Ds':sigma**2/(4.0*fmax),  'mu':muA }
    chi = np.load(ff+'chi_Da_'+str(D)+'.npy') 
    psd = np.load(ff+'spectrum_Da_'+str(D)+'.npy')
    theor = np.load(ff+'Coherence_Da_'+str(D)+'.npy') 
    FF = np.load(ff+'f_mask.npy')
    ax[i].plot(FF[1:], (1./Nneur)*psd[1:]+((N-1)/N)*np.abs(chi[1:])**2*sigma**2/(2*fmax), '--', color='k', linewidth =3.0)
    ax[i].set_ylim([0, 1.2*np.max( (1./Nneur)*psd[1:]+((N-1)/N)*np.abs(chi[1:])**2*sigma**2/(2*fmax))])
    ax[i].legend(fontsize=26,loc='best')
plt.xlim([0, 10.1])
plt.xlabel('Frequency')
fig.text(0.01, 0.5, '$S_{xx}$', va='center', rotation='vertical', fontsize=40)

plt.tight_layout()
plt.savefig(folder_imagesE+'figs_3_9_1sxx.eps', dpi=600)
