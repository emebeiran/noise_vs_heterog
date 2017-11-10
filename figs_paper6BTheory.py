# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 10:40:45 2017

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
het_or_hom = 'het'

ff2 = '/home/mbeiran/Documents/MasterThesis/Programming/Coding_Fraction/Data/'
ff = '/home/mbeiran/Documents/MasterThesis/Programming/Coding_Fraction/PNG/'
ffE = '/home/mbeiran/Documents/MasterThesis/Programming/'
if not os.path.isdir(ff):
     os.makedirs(ff)
if not os.path.isdir(ffE):
    os.makedirs(ffE)
Das = [np.logspace(-3.5,1.2,40)[::2]]
DaS = np.zeros(25)
DaS[5:] = np.logspace(-3.5,1.2,40)[::2]

nN = 5
DAS = np.zeros(nN)
for i in range(nN):
    if i ==0:
        DAS[nN-1-i] = Das[0][0]*Das[0][0]/Das[0][1]
    else:
        DAS[nN-1-i] = DAS[nN-i]*Das[0][0]/Das[0][1]
DaS[0:5] = DAS
Das = Das
tab, tab1 = fn.get_nice_colors()

#Parameters of the network
Nneur = 300
T = 30.0
dt = 5e-3 #0.005
time = np.arange(0,T, dt)
N = len(time)

thres = 1.0
res = 0.0
tref = 0.1 #units of tau
tau = 1 #ms
to_ms = 20.0

df = 1.0/T 
f = np.fft.fftfreq(len(time), dt)               
fmax = 15.0
mask = (f>0)*(f<fmax)

# THEORY
het_or_hom='hom'
if het_or_hom =='hom':
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    plt.tight_layout()
    majory = 0.2
    minory = 0.1
    majorx = 2
    minorx= 1
    ylab = 'Coding fraction'

    fn.get_rightframe(ax, majory=majory, minory = minory, majorx=majorx, minorx=minorx, xlabel = 'Noise intensity $D_{hom}$', ylabel=ylab)
    count=0
    
    color_order = [0,2, 4, 10,  8, 5, 1, 8]
    
    for muA in [1.3,]:# 1.05]:
        for sigma in [ 0.5, 0.3, 0.2, 0.1, 0.05]:    
            for fmax in [15.0,]:
                
                mus = np.linspace(0.0, 30.0, 3000)
                
                seed = 696
                np.random.seed(seed)
                dt_spk = dt#0.005
                t_stim = np.arange(0.0,T, dt_spk)
                Da_het = 0.0
                    
                Ds = sigma**2/(4.0*fmax)
                paramN = len(DaS)
                CODtheor = np.zeros(paramN)
                for i,n in enumerate(np.arange(paramN)):
                    print(i)
                    args={'D':DaS[n]+Ds, 'Ds':Ds,  'mu':muA }
                    coh, chi, psd = fn.coherence_thesis(f[mask], thres, res, tref, Nneur, sigma, fmax, verbose=False,  **args)
                    theor = coh 
                    Sss = (sigma**2/(2*fmax))*np.ones(len(theor))
                    epsS = np.sqrt(np.sum(Sss*(1-theor))/np.sum(Sss))
                    CODtheor[i] =1-epsS

                ax.plot(DaS, CODtheor,  '--', linewidth = 5, color = tab[color_order[count]])
                count +=1
    ax.plot(DaS, -1*np.ones(len(DaS)),  '--', linewidth = 5, color = 'k', label='Theory')
    ax.scatter(DaS, -10*np.ones(len(DaS)), s=80, edgecolors='k', facecolors = 'gray', label = "Simulation")
                                
    ax.set_xscale('log')
    ax.set_xlim([9.99e-6, 2.0e1])
    ax.set_ylim([-0.01, 0.7])
    plt.tight_layout()
    if het_or_hom == 'hom':
        plt.legend(loc=1, fontsize=30)