# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 12:16:59 2016

@author: mbeiran
"""
import numpy as np
import f_network as fn
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.close('all')
plt.rc('text', usetex=True)
plt.rcParams['text.usetex'] = True 
plt.rcParams.update({'figure.autolayout': True})
plt.rc('font', size = 28)

#%%

#==============================================================================
#   Parameters 
#==============================================================================

thres = 1.0
res = 0.0
tref = 0.1 #units of tau
tau = 1 #ms
to_ms = 20.0

muA = 1.5
Da = 0.01#0.05
Ds = 0.0
#Tools
T = 60.0
dt = 0.05
time = np.arange(0,T, dt)
N = len(time)


df = 1.0/T 
f = np.fft.fftfreq(len(time))
f = f

mus = np.linspace(0.0, 30.0, 3000)

seed = 694
np.random.seed(seed)


Nneur = 300
dt = 0.0005 #units of tau (20ms) All times are in units of tau
dt_spk = 0.01                                                                                                                                                                                       


#%%
#==============================================================================
#       PDF of mus and P_ISI of homogeneous
#==============================================================================
muA = 1.8
Da = 0.1
P_mu, tme, pISI, muS, P1_mu, TTs, PItheor = fn.get_hetmus_nonoise(mus, muA, Da, tref, tol=1e-10, dt=1e-4)
#%%
#==============================================================================
#               Choose random mus for heterogeneous 
#==============================================================================
T = 100
x0 = 0.0*np.zeros((Nneur))
fmax = (0.1/dt)
mus_pop = np.random.choice(mus, size=Nneur, p=P_mu/np.sum(P_mu))
mus_hom = np.ones(len(mus_pop))*muA

#%%
spks_hom, t_spk, y_model, stim1, t_v, initial_cond, initial_cond1 =\
fn.euler_maru_withnoise(fn.ffun2, Ds, Da, mus_hom, dt, dt_spk,\
T, Nneur, fmax, T_prerun=10.0, t_ref = tref,example=True,verbose=False)

#%%
Da_het = 0.0
spks_het, t_spk, y_model, stim1, t_v, initial_cond, initial_cond1 =\
fn.euler_maru_withnoise(fn.ffun2, Ds, Da_het, mus_pop, dt, dt_spk,\
T, Nneur, fmax, T_prerun=10.0, t_ref = tref,example=True,verbose=False)
#%%
#==============================================================================
#   Count ISI
#==============================================================================
ISI_hom = fn.giveISIs(spks_hom, dt_spk)
ISI_het = fn.giveISIs(spks_het, dt_spk)

#%%
plot_comparison = True
if plot_comparison:
    fig = plt.figure(figsize=(6.5,5.75))
    fig.set_rasterized(True)
    majory = 1.0
    minory = 0.5
    ax = fig.add_subplot(111)
    fn.get_rightframe(ax, xlabel='', ylabel='',majory=majory, minory = minory, fontsize=25, labelsize= 25)
    #ax.hist(ISI_hom,40, alpha=1.0, color = 'k', lw = 2.0, fc = 'white',  normed=True,histtype='stepfilled',  rasterized=False, edgecolor='black')  
    #ax.hist(ISI_het,40,alpha=1.0, lw = 2.0, color='k', fc = 'white', normed=True,histtype='stepfilled', rasterized=False)    
    ax.hist(ISI_hom,40,  lw = 2.0, alpha = 0.5, edgecolor='k', fc = 'blue',  normed=True,histtype='stepfilled', label='Homog. pop.', rasterized=False,)  
    ax.hist(ISI_het,40, lw = 2.0, alpha=0.4, edgecolor='k', fc = 'green', normed=True,histtype='stepfilled', label='Heter. pop.', rasterized=False)
    ax.hist(ISI_hom,40,  lw = 3.0,  edgecolor='b', fc = 'None',  normed=True,histtype='stepfilled',  rasterized=False,)  
    ax.hist(ISI_het,40, lw = 3.0, edgecolor='g', fc = 'None', normed=True,histtype='stepfilled',  rasterized=False)
    ax.axhline(linewidth=2.5, color='k')
    ax.axvline(linewidth=2.5, color='k')
    ax.plot(tme, pISI, 'k', linewidth=4.5, label=r'$\rho_{ISI}(T)$')
    ax.set_xlim([0.0, 2.8])
    ax.set_ylim([0.0, 2.0])
    ax.xaxis.set_tick_params(width=2.0)
    ax.yaxis.set_tick_params(width=2.0)
    ax.set_yticklabels('')    
    ax.set_xticklabels('')
    plt.legend(loc=1, fontsize=25)
    plt.savefig('figs_2_3_1.eps', dpi=600, rasterized=True)    
    
#%%
plot_rasters = True

if plot_rasters: 
    majory = 150
    minory = 50
    majorx = 5
    minorx= 1
    fig = plt.figure(figsize=(9,8))
    
    
    ax = fig.add_subplot(212)
    fn.get_rightframe(ax, majory=majory, minory = minory, majorx=majorx, minorx=minorx, xlabel = 'time', fontsize=25, labelsize= 25)
    mask = np.argsort(np.sum(spks_het, axis=0))
    
    spks_het_ord= spks_het[:,mask]
    fn.rasterplot(spks_het_ord[t_spk<11.,:],t_spk[t_spk<11.], ax, coL = 'g', alpha = 0.5, S=3.0, rr = False)
    textstr='Heterogeneous pop.'
    # these are matplotlib.patch.Patch properties
    #props = dict(boxstyle='square', facecolor='white', alpha=0.9)
    #ax.text(0.15, 0.9, textstr, transform=ax.transAxes, fontsize=25,
    #        verticalalignment='top', bbox=props)
    
    ax.set_xlim([0,10.01])
    ax.invert_yaxis()
    
    ax.set_ylabel(r'$\#$ neuron', fontsize=25)
    ax = fig.add_subplot(211)
    fn.get_rightframe(ax, majory=majory, minory = minory, majorx=majorx, minorx=minorx, fontsize=25, labelsize= 25)
    mask = np.argsort(np.sum(spks_hom, axis=0))
    
    spks_hom_ord= spks_hom[:,mask]
    fn.rasterplot(spks_hom_ord[t_spk<11.,:],t_spk[t_spk<11.], ax, coL = 'b', alpha = 0.5, S=3.0, rr = False)
    ax.invert_yaxis()
    textstr='Homogeneous pop.'
    # these are matplotlib.patch.Patch properties
    #props = dict(boxstyle='square', facecolor='white', alpha=0.9)
    #ax.text(0.15, 0.9, textstr, transform=ax.transAxes, fontsize=25,
    #        verticalalignment='top', bbox=props)
    
    #ax.set_xticks([])
    ax.set_xlim([0,10.01])
    ax.set_ylabel(r'$\#$ neuron', fontsize=25)
    plt.savefig('figs_2_3_2.eps', dpi=600, rasterized=True)

