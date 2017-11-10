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
plt.close('all')
plt.rc('text', usetex=True)
plt.rcParams['text.usetex'] = True 
#plt.rcParams.update({'figure.autolayout': True})
plt.rc('font', family='serif', size = 16)
from matplotlib.ticker import MultipleLocator
#%%
muA = 1.3
sigma = 0.3
fmax = 4.0
    
Nneur = 300

HorHH = ['hom' ,'het']
T = 30.0
dt = 5e-3 #0.005
time = np.arange(0,T, dt)
N = len(time)
Stim = fs.gen_band_limited(T, dt, fmax)
    
seed = 696
np.random.seed(seed)

trials_s = 11
nums = [1, 13, 23]

#==============================================================================
#   Parameters 
#==============================================================================
thres = 1.0
res = 0.0
tref = 0.1 #units of tau
tau = 1 #ms
to_ms = 20.0

df = 1.0/T 
f = np.fft.fftfreq(len(time))
f = f

mus = np.linspace(0.0, 30.0, 3000)



dt_spk = dt#0.005
t_stim = np.arange(0.0,T, dt_spk)
mus_hom = np.ones(Nneur*trials_s)*muA
Da_het = 0.0
Ds = 0.5*sigma**2

strss = ['Low', 'Medium', 'High']

for  het_or_hom in HorHH:
        
        
    trials = 1
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
    for I,num in enumerate(nums):        
        Da = DaS[num]            
              
        if het_or_hom == 'hom':
            spks_hom, t_spk, y_model, stim1, t_v, initial_cond, initial_cond1 =\
                fn.euler_maru_withnoise(fn.ffun2, Ds, Da, mus_hom, dt, dt_spk,\
                T, trials_s*Nneur, fmax, T_prerun=30.0, t_ref = tref,example=True,verbose=False, Stim=Stim)
            stim_het = np.interp(t_spk, t_v, stim1)
                    
            OUT = []
            for t_s in range(trials_s):
                
                stim_het = np.interp(t_spk, t_v, stim1)
                out_het = np.mean(spks_hom[:,t_s*Nneur:(t_s+1)*Nneur], axis=-1)/dt_spk
                if t_s == trials_s-11:
                    OUT = out_het
                elif t_s>trials_s-11:
                    OUT = np.vstack([OUT, out_het])
                
        
            out = OUT
            spks = spks_hom[:,-300:]

        elif het_or_hom == 'het':   
            OUT = []        
            string = '/home/mbeiran/Documents/MasterThesis/Programming/mus_pop_Da_'+str(Da)+'.npy'
            
            mus_largepop = np.load(string)
            mus_pop_mask = np.random.random_integers(0, 20000-1, trials_s*Nneur)
            mus_pop = mus_largepop[mus_pop_mask]
            spks_het, t_spk, y_model, stim1, t_v, initial_cond, initial_cond1 =\
            fn.euler_maru_withnoise(fn.ffun2, Ds, Da_het, mus_pop, dt, dt_spk,\
            T, trials_s*Nneur, fmax, T_prerun=5.0, t_ref = tref,example=True,verbose=False, Stim=Stim)
            stim_het = np.interp(t_spk, t_v, stim1)
            
            for t_s in range(trials_s):
                stim_het = np.interp(t_spk, t_v, stim1)
                out_het = np.mean(spks_het[:,t_s*Nneur:(t_s+1)*Nneur], axis=-1)/dt_spk
                if t_s == trials_s-11:
                    OUT = out_het	
                elif t_s>trials_s-11:
                    OUT = np.vstack([OUT, out_het])

            out = OUT
            spks = spks_het[:,-300:]
        
        stim = Stim
        window = 0.05
        
            
        t_mask = (t_spk>23.5)*(t_spk<29.5)
        TT = t_spk[t_mask]-t_spk[t_mask][0]
        fsiz = 20
        fig, (ax, ax2,) = plt.subplots(2, 1, sharex=True, sharey=False, 
                                                squeeze = True, figsize=(7.5,4))
        fig.subplots_adjust(hspace=0.00)
        
        #Asis 2
        window = 0.5
        
        frate = (fn.firingrate_cond1(out[-1,:], t_spk, window))
        frate2 = (fn.firingrate_cond1(out[-2,:], t_spk, window))    
        frate3 = (fn.firingrate_cond1(out[-3,:], t_spk, window))
        
        #Asis 2
        window = 0.1
        frate = (fn.firingrate_cond1(out[-1,:], t_spk, window))
        
        frate2 = (fn.firingrate_cond1(out[-2,:], t_spk, window))
        
        frate3 = (fn.firingrate_cond1(out[-3,:], t_spk, window))
        
        mm = np.mean(spks, axis=1)[t_mask]
        ax2.plot(TT, frate[t_mask]/Nneur, linewidth=2)
        ax2.plot(TT, frate2[t_mask]/Nneur, linewidth=1)
        ax2.plot(TT, frate3[t_mask]/Nneur, linewidth=1)

        ax2.tick_params(axis='both',  labelsize=12, size=12)
        ax2.set_xticks([])
        ax2.set_ylabel('Output', fontsize= fsiz)
        ax2.tick_params(axis='both',  labelsize=12, size=12)
        ax2.xaxis.set_tick_params(labelsize=fsiz)
        ax2.yaxis.set_tick_params(labelsize=fsiz)
        
        majory = np.round(np.max(frate[t_mask]/Nneur)/2.-np.min(frate[t_mask]/Nneur)/2., -np.int(np.floor(np.log10(np.abs(np.max(frate[t_mask]/Nneur)/2.-np.min(frate[t_mask]/Nneur)/2.)))))
        minory = majory/2.
        minorLocator = MultipleLocator(minory)
        majorLocator = MultipleLocator(majory)
        ax2.yaxis.set_major_locator(majorLocator)
        ax2.yaxis.set_minor_locator(minorLocator)
        ymin, ymax = ax2.get_ylim()
        #ax2.set_ylim([ymin+0.05, ymax-0.05])
        
    
            #Axis 1
        ax.tick_params(axis='both',  labelsize=12, size=12)    
        mascara = np.argsort(np.mean(spks[t_mask,:],axis=0))
        fn.rasterplot(spks[t_mask,:][:,mascara[::-1]],TT, ax)
        ax.set_yticks([ 100, 200, 300])
        ax.tick_params(axis='both',  labelsize=12, size=12)
        ax.set_ylabel('Neuron', fontsize = fsiz)
#                ax.text(1.5, 150, r"$D_{hom}=$"+str(Da/(10**np.floor(np.log10(Da))))[0:3]+'e'+str(int(np.log10(Da))), ha = 'center', va = 'center',
#                            fontsize = fsiz, bbox={'facecolor':'white', 'pad':10})
        ax.text(1.5, 150, strss[I], ha = 'center', va = 'center',
                    fontsize = fsiz, bbox={'facecolor':'white', 'pad':10})
                        
        ax.yaxis.set_tick_params(labelsize=fsiz)
    
        minory = 50
        majory = 100
        minorLocator = MultipleLocator(minory)
        majorLocator = MultipleLocator(majory)
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_minor_locator(minorLocator)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim([1.01*ymin+2, 0.99*ymax-2])
    
        string = 'Example_'
        if I ==0:
            string+='Ohne'+'weak'+het_or_hom
        elif I ==1:
            string +='Ohne'+ 'medium'+het_or_hom
        elif I ==2:
            string +='Ohne'+ 'strong'+het_or_hom
        folder = '/home/mbeiran/Documents/MasterThesis/Programming/'
       
        #Axis 3+'_1.eps', dpi=200)
        #plt.savefig(folder+string+'_1.png', dpi=200)
        plt.savefig(folder+string+'_1.eps', dpi=200)
            
#%%
window = 0.05

fsiz = 20
fig, ( ax3) = plt.subplots(1, 1, sharex=True, sharey=False, 
                                        squeeze = True, figsize=(7.5,2))
fig.subplots_adjust(hspace=0.00)


##    #Axis 3
ax3.plot(TT, sigma*stim[t_mask]/np.std(stim), 'k', linewidth=2)
from scipy import signal
b, a = signal.butter(8, 0.04)
why = signal.filtfilt(b, a, sigma*stim/np.std(stim), padlen=150)
#ax3.plot(TT, why[t_mask], 'k', linewidth = 2)    
ax3.set_ylabel('Signal (a.u.)', fontsize= fsiz)
 
ax3.set_xticks([0, 1, 2, 3, 4, 5])#, ['0', '1', '2', '3', '4', '5'])  
ax3.set_yticks([-0.1,0.0, 0.1])
ax3.set_xlabel('Time [ms]', fontsize= fsiz)
ax3.tick_params(axis='both',  labelsize=12, size=12)
ax3.xaxis.set_tick_params(labelsize=fsiz)
ax3.yaxis.set_tick_params(labelsize=fsiz)

minory = 0.25
majory = 0.5
minorLocator = MultipleLocator(minory)
majorLocator = MultipleLocator(majory)
ax3.yaxis.set_major_locator(majorLocator)
ax3.yaxis.set_minor_locator(minorLocator)
ymin, ymax = ax3.get_ylim()
ax3.set_ylim([ymin+0.01, ymax-0.01])
#plt.tight_layout()
string = 'Signal_'+het_or_hom 

folder = '/home/mbeiran/Documents/MasterThesis/Programming/'
   
#plt.savefig(folder+string+'_1.png', dpi=200)
plt.savefig(folder+string+'_1.eps', dpi=200)