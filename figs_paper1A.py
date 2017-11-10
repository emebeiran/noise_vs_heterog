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
plt.rc('font',  size = 28)



#%%
# Plot Fig 2.1.

tref = 0.1
Ds = [0.001, 0.01, 0.1]
mu_a = 1.3
Nb = 5000
MUs = []
tref = 0.1

count = 0
D = 1.0
pISI, timeISI, dens = fn.get_densISI(D, mu_a,  tol = 1e-8, tref=tref)
    
lngth = len(pISI)
pISIs = np.zeros((len(Ds), lngth))
pISIs[0,:]=pISI

dt = timeISI[1]-timeISI[0]
mus = np.linspace(0.0, 10.0, 1000)
dmu = mus[1]-mus[0]
Mus = np.hstack((mus, (mus[-1]+dmu)))-dmu/2.0

Pmu, muP, TT, pI = fn.get_theorPmu_d0(pISIs[0,:],timeISI, tref)
CDF = np.cumsum(Pmu*(muP*(muP-1))*dt/np.sum(Pmu*(muP*(muP-1))*dt))
interCDF = np.interp(Mus,muP, CDF)
interPDF = np.diff(interCDF/dmu)

#%%

#Define grid for subplots
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2.2], height_ratios=[3.2, 1])

#Create contour plot
fig = plt.figure(figsize=(8,9))
ymx = 2.0
ymn = 0.0

#Turn off all axes
#ax.axis('off')
#Create Y-marginal (bottom)8
axr = fig.add_subplot(gs[1,1],  xticks = [ 4, 8], yticks = [], frameon = True, xlim = (0,9), ylim=(0,0.11))# xlim = (0, 1.4*dy.max()), ylim=(ymin, ymax))
axr.plot( muP, Pmu/np.sum(Pmu*(muP*(muP-1))*dt), color = 'b', linewidth = 3, label=r'$P(\mu)$')
plt.setp(axr, 'ylim', reversed(plt.getp(axr, 'ylim')))
Idmax = np.argmax(Pmu/np.sum(Pmu*(muP*(muP-1))*dt))
Max_m = np.max(Pmu/np.sum(Pmu*(muP*(muP-1))*dt))
axr.plot( [muP[Idmax], muP[Idmax]], [0, Max_m],'-.k',linewidth=1.5)
axr.scatter( muP[Idmax],Max_m, 150, 'white', )
axr.legend(loc=1, fontsize = 'medium')

xmin, xmax = axr.get_xlim()
ymin, ymax = axr.get_ylim()

# removing the default axis on all sides:
for side in ['bottom','right','top','left']:
    axr.spines[side].set_visible(False)

# removing the axis ticks
#axr.set_xticks([]) # labels

axr.set_yticks([])
axr.set_xticklabels('')
axr.get_xaxis().set_tick_params(direction='in', width=1, length=10)
#axr.xaxis.set_ticks_position('none') # tick markers
axr.yaxis.set_ticks_position('none')
axr.spines["top"].set_visible(False)  
axr.spines["right"].set_visible(False) 
axr.xaxis.set_ticks_position('top')
# get width and height of axes object to compute
# matching arrowhead length and width
dps = fig.dpi_scale_trans.inverted()
bbox = axr.get_window_extent().transformed(dps)
width, height = bbox.width, bbox.height

# manual arrowhead width and length
hw = 1./10.*(ymax-ymin)
hl = 1./20.*(xmax-xmin)
lw = 1. # axis line width
ohg = 0.3 # arrow overhang

# compute matching arrowhead length and width
yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

# draw x and y axis
axr.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw,
         head_width=hw, head_length=hl, overhang = ohg,
         length_includes_head= True, clip_on = False, linewidth=1.)

axr.arrow( 0., ymax, 0.,  -ymax+ymin, fc='k', ec='k', lw = lw,
         head_width=yhw, head_length=-yhl, overhang = ohg,
         length_includes_head= True, clip_on = False)
         
#Create X-marginal (left)
axt = fig.add_subplot(gs[0,0],  frameon = True, ylim=(0,2), xlim = (0,ymx), xticks = [], yticks = [])# xticks=[],yticks=[],)# xlim = (xmin, xmax), ylim=(0, 1.4*dx.max()))
plt.setp(axt, 'xlim', reversed(plt.getp(axt, 'xlim')))
axt.plot( pISIs[0,:],timeISI, 'b', linewidth = 3, label=r"$Q(T)$")
axt.plot( timeISI*pISIs[0,:], timeISI, '--k', linewidth = 2, label=r"$T \cdot Q(T)$")
idmax = np.argmax(timeISI*pISIs[0,:])
max_m = np.max(timeISI*pISIs[0,:])
axt.plot( [0, max_m], [timeISI[idmax], timeISI[idmax]],'-.k',linewidth=1.5)
axt.scatter(max_m, timeISI[idmax], 100, 'white', )
axt.legend(fontsize = 'small')

axt.set_yticks([0, 1])
axt.set_yticklabels('')

#axt.yaxis.set_ticks_position('none')

axt.get_yaxis().set_tick_params(direction='in', width=1, length=10)

xmin, xmax = axt.get_xlim()
ymin, ymax = axt.get_ylim()

# removing the default axis on all sides:
for side in ['bottom','right','top','left']:
    axt.spines[side].set_visible(False)
    
axt.yaxis.set_ticks_position('right')
# removing the axis ticks
axt.set_xticks([]) # labels
#axt.set_yticks([])
#axt.xaxis.set_ticks_position('none') # tick markers
#axt.yaxis.set_ticks_position('none')

# get width and height of axes object to compute
# matching arrowhead length and width
dps = fig.dpi_scale_trans.inverted()
bbox = axt.get_window_extent().transformed(dps)
width, height = bbox.width, bbox.height

# manual arrowhead width and length
hw = 1./30.*(ymax-ymin)
hl = 1./8.*(xmax-xmin)
lw = 1. # axis line width
ohg = 0.3 # arrow overhang

# compute matching arrowhead length and width
yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

# draw x and y axis
axt.arrow(xmax, 0, -xmax+xmin, 0., fc='k', ec='k', lw = lw,
         head_width=hw, head_length=-hl, overhang = ohg,
         length_includes_head= True, clip_on = False)

axt.arrow( 0., ymin, 0.,  ymax-ymin, fc='k', ec='k', lw = lw,
         head_width=yhw, head_length=yhl, overhang = ohg,
         length_includes_head= True, clip_on = False)
         

ax = fig.add_subplot(gs[0,1], xlim = (0,9), ylim=(ymn,ymx))
fn.get_rightframe(ax, majorx=4.0, minorx=100.0, majory=1.0, minory=10.)

ax.plot(muP, TT, color='k', linewidth=3)

ax.plot([0, muP[Idmax]], [timeISI[idmax], timeISI[idmax]],'-.k',linewidth=1.5)
ax.plot([muP[Idmax], muP[Idmax]], [0, timeISI[idmax]],'-.k',linewidth=1.5)
ax.scatter(muP[Idmax], timeISI[idmax], 100, 'white', )
ax.set_xlabel(r'mean input $\mu$', fontsize='medium')
ax.xaxis.labelpad = 0
ax.set_ylabel(r"interspike interval $ T$", fontsize='medium')
ax.tick_params(axis='both', which='major', labelsize='small')
ax.get_xaxis().set_tick_params(direction='in', width=1, length=10)
ax.get_yaxis().set_tick_params(direction='in', width=1, length=10)
gs.tight_layout(fig,w_pad=0.5, h_pad=0.25)
plt.savefig('fig2_1.pdf', dpi=600)
