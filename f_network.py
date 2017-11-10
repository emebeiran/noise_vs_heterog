# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 17:15:45 2015

@author: mbeiran
"""
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.signal import convolve2d
from scipy.special import erf
from math import erfc
#from math import erf
import sympy.mpmath as mp
import f_noise as fs

def euler_maru_withnoise(ffun2, sigma, Da, mus, dt, dt_spk, T, Nneur, fmax, v_thres = 1.0, \
                        v_res = 0.0,T_prerun=4.0, t_ref = 2.0,example=True,verbose=False, Stim='not_given'):
    '''
    Solves the Langevin equation of the form
    dX(t) = f(X(t))dt + g(X(t))dW(t)
    with threshold rule (one simulation)

    Runs a first prerun of 10% of the running time, in order to start at 
    equilibrium conditons.

    INPUT
    -----
    ffun: function f -non stochastical part. usually drift-
    gfun: function that modulates white noise.
    x0: initial conditions
    dt: discretization time of the integration method
    dt_spk: discretization time of the spike recording
    N: number of time steps in time array   
    args: dictionary including the constants of the functions
    v_thres: spiking threshold
    v_res: reset value of the potential
    t_ref: refractory period
    example: boolean. Say if you want to have the voltage trace of one neuron
    
    OUTPUT
    -----
    y: numpy array with the solution of just one neuron, as an example. 
        if example is False, returns 0   
    spks: np.array with the spike trains of all neurons. 1stdim: time. 2nd: neuron    
    initial_cond: initial conditions
    '''

    ref_int = np.floor(t_ref/dt)    
    sqrtdt = np.sqrt(dt)
    neurs = Nneur
    t_spk = np.arange(0.0,T, dt_spk)
    t_v = np.arange(0.0,T, dt)
    y = np.zeros(Nneur)
    N = len(t_v)
    buffr = np.zeros((ref_int,neurs))
    spks = np.zeros((int(np.floor(T/dt_spk)),neurs))
    noise = np.random.randn(N,neurs)
    stim1 = fs.gen_band_limited(T, dt, fmax)

#==============================================================================
#     PRE-RUN
#==============================================================================
    initial_cond = 0.1*np.random.rand(neurs)-0.1         

    y = initial_cond        
    buffer_counter = 0 
    spk_register = 0  #Check when new recording bin has been reached
    spk_timeidx = 0    
    
    N_prerun = int(np.round(T_prerun/dt))
    for i in range(1,N_prerun):                         
        yp1 =  y + dt*ffun2(y,mus)+ sqrtdt*np.sqrt(2*(Da))*noise[i-1,:]\
                + dt*sigma*stim1[i-1]

        
        mask_thres = np.sum(buffr,axis=0)>0.0
        yp1[mask_thres] = v_res        
        mask_spk = yp1>v_thres  
        buffr[buffer_counter,:] = mask_spk
        spks[spk_timeidx,mask_spk] = 1
        
        buffer_counter += 1
        if buffer_counter == ref_int:
            buffer_counter = 0
            
        mask_thres2 = yp1>v_thres
        yp1[mask_thres2] = v_res
            
        spk_register += 1
        if spk_register>=(dt_spk/dt):
            
            spk_register = 0
            spk_timeidx +=1
        y= yp1
        
    if verbose:
        print('Pre-run integration completed')
        print('')

    initial_cond = y
    spk_timeidx = 0 
    spk_register = 0
#==============================================================================
#   REAL RUN
#==============================================================================
    #Save one voltage trace
    spks = np.zeros((int(np.floor(T/dt_spk)),neurs))
    if example:    
        y_model = np.zeros(N)
        y_model[0] = y[0]
    else:
        y_model = 0
    if Stim =='not_given':
        stim = fs.gen_band_limited(T, dt, fmax)
    else:
        stim = Stim 
    for i in range(1,N):

        yp1 = y + dt*ffun2(y,mus)+ sqrtdt*np.sqrt(2*(Da))*noise[i-1,:]\
                + dt*sigma*stim1[i-1]

        mask_thres = np.sum(buffr,axis=0)>0.0
        yp1[mask_thres] = v_res

        mask_spk = yp1>v_thres        
        buffr[buffer_counter,:] = mask_spk

        spks[spk_timeidx,mask_spk] = 1
        
        buffer_counter += 1
        if buffer_counter == ref_int:
            buffer_counter = 0
            
        if example:
            y_model[i] = yp1[0]
            
        spk_register += 1
        if spk_register>=(dt_spk/dt):                
            spk_register = 0
            spk_timeidx +=1
        y= yp1
    if verbose:
        print('Langevin integration completed')
        print('')
    initial_cond1 = y
    
    return spks, t_spk, y_model, stim, t_v, initial_cond, initial_cond1

def euler_maru_withnoise2(ffun2, gfun, stim, mus, x0, dt, dt_spk, N, buffr, args, v_thres = 1.0, \
                        v_res = 0.0, t_ref = 2.0,example=True,verbose=True):
    '''
    Solves the Langevin equation of the form
    dX(t) = f(X(t))dt + g(X(t))dW(t)
    with threshold rule (one simulation)

    Runs a first prerun of 10% of the running time, in order to start at 
    equilibrium conditons.

    INPUT
    -----
    ffun: function f -non stochastical part. usually drift-
    gfun: function that modulates white noise.
    x0: initial conditions
    dt: discretization time of the integration method
    dt_spk: discretization time of the spike recording
    N: number of time steps in time array   
    args: dictionary including the constants of the functions
    v_thres: spiking threshold
    v_res: reset value of the potential
    t_ref: refractory period
    example: boolean. Say if you want to have the voltage trace of one neuron
    
    OUTPUT
    -----
    y: numpy array with the solution of just one neuron, as an example. 
        if example is False, returns 0   
    spks: np.array with the spike trains of all neurons. 1stdim: time. 2nd: neuron    
    initial_cond: initial conditions
    '''

    ref_int = np.floor(t_ref/dt)    
    sqrtdt = np.sqrt(dt)
    neurs = len(x0)
    D = args['D']
    buffr = np.zeros((ref_int,neurs))
    spks = np.zeros((int(np.floor((N*dt)/dt_spk))+1,neurs))
    #PRE RUN    
    buffer_counter = 0
    #If there is no refractory period:
    if buffr.size == 0:
        initial_cond = np.random.rand(len(x0))
        y = initial_cond        
        #Save one voltage trace
        buffer_counter = 0 
        spk_register = 0  #Check when new recording bin has been reached
        spk_timeidx = 0    
        
        
        for i in range(1,N):
            yp1 = y + dt*ffun2(y,mus)+ sqrtdt*np.sqrt(2*D)*stim[i-1,:]
            
            mask_thres = np.sum(buffr,axis=0)>0.0
            yp1[mask_thres] = v_res
            
            mask_spk = yp1>v_thres        
    
            spks[spk_timeidx,mask_spk] = 1
            mask_thres2 = yp1>v_thres
                        
            yp1[mask_thres2] = v_res
                
            spk_register += 1
            if spk_register>=(dt_spk/dt):
                
                spk_register = 0
                spk_timeidx +=1
            y= yp1
            
        if verbose:
            print('Langevin integration completed')
            print('')
        
    else:
        print('down')
        initial_cond = np.random.rand(len(x0))
        y = initial_cond       
        #Save one voltage trace
        if example:    
            y_model = np.zeros(N)
            y_model[0] = y[0]
        else:
            y_model = 0
        buffer_counter = 0 
        spk_register = 0  #Check when new recording bin has been erreicht
        spk_timeidx = 0    
        for i in range(1,N):
            yp1 = y + dt*ffun2(y,mus)+ sqrtdt*np.sqrt(2*D)*stim[i-1,:]
            mask_thres = np.sum(buffr,axis=0)>0.0
            yp1[mask_thres] = v_res

    
            mask_spk = yp1>v_thres        
            buffr[buffer_counter,:] = mask_spk
    
            spks[spk_timeidx,mask_spk] = 1
            
            buffer_counter += 1
            if buffer_counter == ref_int:
                buffer_counter = 0
                
            if example:
                y_model[i] = yp1[0]
    
            spk_register += 1
            if spk_register>=(dt_spk/dt):
                print(spk_register)                
                spk_register = 0
                spk_timeidx +=1
            y= yp1
        if verbose:
            print('Langevin integration completed')
            print('')
    initial_cond = y
    
    #Save one voltage trace
    if example:    
        y_model = np.zeros(N)
        y_model[0] = y[0]
    else:
        y_model = 0
    buffer_counter = 0 
    spk_register = 0  #Check when new recording bin has been erreicht
    spk_timeidx = 0    
    for i in range(1,N):
        yp1 = y + dt*ffun2(y,mus)+ sqrtdt*np.sqrt(2*D)*stim[i-1,:]
        mask_thres = np.sum(buffr,axis=0)>0.0
        yp1[mask_thres] = v_res


        mask_spk = yp1>v_thres        
        buffr[buffer_counter,:] = mask_spk

        spks[spk_timeidx,mask_spk] = 1
        
        buffer_counter += 1
        if buffer_counter == ref_int:
            buffer_counter = 0
            
        if example:
            y_model[i] = yp1[0]

        spk_register += 1
        if spk_register>=(dt_spk/dt):
            print(spk_register)                
            spk_register = 0
            spk_timeidx +=1
        y= yp1
    if verbose:
        print('Langevin integration completed')
        print('')
    initial_cond = y
    return y_model, spks, initial_cond


def preeuler_maru_withnoise(ffun2, gfun, stim, mus, x0, dt, dt_spk, N, args, v_thres = 1.0, \
                        v_res = 0.0, t_ref = 2.0,example=True,verbose=True):
    '''
    Solves the Langevin equation of the form
    dX(t) = f(X(t))dt + g(X(t))dW(t)
    with threshold rule (one simulation)

    Runs a first prerun of 10% of the running time, in order to start at 
    equilibrium conditons.

    INPUT
    -----
    ffun: function f -non stochastical part. usually drift-
    gfun: function that modulates white noise.
    x0: initial conditions
    dt: discretization time of the integration method
    dt_spk: discretization time of the spike recording
    N: number of time steps in time array   
    args: dictionary including the constants of the functions
    v_thres: spiking threshold
    v_res: reset value of the potential
    t_ref: refractory period
    example: boolean. Say if you want to have the voltage trace of one neuron
    
    OUTPUT
    -----
    y: numpy array with the solution of just one neuron, as an example. 
        if example is False, returns 0   
    spks: np.array with the spike trains of all neurons. 1stdim: time. 2nd: neuron    
    initial_cond: initial conditions
    '''

    ref_int = np.floor(t_ref/dt)    
    sqrtdt = np.sqrt(dt)
    neurs = len(x0)
    D = args['D']
    y = x0
    buffr = np.zeros((ref_int,neurs))
    spks = np.zeros((int(np.floor((N*dt)/dt_spk))+1,neurs))
    buffer_counter = 0 
    #Find stationary solution
    if buffr.size == 0:
        print('up')
        initial_cond = np.random.rand(len(x0))
        y = initial_cond        
        #Save one voltage trace
        if example:    
            y_model = np.zeros(N)
            y_model[0] = y[0]
        else:
            y_model = 0
        buffer_counter = 0 
        spk_register = 0  #Check when new recording bin has been reached
        spk_timeidx = 0    
        
        
        for i in range(1,N):
            yp1 = y + dt*ffun2(y,mus)+ sqrtdt*np.sqrt(2*D)*stim[i-1,:]
            
            mask_thres = np.sum(buffr,axis=0)>0.0
            yp1[mask_thres] = v_res
            
            mask_spk = yp1>v_thres        
    
            spks[spk_timeidx,mask_spk] = 1
            mask_thres2 = yp1>v_thres
                        
            yp1[mask_thres2] = v_res
            if example:
                y_model[i] = yp1[0]
                
            spk_register += 1
            if spk_register>=(dt_spk/dt):
                
                spk_register = 0
                spk_timeidx +=1
            y= yp1
            
        if verbose:
            print('Langevin integration completed')
            print('')
        
    else:
        print('down')
        initial_cond = np.random.rand(len(x0))
        y = initial_cond       
        #Save one voltage trace
        if example:    
            y_model = np.zeros(N)
            y_model[0] = y[0]
        else:
            y_model = 0
        buffer_counter = 0 
        spk_register = 0  #Check when new recording bin has been erreicht
        spk_timeidx = 0    
        for i in range(1,N):
            yp1 = y + dt*ffun2(y,mus)+ sqrtdt*np.sqrt(2*D)*stim[i-1,:]
            mask_thres = np.sum(buffr,axis=0)>0.0
            yp1[mask_thres] = v_res

    
            mask_spk = yp1>v_thres        
            buffr[buffer_counter,:] = mask_spk
    
            spks[spk_timeidx,mask_spk] = 1
            
            buffer_counter += 1
            if buffer_counter == ref_int:
                buffer_counter = 0
                
            if example:
                y_model[i] = yp1[0]
    
            spk_register += 1
            if spk_register>=(dt_spk/dt):
                print(spk_register)                
                spk_register = 0
                spk_timeidx +=1
            y= yp1
        if verbose:
            print('Langevin integration completed')
            print('')
    initial_cond = y
    return y_model, spks, initial_cond, buffr


def euler_maru_noiseless(ffun2, gfun, stim, mus, x0, dt, dt_spk, N, args, v_thres = 1.0, \
                        v_res = 0.0, t_ref = 2.0,example=True,verbose=True):
    '''
    Solves the Langevin equation of the form
    dX(t) = f(X(t))dt + g(X(t))dW(t)
    with threshold rule (one simulation)

    Runs a first prerun of 10% of the running time, in order to start at 
    equilibrium conditons.

    INPUT
    -----
    ffun: function f -non stochastical part. usually drift-
    gfun: function that modulates white noise.
    x0: initial conditions
    dt: discretization time of the integration method
    dt_spk: discretization time of the spike recording
    N: number of time steps in time array   
    args: dictionary including the constants of the functions
    v_thres: spiking threshold
    v_res: reset value of the potential
    t_ref: refractory period
    example: boolean. Say if you want to have the voltage trace of one neuron
    
    OUTPUT
    -----
    y: numpy array with the solution of just one neuron, as an example. 
        if example is False, returns 0   
    spks: np.array with the spike trains of all neurons. 1stdim: time. 2nd: neuron    
    initial_cond: initial conditions
    '''

    ref_int = np.floor(t_ref/dt)    
    sqrtdt = np.sqrt(dt)
    neurs = len(x0)
    D = args['D']
    y = x0
    buffr = np.zeros((ref_int,neurs))
    spks = np.zeros((int(np.floor((N*dt)/dt_spk))+1,neurs))
    buffer_counter = 0 
    #Find stationary solution
    if buffr.size == 0:
        print('up')
        initial_cond = np.random.rand(len(x0))
        y = initial_cond        
        #Save one voltage trace
        if example:    
            y_model = np.zeros(N)
            y_model[0] = y[0]
        else:
            y_model = 0
        buffer_counter = 0 
        spk_register = 0  #Check when new recording bin has been reached
        spk_timeidx = 0    
        
        
        for i in range(1,N):
            yp1 = y + dt*ffun2(y,mus)+ sqrtdt*np.sqrt(2*D)*stim[i-1]
            
            mask_thres = np.sum(buffr,axis=0)>0.0
            yp1[mask_thres] = v_res
            
            mask_spk = yp1>v_thres        
    
            spks[spk_timeidx,mask_spk] = 1
            mask_thres2 = yp1>v_thres
                        
            yp1[mask_thres2] = v_res
            if example:
                y_model[i] = yp1[0]
                
            spk_register += 1
            if spk_register>=(dt_spk/dt):
                
                spk_register = 0
                spk_timeidx +=1
                print('h',np.sum(mask_spk))
            y= yp1
            
        if verbose:
            print('Langevin integration completed')
            print('')
            
    else:
        print('down')
        initial_cond = np.random.rand(len(x0))
        y = initial_cond       
        #Save one voltage trace
        if example:    
            y_model = np.zeros(N)
            y_model[0] = y[0]
        else:
            y_model = 0
        buffer_counter = 0 
        spk_register = 0  #Check when new recording bin has been erreicht
        spk_timeidx = 0    
        for i in range(1,N):
            yp1 = y + dt*ffun2(y,mus)+ sqrtdt*np.sqrt(2*D)*stim[i-1]
            mask_thres = np.sum(buffr,axis=0)>0.0
            yp1[mask_thres] = v_res

    
            mask_spk = yp1>v_thres        
            buffr[buffer_counter,:] = mask_spk
    
            spks[spk_timeidx,mask_spk] = 1
            
            buffer_counter += 1
            if buffer_counter == ref_int:
                buffer_counter = 0
                
            if example:
                y_model[i] = yp1[0]
    
            spk_register += 1
            if spk_register>=(dt_spk/dt):
                print(spk_register)                
                spk_register = 0
                spk_timeidx +=1
            y= yp1
        if verbose:
            print('Langevin integration completed')
            print('')
    return y_model, spks, initial_cond


def euler_maru_noise(ffun, gfun, stim, noise, c, x0, dt, dt_spk, N, args, v_thres = 1.0, \
                        v_res = -0.1, t_ref = 2.0,example=True,verbose=True):
    '''
    Solves the Langevin equation of the form
    dX(t) = f(X(t))dt + g(X(t))dW(t)
    with threshold rule (one simulation)

    Runs a first prerun of 10% of the running time, in order to start at 
    equilibrium conditons.

    INPUT
    -----
    ffun: function f -non stochastical part. usually drift-
    gfun: function that modulates white noise.
    x0: initial conditions
    dt: discretization time of the integration method
    dt_spk: discretization time of the spike recording
    N: number of time steps in time array   
    args: dictionary including the constants of the functions
    v_thres: spiking threshold
    v_res: reset value of the potential
    t_ref: refractory period
    example: boolean. Say if you want to have the voltage trace of one neuron
    
    OUTPUT
    -----
    y: numpy array with the solution of just one neuron, as an example. 
        if example is False, returns 0   
    spks: np.array with the spike trains of all neurons. 1stdim: time. 2nd: neuron    
    initial_cond: initial conditions
    '''

    ref_int = np.floor(t_ref/dt)    
    sqrtdt = np.sqrt(dt)
    neurs = len(x0)
    D = args['D']
    y = x0
    buffr = np.zeros((ref_int,neurs))
    spks = np.zeros((int(np.floor((N*dt)/dt_spk))+1,neurs))
    buffer_counter = 0 
    #Find stationary solution
    if buffr.size == 0:
        for i in range(int(np.floor(0.1*N))):
            yp1 = y + dt*ffun(y,**args)+ gfun(y,**args)*sqrtdt*\
            np.random.normal(loc=0.0, scale =1.0, size=np.size(y))
            
            mask_thres = np.sum(buffr,axis=0)>0.0
            yp1[mask_thres] = v_res
    
            mask_spk = yp1>v_thres        
            y = yp1
            
        if verbose:
            ('Pre-run completed')
            
        initial_cond = y
        #Save one voltage trace
        if example:    
            y_model = np.zeros(N)
            y_model[0] = y[0]
        else:
            y_model = 0
        buffer_counter = 0 
        spk_register = 0  #Check when new recording bin has been reached
        spk_timeidx = 0    
        
        
        for i in range(1,N):
            yp1 = y + dt*ffun(y,**args)+ sqrtdt*np.sqrt(2*D*(c))*stim[i-1] + \
                sqrtdt*np.sqrt(2*D*(1-c))*noise[i-1,:]
            
    
            mask_thres = np.sum(buffr,axis=0)>0.0
            yp1[mask_thres] = v_res
    
            mask_spk = yp1>v_thres        
    
            spks[spk_timeidx,mask_spk] = 1
            
            if example:
                y_model[i] = yp1[0]
    
            spk_register += 1
            if spk_register>=(dt_spk/dt):
                spk_register = 0
                spk_timeidx +=1
            y= yp1
            
        if verbose:
            print('Langevin integration completed')
            print('')
            
    else:
        for i in range(int(np.floor(0.1*N))):
            yp1 = y + dt*ffun(y,**args)+ gfun(y,**args)*sqrtdt*\
            np.random.normal(loc=0.0, scale =1.0, size=np.size(y))
            
            mask_thres = np.sum(buffr,axis=0)>0.0
            yp1[mask_thres] = v_res
    
            mask_spk = yp1>v_thres        
            buffr[buffer_counter,:] = mask_spk
            buffer_counter += 1
            if buffer_counter == ref_int:
                buffer_counter = 0
            y = yp1

        initial_cond = y
        #Save one voltage trace
        if example:    
            y_model = np.zeros(N)
            y_model[0] = y[0]
        else:
            y_model = 0
        buffer_counter = 0 
        spk_register = 0  #Check when new recording bin has been erreicht
        spk_timeidx = 0    
        for i in range(1,N):
            yp1 = y + dt*ffun(y,**args)+ dt*stim[i-1]*np.ones(np.shape(y)) + gfun(y,**args)*sqrtdt*\
            np.random.normal(loc=0.0, scale =1.0, size=np.size(y))
    
            mask_thres = np.sum(buffr,axis=0)>0.0
            yp1[mask_thres] = v_res
    
            mask_spk = yp1>v_thres        
            buffr[buffer_counter,:] = mask_spk
    
            spks[spk_timeidx,mask_spk] = 1
            
            buffer_counter += 1
            if buffer_counter == ref_int:
                buffer_counter = 0
                
            if example:
                y_model[i] = yp1[0]
    
            spk_register += 1
            if spk_register>=(dt_spk/dt):
                spk_register = 0
                spk_timeidx +=1
            y= yp1
        if verbose:
            print('Langevin integration completed')
            print('')
    return y_model, spks, initial_cond


def euler_maru_stim(ffun, gfun, stim, x0, dt, dt_spk, N, args, v_thres = 1.0, \
                        v_res = -0.1, t_ref = 2.0,example=True,verbose=True):
    '''
    Solves the Langevin equation of the form
    dX(t) = f(X(t))dt + g(X(t))dW(t)
    with threshold rule (one simulation)

    Runs a first prerun of 10% of the running time, in order to start at 
    equilibrium conditons.

    INPUT
    -----
    ffun: function f -non stochastical part. usually drift-
    gfun: function that modulates white noise.
    x0: initial conditions
    dt: discretization time of the integration method
    dt_spk: discretization time of the spike recording
    N: number of time steps in time array   
    args: dictionary including the constants of the functions
    v_thres: spiking threshold
    v_res: reset value of the potential
    t_ref: refractory period
    example: boolean. Say if you want to have the voltage trace of one neuron
    
    OUTPUT
    -----
    y: numpy array with the solution of just one neuron, as an example. 
        if example is False, returns 0   
    spks: np.array with the spike trains of all neurons. 1stdim: time. 2nd: neuron    
    initial_cond: initial conditions
    '''

    ref_int = np.floor(t_ref/dt)    
    sqrtdt = np.sqrt(dt)
    neurs = len(x0)
    D = args['D']
    y = x0
    buffr = np.zeros((ref_int,neurs))
    spks = np.zeros((int(np.floor((N*dt)/dt_spk))+1,neurs))
    buffer_counter = 0 
    #Find stationary solution
    if buffr.size == 0:
        
        initial_cond = y
        #Save one voltage trace
        if example:    
            y_model = np.zeros(N)
            y_model[0] = y[0]
        else:
            y_model = 0
        buffer_counter = 0 
        spk_register = 0  #Check when new recording bin has been reached
        spk_timeidx = 0    
        
        
        for i in range(1,N):
            yp1 = y + dt*ffun(y,**args)+ sqrtdt*stim[i-1] + gfun(y,**args)*sqrtdt*\
            np.random.normal(loc=0.0, scale =1.0, size=np.size(y))
    
            mask_thres = np.sum(buffr,axis=0)>0.0
            yp1[mask_thres] = v_res
    
            mask_spk = yp1>v_thres        
    
            spks[spk_timeidx,mask_spk] = 1
            
            if example:
                y_model[i] = yp1[0]
    
            spk_register += 1
            if spk_register>=(dt_spk/dt):
                spk_register = 0
                spk_timeidx +=1
            y= yp1
            
        if verbose:
            print('Langevin integration completed')
            print('')
            
    else:
        initial_cond = y
        #Save one voltage trace
        if example:    
            y_model = np.zeros(N)
            y_model[0] = y[0]
        else:
            y_model = 0
        buffer_counter = 0 
        spk_register = 0  #Check when new recording bin has been erreicht
        spk_timeidx = 0    
        for i in range(1,N):
            yp1 = y + dt*ffun(y,**args)+ dt*stim[i-1]*np.ones(np.shape(y)) + gfun(y,**args)*sqrtdt*\
            np.random.normal(loc=0.0, scale =1.0, size=np.size(y))
    
            mask_thres = np.sum(buffr,axis=0)>0.0
            yp1[mask_thres] = v_res
    
            mask_spk = yp1>v_thres        
            buffr[buffer_counter,:] = mask_spk
    
            spks[spk_timeidx,mask_spk] = 1
            
            buffer_counter += 1
            if buffer_counter == ref_int:
                buffer_counter = 0
                
            if example:
                y_model[i] = yp1[0]
    
            spk_register += 1
            if spk_register>=(dt_spk/dt):
                spk_register = 0
                spk_timeidx +=1
            y= yp1
        if verbose:
            print('Langevin integration completed')
            print('')
    return y_model, spks, initial_cond

def euler_maruyama_light(ffun, gfun, x0, dt, dt_spk, N, args, v_thres = 1.0, \
                        v_res = -0.1, t_ref = 2.0,example=True,verbose=True):
    '''
    Solves the Langevin equation of the form
    dX(t) = f(X(t))dt + g(X(t))dW(t)
    with threshold rule (one simulation)

    Runs a first prerun of 10% of the running time, in order to start at 
    equilibrium conditons.

    INPUT
    -----
    ffun: function f -non stochastical part. usually drift-
    gfun: function that modulates white noise.
    x0: initial conditions
    dt: discretization time of the integration method
    dt_spk: discretization time of the spike recording
    N: number of time steps in time array   
    args: dictionary including the constants of the functions
    v_thres: spiking threshold
    v_res: reset value of the potential
    t_ref: refractory period
    example: boolean. Say if you want to have the voltage trace of one neuron
    
    OUTPUT
    -----
    y: numpy array with the solution of just one neuron, as an example. 
        if example is False, returns 0   
    spks: np.array with the spike trains of all neurons. 1stdim: time. 2nd: neuron    
    initial_cond: initial conditions
    '''

    ref_int = np.floor(t_ref/dt)    
    sqrtdt = np.sqrt(dt)
    neurs = len(x0)
    
    y = x0
    buffr = np.zeros((ref_int,neurs))
    spks = np.zeros((int(np.floor((N*dt)/dt_spk))+1,neurs))
    buffer_counter = 0 
    #Find stationary solution
    if buffr.size == 0:
        for i in range(int(np.floor(0.1*N))):
            yp1 = y + dt*ffun(y,**args)+ gfun(y,**args)*sqrtdt*\
            np.random.normal(loc=0.0, scale =1.0, size=np.size(y))
            
            mask_thres = np.sum(buffr,axis=0)>0.0
            yp1[mask_thres] = v_res
    
            mask_spk = yp1>v_thres        
            y = yp1
            
        if verbose:
            ('Pre-run completed')

        initial_cond = y
        #Save one voltage trace
        if example:    
            y_model = np.zeros(N)
            y_model[0] = y[0]
        else:
            y_model = 0
        buffer_counter = 0 
        spk_register = 0  #Check when new recording bin has been reached
        spk_timeidx = 0    
        
        
        for i in range(1,N):
            yp1 = y + dt*ffun(y,**args)+ gfun(y,**args)*sqrtdt*\
            np.random.normal(loc=0.0, scale =1.0, size=np.size(y))
    
            mask_thres = np.sum(buffr,axis=0)>0.0
            yp1[mask_thres] = v_res
    
            mask_spk = yp1>v_thres        
    
            spks[spk_timeidx,mask_spk] = 1
            
            if example:
                y_model[i] = yp1[0]
    
            spk_register += 1
            if spk_register>=(dt_spk/dt):
                spk_register = 0
                spk_timeidx +=1
            y= yp1
            
        if verbose:
            print('Langevin integration completed')
            print('')
            
    else:
        for i in range(int(np.floor(0.1*N))):
            yp1 = y + dt*ffun(y,**args)+ gfun(y,**args)*sqrtdt*\
            np.random.normal(loc=0.0, scale =1.0, size=np.size(y))
            
            mask_thres = np.sum(buffr,axis=0)>0.0
            yp1[mask_thres] = v_res
    
            mask_spk = yp1>v_thres        
            buffr[buffer_counter,:] = mask_spk
            buffer_counter += 1
            if buffer_counter == ref_int:
                buffer_counter = 0
            y = yp1
        if verbose:
            print('Pre-run completed')
        initial_cond = y
        #Save one voltage trace
        if example:    
            y_model = np.zeros(N)
            y_model[0] = y[0]
        else:
            y_model = 0
        buffer_counter = 0 
        spk_register = 0  #Check when new recording bin has been erreicht
        spk_timeidx = 0    
        for i in range(1,N):
            yp1 = y + dt*ffun(y,**args)+ gfun(y,**args)*sqrtdt*\
            np.random.normal(loc=0.0, scale =1.0, size=np.size(y))
    
            mask_thres = np.sum(buffr,axis=0)>0.0
            yp1[mask_thres] = v_res
    
            mask_spk = yp1>v_thres        
            buffr[buffer_counter,:] = mask_spk
    
            spks[spk_timeidx,mask_spk] = 1
            
            buffer_counter += 1
            if buffer_counter == ref_int:
                buffer_counter = 0
                
            if example:
                y_model[i] = yp1[0]
    
            spk_register += 1
            if spk_register>=(dt_spk/dt):
                spk_register = 0
                spk_timeidx +=1
            y= yp1
        if verbose:
            print('Langevin integration completed')
            print('')
    return y_model, spks, initial_cond


def euler_maruyama(ffun, gfun, x0, t, args, thres = 1.0, res = -0.1, t_ref = 2.0):
    '''
    Solves the Langevin equation of the form
    dX(t) = f(X(t))dt + g(X(t))dW(t)
    with threshold rule (one simulation). Simplest version. No prerun

    INPUT
    -----
    ffun: function f -non stochastical part. usually drift-
    gfun: function that modulates white noise.
    x0: initial conditions
    t: numpy array with the discretization times
    args: dictionary including the contants of the functions
    thres: spiking threshold
    res: reset value of the potential
    t_ref: refractory period
    
    OUTPUT
    -----
    y: numpy array with the solution. 1stdim: time. 2nd: neuron    
    spks: binary numpy array with True values when spiking occurs. Same format 
          as y.
    '''

    dt = t[1]-t[0]
    ref_int = np.floor(t_ref/dt)

        
    sqrtdt = np.sqrt(dt)
    
    N = t.size
    y = np.zeros((N,x0.size))
    spks = np.zeros((N,x0.size))
    y[0,:] = x0

    for i in range(1,N):
        last_refidx = max(i-ref_int-1, 0)        
        mask_spk = y[last_refidx:i,:]>thres

        if np.shape(mask_spk)[0]>1:
            mask_spk_col = np.sum(mask_spk,axis=0)>0.1
        else:
            mask_spk_col = mask_spk[0,:]

        y[i,mask_spk_col] = res
        y[i,~mask_spk_col] = y[i-1,~mask_spk_col] + dt*ffun(y[i-1,~mask_spk_col],t,**args)+ gfun(y[i-1,~mask_spk_col],t,**args)*sqrtdt*np.random.normal(loc=0.0, scale =1.0, size=np.size(y[i-1,~mask_spk_col]))
        spks[i-1,mask_spk[-1,:]] = 1
    return y, spks
        
        
def gfun(y,**args):
    D = args['D']
    return(np.sqrt(2*D))
    
def ffun2(y,mus):
    return(-y+mus)
    
def ffun(y,**args):
    #tau = args['tau']
    mu = args['mu']
    return(-y+mu)
    
def ffun_PIF(y,**args):
    #tau = args['tau']
    mu = args['mu']
    return(mu)

def get_densISI(D, mu_a, thres = 1.0, res = 0.0, tref = 0.0, T = 16, dt = 0.001, tol = 1e-15, verbose = False):
    '''
    Calculate the real ISI density of an LIF neuron given D and mu_a
    
    INPUT
    -----
    D: Dispersion coefficient in "a"
    mu_a: Value of mu_a
    T: double the length of the support of the ISI array
    
    OUTPUT
    -----
    pISI: pISI of population a
    timeISI:  time of the ISI in order to plot(timeISI, pISI)
    '''
        
    # Calculate P_ISI of population A

    time = np.arange(0,T, dt)

    f = np.fft.fftfreq(len(time))
    f[0] = 0.0001
    
    args = { 'D':D, 'mu': mu_a, 'verbose':verbose}
    dens_re, dens_im = isidens_w( f/dt, thres, res, tref, times = time, tol= tol, **args)            

    densA = dens_re+1j*dens_im
    a = np.fft.fftshift(np.fft.ifft((densA)))
    idx = int(np.round(len(a)/2))
    b = np.real(a[idx:])
    
    timeISI = time[0:len(b)]
    b = 2.0*(b - np.min(b))
    b[timeISI<=tref]=0.0
    b = np.clip(b - np.max(b[-10:]), 0.0, 1.0/dt)
    pISI = b/np.sum(b*dt)
    
    return(pISI, timeISI, densA) 

def trick_cdf(old_cdf, old_mus, new_mus):
    new_cdf = np.zeros(len(new_mus))
    for i in range(len(new_mus)):
        arg = np.argmin(np.abs(new_mus[i]-old_mus))
        new_cdf[i] = old_cdf[arg]
    return(new_cdf)
        
def get_mus_givenTs(Ts, tref):
    '''
    To use only when Ds = 0
    '''
    return(1.0/(1-np.exp(-Ts+tref)))
    

def get_theorPmu_dnot0(pISI, Ts, tref, D):
    res = 0.0
    thres = 1.0
    #Now let's take a look at the function that relates mu to the ISI
    mu_support = np.linspace(0.0, 20.0, 8000)
    ISI_support = np.zeros(len(mu_support))
    P_mu = np.zeros(len(mu_support))
    dmu = mu_support[1]-mu_support[0]    
    count = 0    
    args = {'D':D, 'mu':mu_support[0]}
    for m in mu_support:
        args['mu']= m
        ISI_support[count]= (1.0/LIF_fn( tref, res, thres, **args))
        count += 1
    ISI_support = np.nan_to_num(ISI_support)
    derISI = np.diff(ISI_support)
    derISI = np.append(derISI, derISI[-1])/dmu

    #Now find the right mus for the sampled ISIs
    count = 0
    for ISI in Ts:
        idx = np.abs(ISI_support -ISI).argmin()
        P_mu[count] = pISI[idx]*derISI[idx] 
        count +=1
        
    rho_hom_mu = P_mu/np.sum(P_mu*dmu)
    return(rho_hom_mu, mu_support)
    
def calculate_rhoB(mus, pISI_A, Ds, tref, T, dt, verbose=True, tol=1e-2):
    rho_B = np.zeros((len(pISI_A),len(mus)))
    problems = False
    for i in range(len(mus)):
        if verbose:
            if i/15.0 == i/15:           
                print(i)
        if mus[i]<1.5:
            toL=1e-3
        elif mus[i]<2.0:
            toL=5e-3
        elif mus[i]>3.2:
            toL=1e-1
        else:
            toL=tol
        pISI, timeISI, dens = get_densISI(Ds, mus[i],  tref = tref, T=T, dt = dt, tol=toL)
        #For very high values of mu, there are problems. Then we freeze 
        # the solution to the last mu that worked
        if np.isnan(np.sum(np.real(dens))):
            if problems == False:
                I = i-2
            problems = True
            pISI = rho_B[:,I]

        if mus[i]<1.0 and np.isnan(np.sum(pISI)):
            pISI = np.zeros(len(pISI))
            
        rho_B[:,i] = pISI
        mu_max = 10.0
        if mus[i]>mu_max:
            tail = np.mean(rho_B[-10:,i])
            mask = rho_B[:,i]<2.0*tail
            rho_B[mask,i] = 0.0
        if np.sum(rho_B[:,i]>0):
            rho_B[:,i] = rho_B[:,i]/np.sum(rho_B[:,i]*dt)         
    return(rho_B, timeISI)       
    
def calculate_rhoB_noise(Dds, tref, T, dt, verbose=True, tol=1e-2):
    timeISI = np.arange(0,T,dt)
    rho_B = np.zeros((len(timeISI),len(Dds)))
    for i in range(len(Dds)):
        if verbose:
            if i/15.0 == i/15:           
                print(i)
        rho_B[:,i] = LIF_ISI_mu1(Dds[i], tref, T, dt)
    
    return(rho_B, timeISI)   
    
def get_heteromus(Nb, timeISI, pISI, D,  thres = 1.0, res = 0.0, tref = 0.0, T = 16, bias = False):
    '''
    Sets the mu values of population b so that the ISI density of "b"
    is equal to the ISI density of "a" when c = 0
    
    INPUT
    -----
    D: Dispersion coefficient in "a"
    mu_a: Value of mu_a
    Nb: number
    
    OUTPUT
    -----
    mus_b: mu values of each neuron in population b
    pISI: pISI of population a
    timeISI:  time of the ISI in order to plot(timeISI, pISI)
    '''
    #Sample n times from pISI
    ISIsamples = np.random.choice(timeISI, Nb, p = pISI/np.sum(pISI))
    
    #Now let's take a look at the function that relates mu to the ISI
    mu_support = np.linspace(0.0, 20.0, 8000)
    ISI_support = np.zeros(len(mu_support))
    count = 0    
    args = {'D':0.000000001, 'mu':mu_support[0]}
    for m in mu_support:
        args['mu']= m
        ISI_support[count]= (1.0/LIF_fn( tref, res, thres, **args))
        count += 1
    ISI_support = np.nan_to_num(ISI_support)
    mus_b = []
    #Now find the right mus for the sampled ISIs
    
    high_number = 2000
    for ISI in ISIsamples:
        idx = np.abs(ISI_support -ISI ).argmin()
        if bias:        
            mus_b.extend(mu_support[idx])
        else:
            unbias = int(np.round(high_number/ISI))
            mus_b.extend([mu_support[idx]]*unbias)
        count +=1
      
    
    return(mus_b, mu_support, ISI_support, ISIsamples)

    
def get_heteromus_nonoise(Nb, timeISI, pISI, thres = 1.0, res = 0.0, tref = 0.0, T = 16, bias = False):
    '''
    Sets the mu values of population b so that the ISI density of "b"
    is equal to the ISI density of "a" when c = 0
    
    INPUT
    -----
    D: Dispersion coefficient in "a"
    mu_a: Value of mu_a
    Nb: number
    
    OUTPUT
    -----
    mus_b: mu values of each neuron in population b
    pISI: pISI of population a
    timeISI:  time of the ISI in order to plot(timeISI, pISI)
    '''
    #Sample n times from pISI
    ISIsamples = np.random.choice(timeISI, Nb, p = pISI/np.sum(pISI))
    
    #Now let's take a look at the function that relates mu to the ISI
    mu_support = np.linspace(0.0, 20.0, 8000)
    
    ISI_support = tref + np.log(mu_support/(mu_support-1))
    ISI_support = np.nan_to_num(ISI_support)
    mus_b = []
    #Now find the right mus for the sampled ISIs
    
    high_number = 2000
    for ISI in ISIsamples:
        idx = np.abs(ISI_support -ISI ).argmin()
        if bias:        
            mus_b.extend(mu_support[idx])
        else:
            unbias = int(np.round(high_number/ISI))
            mus_b.extend([mu_support[idx]]*unbias)
       
      
    
    return(mus_b, mu_support, ISI_support, ISIsamples)

def get_hetmus_nonoise2(pISI, tme, tref):
    mus = get_mus_givenTs(tme[tme>tref], tref)
    mask = np.argsort(mus)  
    rho_hom_mu = pISI[tme>tref][mask]  
    mean = np.sum(tme*pISI)/np.sum(pISI)        
    P_mu = (tref+np.log(mus/(mus-1)))*rho_hom_mu/(mus*mean*(mus-1))    
    #P_mu2 = np.interp(mu_support)
    
    return(P_mu, mus)


def get_hetmus_nonoise_gaussapprox(mu_support, muA, Da, tref, T=30, dt = 0.01, tol=1e-8):
    '''
    using dirty tricks and reusing functions. no hay ganas para otra cosa
    '''
    A = (muA-0.)#/np.sqrt(2*D)
    B = (muA-1.0)#/np.sqrt(2*D)

    mean = tref + np.log(muA/(muA-1)) - 0.5*Da*((1.0/B**2)-(1.0/A**2)) 
    Var = Da*((1.0/B**2)-(1.0/A**2))
    St = np.sqrt(Var)
    
    pISI, tme, dens = get_densISI(0.1, muA, tref=tref, tol = 1e-8, T=T, dt = dt)        
    pISI = np.exp(-(tme-mean)**2/(2*St**2))/np.sqrt(2*np.pi*St**2)

    P1_mu, mus, TTs, PItheor = get_theorPmu_d0(pISI, tme, tref)
    interPDF = np.interp(mu_support, mus, P1_mu)
    P_mu = interPDF/(np.sum(interPDF*(mu_support[1]-mu_support[0])))
    
    P_mu = P_mu/(mu_support*(mu_support-1))
    P_mu = P_mu/np.sum(P_mu[1:]*(mu_support[1]-mu_support[0]))
    P_mu[0]=0.0
    return(P_mu, tme, pISI,mus, P1_mu, TTs, PItheor)
    

def get_hetmus_nonoise(mu_support, muA, Da, tref, T=30, dt = 0.01, tol=1e-8):
    pISI, tme, dens = get_densISI(Da, muA, tref=tref, tol = 1e-8, T=T, dt = dt)        
    qISI = 1.0*pISI    
    P1_mu, mus, TTs, PItheor = get_theorPmu_d0(pISI, tme, tref)
    interPDF = np.interp(mu_support, mus, P1_mu)
    P_mu = interPDF/(np.sum(interPDF*(mu_support[1]-mu_support[0])))
    
    P_mu = P_mu/(mu_support*(mu_support-1))
    P_mu = P_mu/np.sum(P_mu[1:]*(mu_support[1]-mu_support[0]))
    P_mu[0]=0.0
    return(P_mu, tme, qISI,mus, P1_mu, TTs, PItheor)

    
def get_theorPmu_d0(pISI, Ts, tref):
    #First move to Phat    
    pI = Ts*pISI 
    dt = Ts[1]-Ts[0]
    pI = pI/np.sum(pI*dt)
    mus = get_mus_givenTs(Ts[Ts>tref], tref)
    mask = np.argsort(mus)
    TTs = Ts[Ts>tref][mask]
    rho_hom_mu = pI[Ts>tref][mask]    
    mus = np.sort(mus)
    
    return(rho_hom_mu, mus, TTs, pI)
    
def rasterplot(spks,t, ax, S=3.0, coL = 'k', alpha=1.0, rr = False):
    '''
    Plots rastergram of a simulation
    INPUT
    -----
    spks: boolean array. 1st dim: time discretization. 2nd dim: number of neurons
    t: Time array
    
    OUTPUT
    -----
    [just the plot]
    
    '''
    N_neur = spks.shape[1]
    T = t[-1]
    
    for col in range(N_neur):
        ax.scatter(t[spks[:,col]==1],(N_neur-col)*np.ones(np.size(t[spks[:,col]==1])),s=S, c=coL, edgecolor=coL, alpha = alpha, rasterized=rr)
        
    # Hide the right and top spines
#    ax.spines['right'].set_visible(False)
#    ax.spines['left'].set_visible(False)
#    ax.spines['bottom'].set_visible(False)
    
    # Only show ticks on the left and bottom spines
#    ax.yaxis.set_ticks_position('left')
#    ax.xaxis.set_ticks_position('bottom')
#    
    ax.set_xlim([0,T])
    ax.set_ylim([0,N_neur])
#    plt.tick_params(axis='y', which='both',bottom='off', top='off', labelbottom='off')
#    plt.title('Rasterplot')
#    plt.xlabel('Time [ms]')
    ax.set_ylabel('Neuron', fontsize=40)
    return()    
        
def firingrate(spks,t,window):
    '''
    Calculates firing rate [Hz] of spike trains. 
    Using a rectagular window centered at time t
    
    INPUT
    -----
    spks: boolean array. 1st dim: time discretization. 2nd dim: number of neurons
    t: Time array
    window: Size of the temporal window to calculate FR in ms

    OUTPUT
    -----
    frate: firing rate in Hz. Same format as spks. Border effects at the 
           initial time points
    
    '''
    frate = np.zeros(np.shape(spks))
    t_steps = np.size(t)
    dt = t[1]-t[0]
    window_steps = np.floor(window/dt)
    eff_window = dt*window_steps
    
    for i in range(t_steps):
        last_idx = max(0,i-window_steps//2)
        first_idx = i+np.ceil(window_steps/2.0)+1
        frate[i,:] = np.sum(spks[last_idx:first_idx,:],axis=0)/eff_window
    
    return(frate)

def LIF_fn( tref, vr, vthres, **args):
    #tref has to be in units of tau!!
    mu = args['mu']
    #tau = args['tau']
    D = args['D']
    
    #if D<0.1 and mu>1.001:
    #    return(1.0/np.log(mu/(mu-1)))
    #else:  
    a = (mu-vr)/np.sqrt(2*D)
    b = (mu-vthres)/np.sqrt(2*D)
    func = lambda x: np.exp(x*x)*erfc(x)    
    partB, err = quad(func, b, a)
    result = 1.0/((np.sqrt(np.pi)*partB+tref))
    if np.isnan(result) or np.isinf(result):
        result = 1.0/(tref+(mp.ln(mu)-mp.ln(mu-1))-D**2*(-mp.power(mu,-2)+mp.power(mu-1,-2)))
    if mu>0.0 and result==0.0:
        result = 1.0/(tref+(mp.ln(mu)-mp.ln(mu-1))-D**2*(-mp.power(mu,-2)+mp.power(mu-1,-2)))    
    return(result)

def LIF_varT(vr, vthres,  timesa = 10, **args):
    #tref has to be in units of tau!!
    mu = args['mu']
    #tau = args['tau']
    D = args['D']
    
    a = (mu-vr)/np.sqrt(2*D)
    b = (mu-vthres)/np.sqrt(2*D)
    
    ysupp = np.linspace(b, timesa*a, 10000)
    dy = ysupp[1]-ysupp[0]
    fac2 = np.zeros(len(ysupp))
    fac1 = np.zeros(len(ysupp))
    fac1[0] = 2*np.pi*dy*np.exp(ysupp[0]**2)*(erfc(ysupp[0]))**2
    funcz = lambda x: np.exp(x**2)    
    fcin = np.zeros(len(ysupp))
    fcin[0] = erfc(0)
    for i in range(len(ysupp)-1):
        fcin[i+1] = erfc(ysupp[i+1])
        fac1[i+1] = 2*np.pi*dy*np.exp(ysupp[i+1]**2)*(fcin[i+1])**2
        if ysupp[i+1]<a:
            fac2[i+1]=0
        else:
            p, err = quad(funcz, max(a,b),ysupp[i+1])
            fac2[i+1] = p
        
    return(np.sum(fac2*fac1))

def LIF_varT2(vr = 0.0, vthres=1.0, **args):
    mu = args['mu']
    D = args['D']
    
    a = (mu-vr)/np.sqrt(2*D)
    b = (mu-vthres)/np.sqrt(2*D)
    limxmin = b
    func = lambda x,y: np.exp(x**2)*(erfc(x))**2*np.exp(y**2)*np.piecewise(y, y<b, [1.0, 0.0])
    hfunc = lambda x: x    
    gfunc = lambda x: b
    args = {'a':a, 'b':b}
    
    cond = 0
    init_times = 28
    while cond == 0:
        init_times = 0.9*init_times
        limxtop = init_times*a
        p, err = dblquad(func, limxmin, limxtop, gfunc, hfunc)
        if np.isnan(p)==False:
            cond = 1
    
    return(p, err, init_times)        
def PSD(signal, t):
    dt = (t[1]-t[0])
    T = t[-1]
    FTsignal = np.fft.fft(signal,axis = 0)
    FTfreq = np.fft.fftfreq(len(signal),d=dt)
    PSD = np.real((FTsignal*np.conj(FTsignal)))*(dt*dt/T)
    mask = FTfreq>0
    return FTfreq[mask], PSD[mask]

    
def crossPSD(signalA, signalB, t):
    dt = (t[1]-t[0])
    T = t[-1]
    FTsignalA = np.fft.fft(signalA,axis = 0)
    FTsignalB = np.fft.fft(signalB,axis=0)
    FTfreq = np.fft.fftfreq(len(signalA),d=dt)
    PSD = np.real((FTsignalA*np.conj(FTsignalB)))*(dt*dt/T)
    mask = FTfreq>0
    return FTfreq[mask], PSD[mask]
def im_crossPSD(signalA, signalB, t):
    dt = (t[1]-t[0])
    T = t[-1]
    FTsignalA = np.fft.fft(signalA,axis = 0)
    FTsignalB = np.fft.fft(signalB,axis=0)
    FTfreq = np.fft.fftfreq(len(signalA),d=dt)
    PSD = (FTsignalA*np.conj(FTsignalB))*(dt*dt/T)
    mask = FTfreq>0
    return FTfreq[mask], PSD[mask]
    
def chi(noisec, spikes, t):
    T = max(t)
    dt = t[1]-t[0]
    noisec = noisec[0:spikes.shape[0]]
    FTspk = np.fft.fft(spikes, axis=0)
    FTnoise = np.tile(np.fft.fft(noisec),(FTspk.shape[1],1))
    FTf = np.fft.fft(box(t-T/2, dt))
    numerator = np.mean(FTspk*np.conj(FTf*FTnoise).T,axis=1)
    denominator = FTnoise*T
    return(numerator/denominator)
    
def PSD_full(signal, t):
    dt = (t[1]-t[0])
    FTsignal = np.fft.fft(signal,axis = 0)
    FTfreq = np.fft.fftfreq(len(signal),d=dt)
    PSD = ((dt)/(len(signal)))*(np.abs(FTsignal))**2
    return FTfreq, PSD
def cross_spectra(frates, t):
    dt = (t[1]-t[0])
    neurons = frates.shape[1]
    lsignal = frates.shape[0]
    cross_spectra = np.zeros((lsignal,neurons//2))*(0.0+1j*0.0)

    for i in range(neurons//2):
        FT_i = np.fft.fft(frates[:,2*i],axis = 0)
        FT_j = np.fft.fft(frates[:,2*i+1],axis = 0)
        cross_spectra[:,i] += (FT_i*np.conj(FT_j)*((dt)/(len(frates[:,i]))))

    FTfreq = np.fft.fftfreq(len(frates[:,i]),d=dt)
    
    return cross_spectra, FTfreq

def FTwindow(freq, window):
    return((np.sin(np.pi*window*freq))/(np.pi*window*freq))
   
def FTgaussian(freq, window):
    return(np.exp(-np.pi*np.pi*freq*freq*2*window*window))

def FTgaussian_red(freq, window):
    return(np.exp(-np.pi*np.pi*freq*freq*window*window*0.5))
def FTsigbox(t, window, k=3.0/10.0):
    
    center = t[-1]/2.0
    filt = sigbox(t-center, window, k)
    filt = np.reshape(filt,(1,len(filt)))
    FTfilt = np.fft.fft(filt)*(t[1]-t[0])
    return(FTfilt)

    
def spectrum_LIF( freq, v_t, v_r, t_ref, **args):
    
    mu = args['mu']
    D = args['D']
    delta = (v_r**2-v_t**2+2*mu*(v_t-v_r))/(4*D)
    fracvT = (mu-v_t)/np.sqrt(D)
    fracvR = (mu-v_r)/np.sqrt(D)
    omega = 2*np.pi*freq
    
    r0 = LIF_fn(t_ref, v_r, v_t, **args)
    
    numerator = (np.abs(mp.pcfd(1j*omega, fracvT)))**2- \
                mp.exp(2*delta)*(np.abs(mp.pcfd(1j*omega,fracvR)))**2
    
    denominator = (np.abs(mp.pcfd(1j*omega,fracvT)-mp.exp(delta)*\
                  mp.exp(1j*omega*t_ref)*mp.pcfd(1j*omega,fracvR)))**2

    return(r0*(numerator/denominator))
    
def isidens_w(freq, v_t, v_r, t_ref, tol=1e-15, times =0, **args):
    
    mu = args['mu']
    D = args['D']
    delta = (v_r**2-v_t**2 + 2.0*mu*(v_t-v_r))/(4.0*D)
    fracvT = (mu-v_t)/np.sqrt(D)
    fracvR = (mu-v_r)/np.sqrt(D)
    omega = 2.0*np.pi*freq

    dens_re = np.zeros(len(freq))
    dens_im = np.zeros(len(freq))
    
    try:
        verbose = args['verbose']
    except KeyError:
        verbose = False
        
    for i in range(int(len(freq)/2)):
        parsed = False
        if verbose:
            print(i)
        while not parsed:
            try:
                if t_ref == 0.0:
                    dens = mp.exp(delta)*mp.pcfd(1j*omega[i], fracvR)/mp.pcfd(1j*omega[i], fracvT)    
                else:
                    dens = mp.exp(delta)*np.exp(1j*omega[i]*t_ref)*mp.pcfd(1j*omega[i], fracvR)/mp.pcfd(1j*omega[i], fracvT)                    

                parsed=True
            except ValueError:
                dens= 0.0*1j
                print('Breaking at i = ', i,' and omega[i]= ', omega[i])  
                break
            
        dens_re[i] = dens.real
        dens_re[-i]= dens.real
        dens_im[i] = dens.imag
        dens_im[-i] = dens.imag
        #print(dens_re[i], dens_im[i], np.abs(dens), tol, np.abs(dens)<tol)
        if i>1 and np.abs(dens)<tol:# and D>0.1:
            dens_re[i:-i] = 0.0
            dens_im[i:-i] = 0.0
            #print(i,'inserting zeros')
            break
        if D<0.1 and np.isnan(np.sum(dens_re)):
            dt = times[1]-times[0]
            meanIS = 1.0/LIF_fn( t_ref, v_r, v_t, **args)
            idx = np.argmin(np.abs(meanIS-times))
            timeISI = np.zeros(len(freq))
            timeISI[idx] = 1.0/dt 
            dens = np.fft.fft(timeISI)
            dens_re = np.real(dens)
            dens_im = np.imag(dens)
            
    return(dens_re, dens_im)
    
def isidens_exact( freq, v_t, v_r, t_ref, tol=1e-15, times =0, **args):
    
    mu = args['mu']
    D = args['D']
    delta = (v_r**2-v_t**2 + 2.0*mu*(v_t-v_r))/(4.0*D)
    fracvT = (mu-v_t)/np.sqrt(D)
    fracvR = (mu-v_r)/np.sqrt(D)
    omega = 2.0*np.pi*freq

    dens_re = np.zeros(len(freq))
    dens_im = np.zeros(len(freq))
    
    try:
        verbose = args['verbose']
    except KeyError:
        verbose = False
        
    for i in range(len(freq)/2):
        parsed = False
        if verbose:
            print(i)
        while not parsed:
            try:
                if t_ref == 0.0:
                    dens = mp.exp(delta)*mp.pcfd(1j*omega[i], fracvR)/mp.pcfd(1j*omega[i], fracvT)    
                else:
                    dens = mp.exp(delta)*np.exp(1j*omega[i]*t_ref)*mp.pcfd(1j*omega[i], fracvR)/mp.pcfd(1j*omega[i], fracvT)                    

                parsed=True
            except ValueError:
                dens= 0.0*1j
                print('Breaking at i = ', i,' and omega[i]= ', omega[i])  
                break
            
        dens_re[i] = dens.real
        dens_re[-i]= dens.real
        dens_im[i] = dens.imag
        dens_im[-i] = dens.imag
        #print(dens_re[i], dens_im[i], np.abs(dens), tol, np.abs(dens)<tol)
        #if i>1 and np.abs(dens)<tol:# and D>0.1:
        #    dens_re[i:-i] = 0.0
        #    dens_im[i:-i] = 0.0
        #    print(i,'inserting zeros')
        #    break
        if D<0.1 and np.isnan(np.sum(dens_re)):
            dt = times[1]-times[0]
            meanIS = 1.0/LIF_fn( t_ref, v_r, v_t, **args)
            idx = np.argmin(np.abs(meanIS-times))
            timeISI = np.zeros(len(freq))
            timeISI[idx] = 1.0/dt 
            dens = np.fft.fft(timeISI)
            dens_re = np.real(dens)
            dens_im = np.imag(dens)
            
    return(dens_re, dens_im)
    
def chi_LIF(freq, v_t, v_r, t_ref, **args):
    
    mu = args['mu']
    D = args['D']
    delta = (v_r**2-v_t**2+2*mu*(v_t-v_r))/(4*D)
    fracvT = (mu-v_t)/np.sqrt(D)
    fracvR = (mu-v_r)/np.sqrt(D)
    omega = 2*np.pi*freq
    r0 = LIF_fn(t_ref, v_r, v_t, **args)
    chi = np.zeros(len(omega),np.complex)
    for i, o in enumerate(omega):
    
        prefact = (r0*o*1j/mp.sqrt(D))/(o*1j-1.0)
        numerator = (mp.pcfd(1j*o-1.0, fracvT))- \
                mp.exp(delta)*(mp.pcfd(1j*o-1.0,fracvR))
        denominator = (mp.pcfd(1j*o, fracvT))- \
                mp.exp(delta)*mp.exp(1j*o*t_ref)*(mp.pcfd(1j*o,fracvR))
        if i>0:
            chi[i]=prefact*(numerator/denominator) 
    return(chi)

def coherence_thesis(freq, v_t, v_r, t_ref, N, sigma, fmax, verbose = False, **args):
    chi = chi_LIF(freq, v_t, v_r, t_ref, **args)
    PSD = np.zeros(len(freq))   
    Sss = (sigma**2/(2*fmax))
    for i, f in enumerate(freq):
        if verbose == True and i/10 == i/10.0:
            print(i)
        PSD[i] = spectrum_LIF(f, v_t, v_r, t_ref, **args)
    coh = N*Sss*np.abs(chi)**2/((PSD)+(N-1)*Sss*np.abs(chi)**2)#N*np.abs(chi)**2*args['Ds']*2/(PSD+(N-1)*2*args['Ds']*np.abs(chi)**2)
    return(coh, chi, PSD)
    
def gaussian(x,sigma):
    return((1.0/(np.sqrt(2*np.pi)*sigma))*np.exp(-x*x/(2.0*sigma*sigma)))
    
def conduct_filt(x, tauC):
    out = np.zeros(np.shape(x))
    mask = (x>0.0)
    out[mask] = (1.0/tauC)*np.exp(-x[mask]/tauC)
    return(out)
    
def sigbox(x, window, k):
    prefac = (1.0/window)
    step_first = 1.0/(1+np.exp(-(4.0/(k*window))*(x+window/2.0)))
    step_second = 1.0/(1+np.exp(-(4.0/(k*window))*(x-window/2.0)))
        
    return(prefac*(step_first-step_second))

def box(x, window):
    prefac = (1.0/window)
    mask = np.abs(x)<window/2.0
    filt = np.zeros(x.shape)
    filt[mask]= prefac
        
    return(filt)

def fastfiringrate_gauss(spks,t,window):
    '''
    Calculates firing rate [Hz] of spike trains. 
    Using a gauss window centered at time t
    
    INPUT
    -----
    spks: boolean array. 1st dim: time discretization. 2nd dim: number of neurons
    t: Time array
    window: Size of the temporal window to calculate FR in ms

    OUTPUT
    -----
    frate: firing rate in Hz. Same format as spks. Border effects at the 
           initial time points
    
    '''
    center = t[-1]/2.0
    filt = gaussian(t-center, window)
    frate = np.convolve(spks.T,filt,mode='same').T
    return(frate)

def firingrate_cond(spks,t,window):
    '''
    Calculates firing rate [Hz] of spike trains. 
    Using a gauss window centered at time t
    
    INPUT
    -----
    spks: boolean array. 1st dim: time discretization. 2nd dim: number of neurons
    t: Time array
    window: Size of the temporal window to calculate FR in ms

    OUTPUT
    -----
    frate: firing rate in Hz. Same format as spks. Border effects at the 
           initial time points
    
    '''
    center = t[-1]/2.0
    filt = conduct_filt(t-center, window)
    filt = np.reshape(filt,(1,len(filt)))
    frate = convolve2d(spks.T,filt,mode='same').T
    return(frate)

def firingrate_cond1(spks,t,window):
    '''
    Calculates firing rate [Hz] of spike trains. 
    Using a gauss window centered at time t
    
    INPUT
    -----
    spks: boolean array. 1st dim: time discretization. 2nd dim: number of neurons
    t: Time array
    window: Size of the temporal window to calculate FR in ms

    OUTPUT
    -----
    frate: firing rate in Hz. Same format as spks. Border effects at the 
           initial time points
    
    '''
    center = t[-1]/2.0
    filt = conduct_filt(t-center, window)
    frate = np.convolve(spks,filt,mode='same')
    return(frate)
    
def firingrate_gauss(spks,t,window):
    '''
    Calculates firing rate [Hz] of spike trains. 
    Using a gauss window centered at time t
    
    INPUT
    -----
    spks: boolean array. 1st dim: time discretization. 2nd dim: number of neurons
    t: Time array
    window: Size of the temporal window to calculate FR in ms

    OUTPUT
    -----
    frate: firing rate in Hz. Same format as spks. Border effects at the 
           initial time points
    
    '''
    center = t[-1]/2.0
    filt = gaussian(t-center, window)
    filt = np.reshape(filt,(1,len(filt)))
    frate = convolve2d(spks.T,filt,mode='same').T
    return(frate)

def firingrate_sigbox(spks,t,window, k = 3.0/10.0):
    '''
    Calculates firing rate [Hz] of spike trains. 
    Using a gauss window centered at time t
    
    INPUT
    -----
    spks: boolean array. 1st dim: time discretization. 2nd dim: number of neurons
    t: Time array
    window: Size of the temporal window to calculate FR in ms
    k: in units of 1/window. Inverse of width of transition
    OUTPUT
    -----
    frate: firing rate in Hz. Same format as spks. Border effects at the 
           initial time points
    
    '''
    center = t[-1]/2.0
    filt = sigbox(t-center, window, k)
    filt = np.reshape(filt,(1,len(filt)))
    frate = convolve2d(spks.T,filt,mode='same').T
    return(frate)


def firingrate_box(spks,t,window):
    '''
    Calculates firing rate [Hz] of spike trains. 
    Using a gauss window centered at time t
    
    INPUT
    -----
    spks: boolean array. 1st dim: time discretization. 2nd dim: number of neurons
    t: Time array
    window: Size of the temporal window to calculate FR in ms
    k: in units of 1/window. Inverse of width of transition
    OUTPUT
    -----
    frate: firing rate in Hz. Same format as spks. Border effects at the 
           initial time points
    
    '''
    center = t[-1]/2.0
    filt = box(t-center, window)
    filt = np.reshape(filt,(1,len(filt)))
    frate = convolve2d(spks.T,filt,mode='same').T
    return(frate)
def get_lognorm(xs, mu, sigma):
    if sigma ==0.0:
        sigma = 1e-1
    P = np.zeros(len(xs))
    P[xs>0.0] = (1.0/(xs[xs>0.0]*sigma*np.sqrt(2*np.pi)))*np.exp(-(np.log(xs[xs>0.0])-mu)**2/(2*sigma**2))
    return(P)
    
def probab_dist_maxim(x,eps):
    Pi = np.pi
    
    integ = np.sqrt(Pi/2.0)*(1.0+erf((x/eps)*np.sqrt((1-eps**2)/2.0)))
    factor1 = 1.0/ np.sqrt(2.0*Pi)
    factor21 = eps*np.exp(-0.5*x**2/eps**2)
    result = factor1*(factor21 + np.sqrt(1.0-eps**2)*x*np.exp(-0.5*x**2)*integ)

    return(result)    

def eq_6_11(N, eps):
    theta = np.log(np.sqrt(1.0-eps**2)*N)
    func = lambda x: (1.0-(1.0-np.exp(-x)/N)**N)*((theta+x)**(-0.5))
    a = -theta
    b = +np.inf
    integral, err = quad(func, a, b)
    
    return((1.0/np.sqrt(2.0))*integral)    

    
def get_nice_colors():
    # These are the "Tableau 20" colors as RGB.    
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
    tableau20blind = [(0, 107, 164), (255, 128, 14), (171, 171, 171), (89, 89, 89),
             (95, 158, 209), (200, 82, 0), (137, 137, 137), (163, 200, 236),
             (255, 188, 121), (207, 207, 207), (255, 187, 120)]
             
    for i in range(len(tableau20blind)):  
        r, g, b = tableau20blind[i]  
        tableau20blind[i] = (r / 255., g / 255., b / 255.)
  # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
    for i in range(len(tableau20)):    
        r, g, b = tableau20[i]    
        tableau20[i] = (r / 255., g / 255., b / 255.)
        
    return(tableau20, tableau20blind)

def get_rightframe(ax, majory=1.0, minory=0.5, majorx=1.0, minorx=0.5, fontsize = 40, \
xlabel='', ylabel='', labelsize = 35):
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)

    minorLocator = MultipleLocator(minory)
    majorLocator = MultipleLocator(majory)
    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    majorLocator = MultipleLocator(majorx)
    minorLocator = MultipleLocator(minorx)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_major_locator(majorLocator)
    
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False) 
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.tick_params(axis='both', which='major', labelsize=labelsize,length=20, width=1, pad=10)
    ax.tick_params(axis='both', which='minor', labelsize=labelsize,length=10, width=1.0, pad=10)    
    
    #ticks_font = font.FontProperties(family='Helvetica', style='normal',
    #size=18, stretch='normal')
    
    #for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #    label.set_fontproperties(ticks_font)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)    
    
    
    return()
    
def find_hetmu(Dhom, c, muA, tref, string):
    '''
    Find the right mu of the heterogeneous population
    that leads to the same mean ISI as in the homogeneous 
    population
    '''
    args = {'D':Dhom, 'mu':muA}
    v_r = 0.0
    v_t = 1.0
    if string=='mean': 
        targetT = 1.0/LIF_fn(tref, v_r, v_t, **args)
    elif string=='mode':
        pISI, timeISI, dens = get_densISI(Dhom, muA, tref = tref, tol = 1e-3)
        targetT = timeISI[np.argmax(pISI)]
        
   # if verbos:
    #print('Target: '+str(targetT))
    mu_left = muA
    mu_right = 2.0*muA
    
    #Check if mu_left gives a longer T and mu_right a shorter
    args_left =  {'D':c*Dhom, 'mu':mu_left}
    args_right =  {'D':c*Dhom, 'mu':mu_right}

    if string=='mean':    
        T_left = 1.0/LIF_fn(tref, v_r, v_t, **args_left)
        T_right = 1.0/LIF_fn(tref, v_r, v_t, **args_right)
    elif string=='mode':
        pISI, timeISI, dens = get_densISI(c*Dhom, mu_left, tref = tref, tol = 1e-3)
        T_left = timeISI[np.argmax(pISI)]
        pISI, timeISI, dens = get_densISI(c*Dhom, mu_right, tref = tref, tol = 1e-3)
        T_right = timeISI[np.argmax(pISI)]
       
    #Start loop
    tol = 1e-4
    tol_mu = 1e-4
    niter = 0
    err = 1.0
    err_mu = 10.0
    old = 20.0
    while niter<20:
        niter += 1
            
        mu_mid = 0.5*(mu_left+mu_right)
        args_mid =  {'D':c*Dhom, 'mu':mu_mid}
        if string=='mean':
            T_mid = 1.0/LIF_fn(tref, v_r, v_t, **args_mid)
        
        if string=='mode':
            pISI, timeISI, dens = get_densISI(c*Dhom, mu_mid, tref = tref, tol = 1e-3)
            T_mid = timeISI[np.argmax(pISI)]
        
        err_mu = np.abs(old-T_mid)
        err = np.abs(T_mid-targetT)
        
        if (T_mid-targetT)>0:
            mu_left = mu_mid
            T_left = T_mid
        else:
            mu_right = mu_mid
            T_right = T_mid
        
        old = T_mid
        if err<tol or err_mu<tol_mu:
            break
    mu_sol = mu_mid
    if niter>20:
        #mu_sol = np.nan
        print('maximum iterations reached')
    return(mu_sol, err, T_mid)
    
def find_hetD(Dhom, muA, tref, string):
    '''
    Find the right mu of the heterogeneous population
    that leads to the same mean ISI as in the homogeneous 
    population
    '''
    args = {'D':Dhom, 'mu':muA}
    v_r = 0.0
    v_t = 1.0
    if string=='mean': 
        targetT = 1.0/LIF_fn(tref, v_r, v_t, **args)
    elif string=='mode':
        pISI, timeISI, dens = get_densISI(Dhom, muA, tref = tref, tol = 1e-3)
        targetT = timeISI[np.argmax(pISI)]
        
   # if verbos:
    #print('Target: '+str(targetT))
    D_left = 1e-7    
    D_right = 100.0*Dhom
    T=20
    dt = 0.005
    time=np.arange(0,T,dt)
    
    g_left = LIF_ISI_mu1(D_left, tref, T, dt)
    g_right = LIF_ISI_mu1(D_right, tref, T, dt)
    g_left[0]=0.0
    g_right[0]=0.0    
    if string=='mean':    
        T_left = np.sum(g_left*time)/np.sum(g_left)
        T_right = np.sum(g_right*time)/np.sum(g_right)
                
    elif string=='mode':
        T_left = time[np.argmax(g_left)]
        T_right = time[np.argmax(g_left)]
        
    #Start loop
    tol = dt/10.0
    tol_mu = 1e-4
    niter = 0
    err = 100.0
    err_mu = 100.0
    old = 20.0
    while niter<30:
        niter += 1
            
        D_mid = 0.5*(D_left+D_right)
        g_mid = LIF_ISI_mu1(D_mid, tref, T, dt)
        g_mid[0] = 0.0
        if string=='mean':
            T_mid = np.sum(g_mid*time)/np.sum(g_mid)
        
        if string=='mode':
            T_mid = time[np.argmax(g_mid)]
        
        err_mu = np.abs(old-T_mid)
        err = np.abs(T_mid-targetT)
        
        if (T_mid-targetT)>0:
            D_left = D_mid
            T_left = T_mid
        else:
            D_right = D_mid
            T_right = T_mid
        
        old = T_mid
        if err<tol or err_mu<tol_mu:
            break
    D_sol = D_mid
    if niter>29:
        #mu_sol = np.nan
        print('maximum iterations reached')
    return(D_sol, err, T_mid, targetT)
    
    
def get_deconv(Da, cee, muA, mu_m, tref, T, dt): 
    v_r = 0.0
    v_t = 1.0
    time = np.arange(0,T, dt)
    pISI, timeISI, dC = get_densISI(cee*Da, mu_m,  tref = 0.0,tol = 1e-4)
    tarT = timeISI[np.argmax(pISI)]
    print('Mode (no tref):', tarT)
    
    freq = np.fft.fftfreq(len(time))/dt
    
    delta1 = (v_r**2-v_t**2 + 2.0*muA*(v_t-v_r))/(4.0*Da)
    delta2 = (v_r**2-v_t**2 + 2.0*mu_m*(v_t-v_r))/(4.0*cee*Da)    
    fracvT = (muA-v_t)/np.sqrt(Da)
    fracvR = (muA-v_r)/np.sqrt(Da)
    fracvT2 = (mu_m-v_t)/np.sqrt(cee*Da)
    fracvR2 = (mu_m-v_r)/np.sqrt(cee*Da)
    omega = 2.0*np.pi*freq
    omega[0] = 1e-5
    dens_re = np.zeros(len(freq))
    dens_im = np.zeros(len(freq))
    dens_comp = 1j*np.zeros(len(freq))
    dens_approx = 1j*np.zeros(len(freq))
    dens_approx2 = 1j*np.zeros(len(freq))
    
    for i in range(len(freq)/2):

        
        dens = mp.exp(delta1)*mp.pcfd(1j*omega[i], fracvR)/mp.pcfd(1j*omega[i], fracvT)                    
        dens2 = mp.exp(delta2)*np.exp(-1j*omega[i]*tarT)*mp.pcfd(1j*omega[i], fracvR2)/mp.pcfd(1j*omega[i], fracvT2)
                    
        sol = dens/np.conj(dens2)
        if i/50 == 1/50.0:
            print(i)
        if np.abs(sol)<1e-4:
            break

        dens_re[i] = sol.real
        dens_re[-i]= sol.real
        dens_im[i] = sol.imag        
        dens_im[-i] = sol.imag
        first = 0
        if i>0 and np.abs(dens_re[i-1]+1j*dens_im[i-1])>np.abs(sol):
            dens_approx[i] = sol.real+1j*sol.imag
            dens_approx[-i] = sol.real+1j*sol.imag
            dens_approx2[i] = sol.real+1j*sol.imag
            dens_approx2[-i] = sol.real+1j*sol.imag
        else:
            if i>0:
                first +=1
                if first == 1:
                    I = i
                dens_approx2[i] = dens_approx[I-1]*mp.exp(-0.01*(omega[i]-omega[I])**2)
                dens_approx2[-i] = dens_approx2[i]

                if np.abs(dens_approx2[i])<1e-10:
                    break
                        
    if len(freq)/2!= len(freq)/2.0:
        if tref == 0.0:
            dens = mp.exp(delta1)*mp.pcfd(1j*omega[i], fracvR)/mp.pcfd(1j*omega[i], fracvT) 
            dens2 = mp.exp(delta2)*np.exp(-1j*omega[i]*T_mid)*mp.pcfd(1j*omega[i], fracvR2)/mp.pcfd(1j*omega[i], fracvT2)
        else:
            dens = mp.exp(delta1)*np.exp(1j*omega[i]*tref)*mp.pcfd(1j*omega[i], fracvR)/mp.pcfd(1j*omega[i], fracvT)                    
            dens2 = mp.exp(delta2)*np.exp(1j*omega[i]*tref)*np.exp(-1j*omega[i]*T_mid)*mp.pcfd(1j*omega[i], fracvR2)/mp.pcfd(1j*omega[i], fracvT2)
        dens_re[i+1]=sol.real
        dens_im[i+1]=sol.imag
    dens_comp = dens_re+1j*dens_im
    return dens_re, dens_im, dens_comp, dens_approx, dens_approx2
    
def LIF_ISI_mu1(D, tref, T, dt):
    t = np.arange(0,T,dt)
    fac1 = (2*np.exp(-t))/np.sqrt(2*np.pi*D*(1-np.exp(-2*t))**3)
    fac2 = np.exp(-np.exp(-2*t)/(2*D*(1-np.exp(-t))))
    g = fac1*fac2
    g[0]=0.0
    G = np.roll(g,np.sum(t<tref)-1)
    return(G)
    
def giveISIs(spks, dt_spk):
    lis=[]
    for n in range(np.shape(spks)[1]):
        idx, = np.where(spks[:,n]==1)
        intervals = np.diff(idx)*dt_spk
        lis.extend(intervals)
    return(lis)