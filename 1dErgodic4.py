# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 09:04:29 2022

@author: Vviik
"""
import numpy as np
pi = np.pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random

# derives the vector u w.r.t. time. 
def derivative(t, u): # u = [v, x]
    v, x = u
    omega2 = omegafun(t)
    return [-omega2*x, v] # u' = [v', x']

errors = []
relerrors = []
signres = []
periodlist = [10*1.1**x for x in range(50)]
# periodlist = [10]
nbounces = 10 
tstop = 2*pi*10 # 2*pi is one period for initial omega. Use 2*pi*nbounces for a consistent bounce frequency
# INFÖR EN TBOUNCE FÖR VARJE STUDS OCH INTEGRERA PIECEWISE
tbounces = [tstop*random.random() for i in range(nbounces)]
tbounces.sort()
tbounces += [tstop]

for periods in periodlist:
    # function for omega squared
    def omegafun(t):
        return np.exp(-np.log(2)*t/(2*pi*periods))
        # return 1 # For 10 periods = tstop, constant relerror of 3.4e-8
    
    # plt.figure()
    # plt.plot(tlist, [omegafun(t) for t in tlist])
    # plt.title('Potential evolution')
    # plt.xlabel('time')
    # plt.ylabel(r'Potential $[2U/mx^2]$')
    
    x = []
    v = []
    tlist = []
    
    for i in range(nbounces+1):
        tb = tbounces[i]
        tbprevious = tlist[-1] if i>0 else 0
        timestep = 2*pi/100 # Increase denominator for higher accuracy
        IC = [-v[-1], x[-1]] if i>0 else [0, 1] 
        sol = solve_ivp(derivative, [tbprevious, tb], IC, max_step=timestep)
        
        x += list(sol.y[1])
        v += list(sol.y[0])
        tlist += list(sol.t)
    
    # plt.figure()
    # plt.plot(tlist, x, '-', label='x(t)', color = 'C0')
    # plt.plot(tlist, v, '-', label='v(t)', color = 'C1')
    # plt.title('Oscillations in time')
    # plt.xlabel('Time')
    # plt.ylabel('x or v')
    # plt.grid()
    # plt.legend()
    
    # plt.figure()
    # plt.plot(tlist, x, '-', label='x(t)', color = 'C0')
    # for tb in tbounces[:-1]:
    #     plt.axvline(tb, color='k')
    # plt.title('Marked bounce times')
    # plt.xlabel('Time')
    # plt.ylabel('x(t)')
    
    # plt.figure()
    # plt.plot(tlist, v, '-', label='v(t)', color = 'C1')
    # for tb in tbounces[:-1]:
    #     plt.axvline(tb, color='k')
    # plt.title('Marked bounce times')
    # plt.xlabel('Time')
    # plt.ylabel('v(t)')
    
    plt.figure()
    plt.plot(x, v, '-')
    plt.plot(x[0], v[0], 'x', color='k', label = 'Start')
    plt.plot(x[-1], v[-1], '.', color='k', label = 'Stop')
    plt.title('Phase space')
    plt.xlabel('x')
    plt.ylabel('v')
    plt.gca().axis('equal')
    plt.legend()
    upjumps = []
    downjumps = []
    for i, w in enumerate(v[1:]):
        if w - v[i] > 0.25:
            upjumps += [i]
        elif w - v[i] < -0.25:
            downjumps += [i]
    for i in upjumps:
        plt.arrow((x[i]+x[i+1])/2, 0, 0, 0.1, width = 0.02, color = 'C0')
    for i in downjumps:
        plt.arrow((x[i]+x[i+1])/2, 0, 0, -0.1, width = 0.02, color = 'C0')
    
    # plt.savefig('Plots/10Phi10B.pdf')
    
    
    Estop = v[-1]**2+omegafun(tlist[-1])*x[-1]**2 # find endpoint energy (*2/m)
    J = pi*Estop/(omegafun(tlist[-1]))**(1/2) # calculate endpoint area
    
    J0 = pi*x[0]**2*omegafun(0)**(1/2) # 1/2 m omega^2 x^2 = 1/2 m v^2 ==> omega x = v
    error = abs(J-J0)
    signre = (J-J0)/J0
    relerror = error/J0
    
    errors += [error]
    relerrors += [relerror]
    signres += [signre]

if len(periodlist)>1:
    # plt.figure()
    # plt.plot(periodlist, errors, '-', marker = '.', label='Number of bounces: '+str(nbounces))
    # plt.title('Errors as a function of number of periods')
    # plt.xlabel('Periods over which the potential is halved')
    # plt.ylabel('Errors')
    
    A = [np.log(a) for a in periodlist]
    B = [np.log(b) for b in relerrors]
    
    linpoly = np.polyfit(A, B, 1)
    scaling = linpoly[0]
    rsquared = np.corrcoef(A, B)[0,1]**2
    
    filestart = str(nbounces)+'b1e'+str(int(np.log10(periodlist[0])))+'-1e'+str(int(np.log10(periodlist[-1])))
    
    plt.figure()
    plt.loglog(periodlist, relerrors,'-',marker='.',label='Coef = '+"{:.4}".format(scaling)+'\n R^2 = '+"{:.4}".format(rsquared))
    plt.ylim([1e-6, 1e-1])
    plt.title('Relative errors as a function of number of periods')
    plt.xlabel('Periods over which the potential is halved')
    plt.ylabel('Relative errors')
    plt.legend()
    # plt.savefig('Plots/'+filestart+'loglog.pdf')
    
    plt.figure()
    plt.semilogy(periodlist, relerrors,'-',marker='.')
    plt.title('Relative errors as a function of number of periods')
    plt.xlabel('Periods over which the potential is halved')
    plt.ylabel('Relative errors')
    # plt.savefig('Plots/'+filestart+'linlog.pdf')
    
    plt.figure()
    plt.plot(periodlist, relerrors,'-',marker='.')
    plt.title('Relative errors as a function of number of periods')
    plt.xlabel('Periods over which the potential is halved')
    plt.ylabel('Relative errors')
    # plt.savefig('Plots/'+filestart+'linlin.pdf')
    
    plt.figure()
    plt.plot(periodlist, signres,'-',marker='.')
    plt.title('Signed relative errors as a function of number of periods')
    plt.xlabel('Periods over which the potential is halved')
    plt.ylabel('Relative errors')
    
