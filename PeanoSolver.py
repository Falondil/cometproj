# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:27:07 2022

@author: Vviik

Solve plasma distribution of continuously ionized plasma from initially neutral coma

via the method of Peano et al. (2006): Smooth electrons, finite protons

"""

import numpy as np
import random as random
import matplotlib.pyplot as plt

# normalization, tbd. 
q = 1 
m = 1

# constants
pi = np.pi
C0 = 16/3*pi**2*(2*m)**3/2

r_comet = 1e3 # [m]

#-----------------------------------Creation-----------------------------------

# 1.1 defining functions
def ionizationevents(Delta_t, nu): 
    lam = Delta_t*nu
    return random.gauss(lam, lam**(1/2)) # Poisson of a large lambda is well approximated by a Gaussian

def ioncreation(n, rmax): # 
        r_ion = []
        for i in range(n): 
            r_ion += [r_comet+rmax*random.random()]
        r_ion.sort()
        ilist = [[r, v_ion] for r in r_ion]
        return ilist

def electroncreation(n, kBT):
    elist = []
    # Maxwell-Boltzmann energy distribution is the Gamma distribution with Shape = 3/2 and Scale = k*T
    shape = 3/2
    scale = kBT
    for i in range(n):
        elist += [random.gammavariate(shape, scale)]
    elist.sort()
    return elist


# 1.2 General parameter choices
Del_t = 0.01 # timestep between ionization bursts [s], PLACEHOLDER VALUE. 
shell_thickness = 1e3 # [m]
shell_number = int(1e3)
r_k = [r_comet+shell_thickness*k for k in range(shell_number+1)] # every equally thick shell has an equal number of ionization events per unit time, go out to 1000 km 
r_k_i = r_k[:-int(len(r_k)/2)] # shells in which we consider ionization
nu = 1e6 # ionization frequency [number/s], PLACEHOLDER VALUE. 
v_ion = 1e3 # m/s, all ions are born with 1 km/s radial velocity
electrontemperature = 10 # eV
n_ion = int(ionizationevents(Del_t, nu)) # randomize number of ionization events

# 1.3 Randomly generating the ions and electrons
# ions
ionlist = ioncreation(n_ion, r_k_i[-1]) # Position (sorted) and velocity of all ions.
# [x[0] for x in ionlist] for a list of only the r's. 
plt.figure()
plt.hist([x[0] for x in ionlist], bins = int(n_ion/100)+1, density = True)
plt.hist([x[0] for x in ionlist], bins = int(n_ion/1000)+1, density = True, fill = False)
plt.xlabel('Distance from comet nucleus [m]')
plt.ylabel('Number of ions (pdf normalization)') # Normalization: area of all bins = 1
plt.title('Radial distribution of randomly generated ions')

# electrons 
electronlist = electroncreation(n_ion, electrontemperature) # energy of the electrons, same number as electrons
plt.figure()
plt.hist(electronlist, bins = int(n_ion/100), density = True)
plt.hist(electronlist, bins = int(n_ion/1000), density = True, fill = False)
plt.xlabel('Electron energy [eV]')
plt.ylabel('Number of electrons (pdf normalization)') # Normalization: area of all bins = 1
plt.title('Energy distribution of randomly generated electrons')

# 1.4 Creating the first potential
phi_at_r_comet = 1e7 # PLACEHOLDER. CALCULATE VIA ANDERS PICTURE ON WHITEBOARD
phi0 = [(phi_at_r_comet/r)**2 for r in r_k] # THIS SHOULD BE IN r_k_i INSTEAD BUT THEN THE LIST IS TOO SHORT.


#----------------------------Potential calculation (SCRAP)---------------------
# Count the number of ions inside of each r_k
enc_ions = []
for r in r_k:
    enc_ionlist = [i+1 for i, x in enumerate([x[0] for x in ionlist]) if x<r] # list of the ions that are inside the r_k shell.
    enc_ions += [0] if not enc_ionlist else [enc_ionlist[-1]] # last ion that is inside the r_k shell

plt.figure()
plt.plot(r_k, enc_ions, color = 'k', linestyle='-')
plt.title('Ions enclosed inside each shell')
plt.ylabel('Number of ions')
plt.xlabel('Comet distance [m]')

# Calculate the electron part of the enclosed charge


# Qenc = e*(enc_ions-enc_electrons)

# DO NOT CALCULATE THE POTENTIAL. INSTEAD USE THE POTENTIAL TO FIX THE ELECTRON ENERGY DISTRIBUTION (ENFORCE THAT \rho_e = \rho_i EVERYWHERE).

#----------------------------------Ion motion----------------------------------
# kastparabel för jonerna. s = v_0*Delta_t + a*Delta_t**2/2 där a är eq. 17.  Ny v_ion i slutet.
# Om en jon börjar nära kanten av ett skal och rör sig till det andra måste accelerationen ändras mitt i. 
# Så då kanske man måste ta och flippa på equationen. 
# Först hitta det t>0 som löser att s blir till nästa skal, jämför om t>Delta_t
# Om ja: beräkna som vanligt med jonen behåller sig i skalet. 
# Om nej: beräkna att jonen hamnar i nya skalet och beräkna rörelsen där. 
# Jag tror inte att det blir problem med att jonen kommer byta håll i skalövergången så att man flippar för mycket.
# Om det hade blivit ett problem skulle man kunna fixa det genom att kolla om den byter tecken och har låg hastighet
# och då bara sätta den imellan skalen vid nästa tidssteg.

def ElectricField(philist, rlist):
    return [-(philist[k+1]-philist[k])/(rlist[k+1]-rlist[k]) for k, _ in enumerate(rlist[:-1])] # ElectricField list has one less element. (Number of shells between points = Number of points - 1)

somethingtiny = 0.01 # PLACEHOLDER 
# Recursive method
def ionmotion(ion, Delta_t, rlist, Elist): # ion = [r_ion, v_ion], rlist is the limits of all shells. Elist is the electric field in all shells
    shell_k = max([i for i, r in enumerate(rlist) if r<ion[0]]) # the shell number that the ion is in
    # shell_k = int(ion[0]/shell_thickness) # also works, is quicker (and means we don't need rlist in the arguments) but does not work if the shell_thickness isn't constant and starts at 0. Does this handle the recursive placement?  
    a = q/m*Elist[shell_k]
    new_r = ion[0] + ion[1]*Delta_t + a*Delta_t**2/2
    peak_occurs = -Delta_t*a/ion[1]>1 # criteria for having the parabola peak inside Delta_t.
    if peak_occurs: 
        t_peak = -ion[1]/a 
        peak_r = ion[0] + ion[1]*t_peak + a*t_peak**2/2
    else:
        peak_r = ion[0] # dummy variable that never fulfills either crossing criteria
    if new_r > rlist[shell_k+1] or peak_r > rlist[shell_k+1]:
        # find the time for the crossing, crossing_t
        s = rlist[shell_k+1]-ion[0]
        if a/ion[1] < 0.01: # arbitrary << 1 choice. PLACEHOLDER
            crossing_t = s/ion[1]-a*s**2/(2*ion[1]**3)
        else: 
            phalf = ion[1]/a
            root = np.sqrt(phalf**2+2/a*s)
            crossing_t = min([t for t in [-phalf+root, -phalf-root] if t > 0]) # finds smallest time that is positive (realistic).
        return ionmotion([rlist[shell_k+1]+somethingtiny, ion[1]+a*crossing_t], Delta_t-crossing_t, rlist, Elist)
    elif new_r < rlist[shell_k] or peak_r < rlist[shell_k]:
        # find the time for the crossing, crossing_t
        s = rlist[shell_k]-ion[0]
        if a/ion[1] < 0.01:
            crossing_t = s/ion[1]-a*s**2/(2*ion[1]**3)
        else:
            phalf = ion[1]/a
            root= np.sqrt(phalf**2+2/a*s)
            crossing_t = min([t for t in [-phalf+root, -phalf-root] if t > 0])
        return ionmotion([rlist[shell_k]-somethingtiny, ion[1]+a*crossing_t], Delta_t-crossing_t, rlist, Elist)
    else: 
        return [new_r, ion[1]+a*Delta_t]
    
# och sen är schemen new_ionlist = [ionmotion(x, andra_input_argument) for x in ionlist].sort()

# istället för en rekursiv funktion skulle den kunna returnerna Delta_t - time elapsed och ifall det är 0 så är det fine annars får man i main koden ankalla funktionen igen pga en crossing hände. Då skulle funktionen inte behöva ta hela rlist eller Elist. 

new_ionlist = [ionmotion(x, Del_t, r_k, ElectricField(phi0, r_k)) for x in ionlist] #.sort

plt.figure()
plt.plot([x[0] for x in ionlist], color='k')
plt.plot([x[0] for x in new_ionlist], color='c', linestyle=':')
plt.title('New vs. old ion position')
plt.xlabel('Ion index')
plt.ylabel('Comet distance [m]')

plt.figure()
plt.plot([x[1] for x in ionlist], color='k')
plt.plot([x[1] for x in new_ionlist], color='c', linestyle=':')
plt.title('New vs. old ion speed')
plt.xlabel('Ion index')
plt.ylabel('Ion speed [m/s]')

# def ionsbetween(ilist, rmin, rmax): # finds all ions between rmin and rmax
#     shellions = [x for x in ilist if x[0] >= rmin and x[0] <= rmax]
#     return shellions    

# new_ionlist = []
# Elist = [0 for k in range(shell_number)] # just now while working. This has to be fixed for each E(r_k). Or maybe even piecewise linear so we don't have parabolas. 
# for k in range(shell_number):
#     ions_r_k = ionsbetween(ionlist, r_k[k], r_k[k+1])
#     a = q/m*Elist[k]
#     for ion in ions_r_k:
#         new_r = ion[0] + ion[1] * Del_t + a*Del_t**2/2 # x(t) = x0 + v0*t + a*t**2/2
        
#         #THESE TWO HAVE TO BE FIXED. SOLVE FOR t SOME OTHER WAY.
#         if new_r > r_k[k+1]: 
#             phalf = ion[1]/a # p/2 from the pq-formula
#             root = np.sqrt(phalf**2-2/a*(ion[0]-r_k[k+1])) # root from the pq-formula
#             crossing_t = max(-phalf+root, -phalf-root) # wrong approach. Not max. 
#         elif new_r < r_k[k]:
#             phalf = ion[1]/a # p/2 from the pq-formula
#             root = np.sqrt(phalf**2-2/a*(ion[0]-r_k[k])) # root from the pq-formula
#             crossing_t = max(-phalf+root, -phalf-root) # wrong approach. Not max. 
            
#         else: # if new_r > r_k[k] and new_r < r_k[k+1]:
#             new_ionlist += [[new_r, ion[1]+a*Del_t]] # v(t) = v0 + a*t
    


    






#------------------Electron Energy evolution and dist. func.-------------------

#REMAKE EVERY FUNCTION. POSITION OF ELECTRONS IS DISCRETE? 

# def inv_sq(r):
#     return 1/r**2

# def ergodic_inv(eps, phi, rvec): # eq. 75. phi is a vector of all phi_r_k. Needed after del_eps is calculated to find new distribution func/new Poisson calc. Or is it?
#     return C0*sum([(eps-q*phi[k])**(3/2)*rvec[k]**2*delta_vec(rvec) for k in range(len(rvec))])

# def dJdeps(eps, phi, rvec): # eq. 74. rvec is a vector of all r_k up to R(eps)
#     return 3/2*C0*sum([(eps-q*phi[k])**(1/2)*rvec[k]**2*delta_vec(rvec) for k in range(len(rvec))])

# def dJdphi(eps, phi, rvec, delta_phi): # eq. 76. rvec is a vector of all r_k. 
#     return C0*3/2*-q*sum([(eps-q*phi[k])**(1/2)*rvec[k]**2*delta_vec(rvec)*del_phi for k in range(len(rvec))])

# def del_phi(new_phi, old_phi): # might be excessive to have as a function. Just have a variable that is overwritten all the time.
#     return [new_phi[i]-old_phi[i] for i in range(len(new_phi))]

# def del_eps(eps, phi, rvec, delta_phi, del_phi): # eq. 77. all but eps are vectors
#     return -dJdphi(eps, phi, delta_phi, rvec)/dJdeps(eps, phi, rvec)





