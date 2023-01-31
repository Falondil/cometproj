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



#-----------------------------------Creation-----------------------------------

# 1.1 defining functions
def ionizationevents(Delta_t, nu): 
    lam = Delta_t*nu
    return random.gauss(lam, lam**(1/2)) # Poisson of a large lambda is well approximated by a Gaussian

def ioncreation(n, rmax): # 
        r_ion = []
        for i in range(n): 
            r_ion += [rmax*random.random()]
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
Del_t = 1 # timestep between ionization bursts
shell_thickness = 1e3 # m
r_k = [shell_thickness*k for k in range(int(1e3))] # every equally thick shell has an equal number of ionization events per unit time, go out to 1000 km 
nu = 1e4 # ionization frequency [number/s], PLACEHOLDER VALUE. 
v_ion = 1e3 # m/s, all ions are born with 1 km/s radial velocity
electrontemperature = 10 # eV
n_ion = int(ionizationevents(Del_t, nu)) # randomize number of ionization events


# 1.3 Randomly generating the ions and electrons
# ions
ionlist = ioncreation(n_ion, r_k[-1]) # Position (sorted) and velocity of all ions.
# [x[0] for x in ionlist] for a list of only the r's. 
plt.figure()
plt.hist([x[0] for x in ionlist], bins = int(n_ion/100), density = True)
plt.hist([x[0] for x in ionlist], bins = int(n_ion/1000), density = True, fill = False)
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

#----------------------------Potential calculation-----------------------------
# Count the number of ions inside of each r_k to get Phi_r_k.
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





