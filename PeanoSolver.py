# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:27:07 2022

@author: Vviik

Solve plasma distribution of continuously ionized plasma from initially neutral coma

via the method of Peano et al. (2006): Smooth electrons, finite protons

"""

import numpy as np
import random
import matplotlib.pyplot as plt



# constants
pi = np.pi
r_comet = 1e3 # [m]
v_ion = 6e2 # m/s, all ions are born with 600 m/s radial velocity
electrontemperature = 10 # eV
beta = 147.8215 # electron to ion energy ratio: kT/(Mv^2)

# normalization, tbd. 
u_ion = 1
x_comet = 1
x_thickness = 1
Del_t = 0.1 # timestep between ionization bursts, unitless. 1 is time for unaccelerated ions to traverse one shell width.

#-----------------------------------Creation-----------------------------------

# 1.1 General parameter choices
number_of_shells = int(1e2)
x_k = [x_comet+x_comet*k for k in range(number_of_shells+1)] # every equally thick shell has an equal number of ionization events per unit time. Unitless
x_k_i = x_k[:-int(len(x_k)/2)] # shells in which we consider ionization

nu = 1e-6  # ionization frequency [number/s], Vigren2013a (Table 5. 9.2e-7 at solar maximum)

# 1.2 defining functions
def ioncreation(n, final_k): # creates n ions, equally many in each shell, up to shell number final_k
    n_per_shell = int(n/(final_k+1))
    ilist = []
    for k in range(final_k+1):
        ilist += ionshellcreation(n_per_shell, k)
    return ilist    
    
def ionshellcreation(n, k): # creates n ions uniformly distributed in the k-th shell
        xmin = x_k[k]
        xmax = x_k[k+1]
        x_ion = []
        for i in range(n): 
            x_ion += [random.uniform(xmin, xmax)]
        x_ion.sort()
        ilist = [[x, u_ion, k] for x in x_ion] # when k is given as a function argument
        return ilist

# 1.3 Randomly generating the ions and electrons
# ions
n_ion = 1e3 
ionlist = ioncreation(n_ion, len(x_k_i)) # Position (sorted) and velocity of all ions.
if n_ion > 1000:
    plt.figure()
    plt.hist([x[0] for x in ionlist], bins = 100, density = True)
    plt.hist([x[0] for x in ionlist], bins = 10, density = True, fill = False)
    plt.xlabel('Distance from comet nucleus [m]')
    plt.ylabel('Number of ions (pdf normalization)') # Normalization: area of all bins = 1
    plt.title('Radial distribution of randomly generated ions')
    
# electrons 

# 1.4 Creating the first potential

Q = 1e25 # [s-1], number of neutrals per second leaving the comet surface
N_R = Q/(4*np.pi*r_comet**2*v_ion) # [m-3], neutral density at the comet surface

# phi(r) = k Te/e * ln(nu_R dt N_R (r/R)**2) # 
phi_at_comet = 1e-2 # PLACEHOLDER. CALCULATE
phi0 = [phi_at_comet*(x_comet/x)**2 for x in x_k] # THIS SHOULD BE IN x_k_i INSTEAD BUT THEN THE LIST IS TOO SHORT.
phi_anders = [-phi_at_comet*x for x in x_k]

#----------------------------------Ion motion----------------------------------

# 2.1 Function definitions
def ElectricField(philist):
    return [-(philist[k+1]-philist[k])/(x_k[k+1]-x_k[k]) for k, _ in enumerate(x_k[:-1])] # ElectricField list has one less element. (Number of shells between points = Number of points - 1)

def timeevaluator(v0, a, s, s_alt): # Handles complex roots but requires both s_inner and s_outer
    if abs(2*a*s)<abs(0.01*v0**2): # abs(2*a*s/v0**2)<0.01
        t = s/v0-a*s**2/(2*v0**3) # 2nd order Taylor expanded
    else:
        t = v0/a*(-1+(1+2*a*s/v0**2)**(1/2))
    if type(t) == complex():
        return(timeevaluator(v0, a, s_alt, s)) # swap order
    if t < 0: # The wrong root is calculated, calculate the right root from symmetry of the quadratic.
        return -t-2*v0/a, s
    else:
        return t, s

# Recursive method
def ionmotion(ion, Delta_t, Elist): # ion = [r_ion, v_ion, shell_number], rlist is the limits of all shells. Elist is the electric field in all shells
    alpha = Elist[ion[2]] # unitless acceleration 
    a = alpha*beta # rescaled motion equation has both an alpha and a beta in front of the (unitless) time tau. 
    
    s_inner = x_k[ion[2]]-ion[0] # (signed) distance to border of inner shell
    s_outer = x_k[ion[2]+1]-ion[0] # (signed) distance to border of outer shell
    
    # inner_root_term = 2*a*s_inner/ion[1]**2
    # outer_root_term = 2*a*s_outer/ion[1]**2
    
    if ion[1] > 0:
        crossing_t, s = timeevaluator(ion[1], a, s_outer, s_inner)
    else:
        crossing_t, s = timeevaluator(ion[1], a, s_inner, s_outer)
    if crossing_t > Delta_t: 
        new_x = ion[0] + ion[1]*Delta_t + a*Delta_t**2/2
        return [new_x, ion[1]+a*Delta_t, ion[2]]
    else:
        crossing_v = ion[1]+a*crossing_t
        crossing_x = s+ion[0]
        new_shell = ion[2]+int(np.sign(s)) # -1 if s_inner was used, +1 if s_outer was used
        if new_shell == -1: # handles inner BC where the ion reaches the comet surface
            new_shell = 0
            crossing_v = -crossing_v
        if new_shell == number_of_shells: # handles outer BC where the ion reaches past space discretization
            new_shell = number_of_shells-1
        return ionmotion([crossing_x, crossing_v, new_shell], Delta_t-crossing_t, Elist)
    
def ioncount(ilist):
    i_per_shell = []
    for k in range(number_of_shells):
        i_per_shell += [[ion[2] for ion in ilist].count(k)]
    return i_per_shell
    
def iondensity(i_per_shell):
    i_numberdensity = []
    for k in range(1, number_of_shells):
        i_numberdensity += [1/(4*pi*(r_comet*x_k[k])**2)*(i_per_shell[k]+i_per_shell[k-1])/(2*x_comet*r_comet)]
    return i_numberdensity

# 2.2 Calculating the new ionlist
    
new_ionlist = [ionmotion(ion, Del_t, ElectricField(phi0)) for ion in ionlist]
new_ionlist.sort() # new line is needed for some reason

plt.figure()
plt.plot([r_comet*x[0] for x in ionlist], color='k')
plt.plot([r_comet*x[0] for x in new_ionlist], color='c', linestyle=':')
plt.title('New vs. old ion position')
plt.xlabel('Ion index')
plt.ylabel('Comet distance [m]')

plt.figure()
plt.plot([v_ion*x[1] for x in ionlist], color='k')
plt.plot([v_ion*x[1] for x in new_ionlist], color='c', linestyle=':')
plt.title('New vs. old ion speed')
plt.xlabel('Ion index')
plt.ylabel('Ion speed [m/s]')

# 2.3 Ion density calculation
# Count number of ions in each shell 
new_ions_per_shell = ioncount(new_ionlist)
ions_per_shell = ioncount(ionlist)

plt.figure()
plt.plot(new_ions_per_shell, '.', color='C0', label = 'New ion list')
plt.plot(ions_per_shell, '-', color='k', label = 'Old ion list')
plt.title('Number of ions inside each shell')
plt.xlabel('Shell number '+r'$k$')
plt.ylabel('Number of ions')
plt.legend()

ion_numberdensity = iondensity(ions_per_shell)

plt.figure()
plt.plot(x_k[1:-1], ion_numberdensity, color = 'k')
plt.title('Ion number density in each shell')
plt.xlabel('Comet distance [m]')
plt.ylabel('3D ion number density ' +r'[m$^{-3}$]')

# 2.4 Anders Loop
andersfunction = [np.sqrt(1+2*(x-1)*phi_at_comet)/(x**2*phi_at_comet) for x in x_k]
number_of_loops = int(len(x_k_i)/Del_t) # number of loops it takes for ions born at the comet to reach the edge of the ionization
for j in range(number_of_loops):
    source_ions = ioncreation(n_ion, len(x_k_i))
    ionlist = new_ionlist+source_ions
    ionlist.sort()
    
    plt.figure()
    plt.plot(iondensity(ioncount(ionlist)), '.', color='k')
    # plt.plot(andersfunction, color='k')
    plt.title('Number of ions inside each shell')
    plt.xlabel('Shell number '+r'$k$')
    plt.ylabel('Number of ions')
    
    new_ionlist = [ionmotion(ion, Del_t, ElectricField(phi_anders)) for ion in ionlist]

#------------------Electron Energy evolution and dist. func.-------------------

