# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:45:49 2023

@author: Vviik
"""

import numpy as np
import matplotlib.pyplot as plt

# constants
pi = np.pi
r_comet = 1e3 # [m]
v_n = 6e2 # m/s, all ions are born with 600 m/s radial velocity
electrontemperature = 10 # eV
beta = 147.8215 # electron to ion energy ratio: kT/(Mv^2)
nu = 1e-6  # ionization frequency [number/s], Vigren2013a (Table 5. 9.2e-7 at solar maximum)

# normalization, tbd. 
u_n = 1
x_comet = 1
x_thickness = 1
Del_t = 0.1 # timestep between ionization bursts, unitless. 1 is time for unaccelerated ions to traverse one shell width.

#-----------------------------------Creation-----------------------------------

# 1.1 General parameter choices
number_of_shells = int(1e2)
x_k = np.arange(x_comet, number_of_shells+x_comet, dtype=np.float64) # every equally thick shell has an equal number of ionization events per unit time. Unitless
x_k_i = x_k[:-int(len(x_k)/2)] # shells in which we consider ionization

# 1.2 defining functions
def ioncreation(n, final_k): # creates n ions, equally many in each shell, up to shell number final_k
    n_per_shell = int(n/(final_k+1)) # this is rounded down. n_per_shell * number_of_shells <= n
    ilist = np.empty((0, 3)) # empty 0-by-3 matrix
    for k in range(final_k+1):
        ilist = np.concatenate((ilist, ionshellcreation(n_per_shell, k)), axis=0)
    return ilist

def ionshellcreation(n, k): # creates n ions uniformly distributed in the k-th shell
        ilist = np.empty((n, 3)) # creates an empty n-by-3 matrix 
        ilist[:, 1] = u_n # 1st column stores velocity
        ilist[:, 2] = k # 2nd column stores shell number
        
        xmin = k+x_comet
        x_ion = xmin+np.array([np.random.rand(n)])
        x_ion.sort()
        x_ion.reshape((n,1)) # make the x_ion array a column vector
        ilist[:, 0] = x_ion # 0th column stores position
        return ilist
    
# 1.3 Randomly generating the ions and electrons
# ions
n_ion = 1e3 
ionmatrix = ioncreation(n_ion, int(x_k_i[-1])) # Position (sorted) and velocity of all ions.
if n_ion > 1000:
    plt.figure()
    plt.hist(ionmatrix[:, 0], bins = 100, density = True)
    plt.hist(ionmatrix[:, 0], bins = 10, density = True, fill = False)
    plt.xlabel('Distance from comet nucleus [km]')
    plt.ylabel('Number of ions (pdf normalization)') # Normalization: area of all bins = 1
    plt.title('Radial distribution of randomly generated ions')
    
# electrons 

# 1.4 Creating the first potential

Q = 1e25 # [s-1], number of neutrals per second leaving the comet surface
N_R = Q/(4*np.pi*r_comet**2*v_n) # [m-3], neutral density at the comet surface

# phi(r) = k Te/e * ln(nu_R dt N_R (r/R)**2) # 
phi_at_comet = 1e-4 # PLACEHOLDER. CALCULATE
# phi0 = FIX THIS
phi_anders = -phi_at_comet*x_k # numpy array

#----------------------------------Ion motion----------------------------------

# 2.1 Function definitions
def ElectricField(philist):
    return -(philist[1:]-philist[:-1])/(x_k[1:]-x_k[:-1]) # ElectricField list has one less element than phi. (Number of shells between points = Number of points - 1)
    
# Probably has to be remade too
def timeevaluator(v0, a, s, s_alt): # Handles complex roots but requires both s_inner and s_outer
    if abs(2*a*s)<abs(0.01*v0**2): # abs(2*a*s/v0**2)<0.01
        t = s/v0-a*s**2/(2*v0**3) # 2nd order Taylor expanded
    elif 1+2*a*s/v0**2 < 0:
        return(timeevaluator(v0, a, s_alt, s)) # swap order
    else:
        t = v0/a*(-1+(1+2*a*s/v0**2)**(1/2))
    if t < 0: # The wrong root is calculated, calculate the right root from symmetry of the quadratic.
        return -t-2*v0/a, s
    else:
        return t, s
    
def arraytimeevaluator(v0, a, s, s_alt): # Handles complex roots but requires both s_inner and s_outer
    t = np.empty(v0.shape)
    taylor_ind = (abs(2*a*s)<abs(0.01*v0**2)).nonzero() # abs(2*a*s/v0**2)<0.01
    t[taylor_ind] = s/v0-a*s**2/(2*v0**3) # 2nd order Taylor expanded
    elif 1+2*a*s/v0**2 < 0:
        return(timeevaluator(v0, a, s_alt, s)) # swap order
    else:
        t = v0/a*(-1+(1+2*a*s/v0**2)**(1/2))
    if t < 0: # The wrong root is calculated, calculate the right root from symmetry of the quadratic.
        return -t-2*v0/a, s
    else:
        return t, s
    
def ionmotion(imatrix, Delta_t, Elist): # calculates motion for every ion in ionmatrix. Each ion is a row in the n-by-3 ionmatrix [x_ion, u_ion, k_ion]
    pos = imatrix[:,0] # position of ions
    vs = imatrix[:,1] # velocity of ions
    ks = imatrix[:,2] # shell number of ions    
    
    a=np.zeros(ks.shape) # make a new acceleration vector (acting on each ion)
    for k in range(int(x_k[-1]-1)):
        a[(ks==k).nonzero()] = Elist[k]*beta
    
    # a = Elist*beta # rescaled motion equation has an alpha and a beta infront of the (unitless) time tau.
    
    pos_in_shell = pos%1 # calculates position inside each shell from [0, 1)
    s_inner = -pos_in_shell # distance (negative) to inner shell for each ion
    s_outer = 1-pos_in_shell # distance (positive) to outer shell for each ion
    
    # s_main: New array where s has the same direction as the velocity
    positive_vs_ind = (vs>0).nonzero()
    s_main = s_inner # (Read line below first) rest go to inner s
    s_main[positive_vs_ind] = s_outer[positive_vs_ind] # those with positive velocity go to outer s 
    
    # s_alt: New array where s has the opposite direction to the velocity
    s_alt = s_outer 
    s_alt[positive_vs_ind] =  s_inner[positive_vs_ind]
    
    lenvs = len(vs)
    crossing_t_and_s = np.empty((lenvs, 2))
    for ionnum in range(len(vs)):
        crossing_t_and_s[ionnum] = timeevaluator(vs[ionnum], a[ionnum], s_main[ionnum], s_alt[ionnum])
    
    crossing_t = crossing_t_and_s[:, 0]
    s = crossing_t_and_s[:, 1]
    
    # check for which indices a crossing happens/does not happen
    non_crossing_ind = (crossing_t>=Delta_t).nonzero()
    crossing_ind = (crossing_t<Delta_t).nonzero()
    
    # calculate new position and velocity for ions where crossing does not happen
    pos[non_crossing_ind] = pos[non_crossing_ind] + vs[non_crossing_ind]*Delta_t + a[non_crossing_ind]*Delta_t**2/2
    vs[non_crossing_ind] = vs[non_crossing_ind] + a[non_crossing_ind]*Delta_t
    
    # calculate new position and velocity and shell number for ions where crossing happens
    pos[crossing_ind] = pos[crossing_ind] + vs[crossing_ind]*crossing_t[crossing_ind] + a[crossing_ind]*crossing_t[crossing_ind]**2/2
    vs[crossing_ind] = vs[crossing_ind] + crossing_t[crossing_ind]
    ks[crossing_ind] = ks[crossing_ind]+np.sign(s[crossing_ind])
    
    # handle inner and outer BC
    IBC_ind = (ks==-1).nonzero()
    ks[IBC_ind] = 0 
    vs[IBC_ind] = -vs[IBC_ind] # bounce at comet surface
    
    OBC_ind = (ks>=number_of_shells).nonzero()
    ks[OBC_ind] = number_of_shells-1 # The outermost shell has infinite extent for now
    
    # create a new matrix of ions
    imatrixnew = np.array(list(zip(pos, vs, ks))) # recombine position, velocity and shell number 
    
    # ions which crossed have to be calculated again recursively
    # imatrixnew[crossing_ind] = ionmotion(imatrixnew[crossing_ind], Delta_t-crossing_t[crossing_ind], Elist)
    
    return imatrixnew 

def ioncount(imatrix):
    ks = imatrix[:,2]
    counts = np.empty((number_of_shells,))
    for k in range(number_of_shells):
        counts[k] = np.count_nonzero(ks==k)
    return counts

def iondensity(i_per_shell):
    i_numberdensity = 1/(4*pi*(r_comet*x_k[1:])**2)*(i_per_shell[1:]+i_per_shell[:-1])/(2*x_comet*r_comet)
    return i_numberdensity

# 2.4 Anders Loop. Supplementary, move to last. 
number_of_loops = int(len(x_k_i)/Del_t)
counter = 0
andersfield = ElectricField(phi_anders)
for j in range(number_of_loops):
    ionmatrix = ionmotion(ionmatrix, Del_t, andersfield)
    source_ions = ioncreation(n_ion, int(x_k_i[-1]))
    ionmatrix = np.concatenate((ionmatrix, source_ions))
    ionmatrix = ionmatrix[ionmatrix[:,2].argsort()] # sort after first column
    counter+=1
    
    # icount = ioncount(ionmatrix)
    # plt.figure()
    # plt.plot(icount[:-1], '.', color='k')
    # plt.title('Number of ions inside each shell')
    # plt.xlabel('Shell number '+r'$k$')
    # plt.ylabel('Number of ions')
    
    idensity = iondensity(ioncount(ionmatrix))
    plt.figure()
    plt.plot(idensity[:-1], '.', color='k')
    plt.title('Number density of ions')
    plt.xlabel('Shell number '+r'$k$')
    plt.ylabel('Number density [m'+'$^{-3}$]')
    

# sort a numpy matrix by 0th column a[a[:, 0].argsort()]
    
    
    