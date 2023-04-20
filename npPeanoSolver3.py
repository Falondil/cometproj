# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:45:49 2023

@author: Vviik
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# constants
pi = np.pi
r_comet = 1e3 # [m]
v_n = 6e2 # m/s, all ions are born with 600 m/s radial velocity
electrontemperature = 10 # eV
beta = 147.8215 # electron to ion energy ratio: kT/(Mv^2)
nu = 1e-6  # ionization frequency [number/s], Vigren2013a (Table 5. 9.2e-7 at solar maximum)
Q = 1e25 # [s-1], number of neutrals per second leaving the comet surface
N_R = Q/(4*pi*r_comet**2*v_n) # [m-3], neutral density at the comet surface

# normalization, tbd. 
u_n = 1
x_comet = 1
x_thickness = 1
Del_t = 0.1 # timestep between ionization bursts, unitless. 1 is time for unaccelerated ions to traverse one shell width.

#-----------------------------------Creation-----------------------------------

# 1.1 General parameter choices
number_of_shells = int(1e2) # number of shells for the spatial discretization
number_of_boundaries = number_of_shells+1 # number of shell boundaries and edgepoints
x_k = np.arange(x_comet, number_of_boundaries+x_comet, dtype=int) # every equally thick shell has an equal number of ionization events per unit time. Unitless
x_k_i = x_k[:-int(len(x_k)/2)] # shells in which we consider ionization

# 1.2 defining functions
def ioncreation(n_per_shell, final_k): # creates n ions, equally many in each shell, up to shell number final_k
    # n_per_shell = int(n/(final_k+1)) # this is rounded down.
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

# 1.3 Creating the first potential

# phi(r) = k Te/e * ln(nu_R dt N_R (r/R)**2) # 
phi_at_comet = 1e-2 # PLACEHOLDER. CALCULATE
initial_phi = [phi_at_comet*(x_comet/x)**2 if x in x_k_i else phi_at_comet*(x_comet/x_k_i[-1])**2 for x in x_k]
# ask Anders

phi_anders = -phi_at_comet*x_k # numpy array
phi_anders_log = -phi_at_comet*np.log(x_k)

# 1.4 Randomly generating the ions
n_per_shell = 3 # number of ions generated in each shell

n_ion_sim = n_per_shell # count the number of simulated ions created per shell each timestep
n_ion_real = nu*(r_comet/v_n)**2*Del_t*Q # count the number of ions that they represent
realsimratio = n_ion_real/n_ion_sim # the number of real ions each simulated ion represents

ionmatrix = ioncreation(n_per_shell, int(number_of_shells)) # Position (sorted) and velocity of all ions.
if n_ion_sim > 1000:
    plt.figure()
    plt.hist(ionmatrix[:, 0], bins = 100, density = True)
    plt.hist(ionmatrix[:, 0], bins = 10, density = True, fill = False)
    plt.xlabel('Distance from comet nucleus [km]')
    plt.ylabel('Number of ions (pdf normalization)') # Normalization: area of all bins = 1
    plt.title('Radial distribution of randomly generated ions')
    
    
# 1.5 Electrons
excess = 2*electrontemperature # how many electrontemperatures we consider before truncation
eps, deps = np.linspace(-excess, excess, number_of_boundaries, retstep=True) # centered on 0? 

# Function definitions
def V(eps, phi): # Calculates unitless sqrt of electron kinetic energy. Either eps or phi can be numpy array.
    x = eps+phi    
    return ((x+abs(x))/2)**(1/2) # returns 0 if V is imaginary

def Vmatrix(epslist, philist): # Calculates matrix of values of V. epslist and philist are both 1D numpy arrays
    phi, eps = np.meshgrid(philist, epslist, sparse=True) # creates mesh of phi, eps values
    return V(eps, phi) # returns 2D array where Vmat[i,k] = V(epslist[i], philist[k])

def Fper2V(F0, V0, deps): # Calculates integrand value for integral corresponding to change in unitless potential
    return 2*pi*2**(1/2)*F0/V0*deps
    
def UI(V0prim): # upper integrand component for the change in electron density owing to creation of new electrons.
    # V0prim = V0[:len(x_k_i)] # all V where ionization occurs
    return V0prim*np.exp(-(V0prim/beta)**2)*x_thickness
    
def LI(V0): # lower integrand component for the change in electron density owing to creation of new electrons.
    return V0*x_k*x_thickness
    
def UperL(V0): # Integral fraction for calculating the change in electron density owing to creation of new electrons    
    if np.any(V0): # check if any V0 is nonzero
        V0prim = V0[:len(x_k_i)] # V0 in region where ionization occurs
        U = sum(UI(V0prim)) # compute upper integral
        L = sum(LI(V0)) # compute lower integral
        return U/L
    else: # both U and L will be 0 and division by zero occurs
        return 0
        
def Itilde(Vmat): # Integrand of full integral expression for calculating the change in electron density owing to creation of new electrons
    deps*???
    return 

#----------------------------------Ion motion----------------------------------

# 2.1 Function definitions
def ElectricField(philist): # calculates the unitless electric field
    return -(philist[1:]-philist[:-1])/(x_k[1:]-x_k[:-1]) # ElectricField list has one less element than phi. (Number of shells between points = Number of points - 1)

def arraytimeevaluator(v0, a, s, s_alt): # Calculates the crossing times for an array of initial velocities, accelerations and the two nearest shell boundary distances 
    t = np.empty(v0.shape)
    s_used = np.empty(s.shape)
    
    # t = v0/a*(-1+(1+2*a*s/v0**2)**(1/2))
    # s_used = s 
    
    taylor_ind = (abs(2*a*s)<abs(0.01*v0**2)).nonzero() # abs(2*a*s/v0**2)<0.01
    t[taylor_ind] = s[taylor_ind]/v0[taylor_ind]-a[taylor_ind]*s[taylor_ind]**2/(2*v0[taylor_ind]**3) # 2nd order Taylor expanded
    s_used[taylor_ind] = s[taylor_ind]
    
    real_ind = (2*a*s>=-v0**2).nonzero() # find all t that are calculated to be real
    t[real_ind] = v0[real_ind]/a[real_ind]*(-1+(1+2*a[real_ind]*s[real_ind]/v0[real_ind]**2)**(1/2))
    s_used[real_ind] = s[real_ind]
    
    # without the if block this causes a problem for some reason
    complex_ind = (2*a*s<-v0**2).nonzero() # find all t that are calculated to be complex
    if complex_ind[0].size>0:
        print('Complex_ind is: ', complex_ind)
        t[complex_ind], s_used[complex_ind] = arraytimeevaluator(v0[complex_ind], a[complex_ind], s_alt[complex_ind], s[complex_ind])
    
    negative_ind = (t<=0).nonzero()
    t[negative_ind] = -t[negative_ind]-2*v0[negative_ind]/a[negative_ind]

    return t, s_used

def ioncount(imatrix): # counts the number of ions inside each shell number k
    ks = imatrix[:,2]
    counts = np.empty((number_of_shells,), dtype=int)
    for k in range(number_of_shells):
        counts[k] = np.count_nonzero(ks==k)
    return counts
    
def ionmotion(imatrix, Delta_t, Elist): # calculates motion for every ion in ionmatrix. Each ion is a row in the n-by-3 ionmatrix [x_ion, u_ion, k_ion]
    pos = imatrix[:,0] # position of ions
    vs = imatrix[:,1] # velocity of ions
    ks = imatrix[:,2].astype(int) # shell number of ions, astype may be unnecessary
    
    counts = ioncount(imatrix) # count number of ions in each shell
    a = np.repeat(Elist, counts[:len(Elist)]) # calculate the electric field that each ion experiences (repeating the value of the Elist elements as many times as there are ions in each shell)
    
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
    
    # calculate the crossing times and what shell boundary was crossed
    crossing_t, s = arraytimeevaluator(vs, a, s_main, s_alt)
    
    # check for which indices a crossing happens/does not happen
    non_crossing_ind = (crossing_t>=Delta_t).nonzero()
    crossing_ind = (crossing_t<Delta_t).nonzero()
    
    # calculate new position and velocity for ions where crossing does not happen
    pos[non_crossing_ind] = pos[non_crossing_ind] + vs[non_crossing_ind]*Delta_t[non_crossing_ind] + a[non_crossing_ind]*Delta_t[non_crossing_ind]**2/2
    vs[non_crossing_ind] = vs[non_crossing_ind] + a[non_crossing_ind]*Delta_t[non_crossing_ind]
    
    # calculate new position and velocity and shell number for ions where crossing happens
    pos[crossing_ind] = np.sign(s[crossing_ind])*1e-6+pos[crossing_ind] + vs[crossing_ind]*crossing_t[crossing_ind] + a[crossing_ind]*crossing_t[crossing_ind]**2/2
    vs[crossing_ind] = vs[crossing_ind] + a[crossing_ind]*crossing_t[crossing_ind]
    ks[crossing_ind] = ks[crossing_ind]+np.sign(s[crossing_ind])
    
    # handle inner and outer BC
    IBC_ind = (ks==-1).nonzero()
    ks[IBC_ind] = 0 
    vs[IBC_ind] = -vs[IBC_ind] # bounce at comet surface
    
    OBC_ind = (ks>=number_of_shells).nonzero()
    ks[OBC_ind] = number_of_shells-1 # The outermost shell has infinite extent for now. Largest shell number allowed is number_of_boundaries-1

    # create a new matrix of ions
    imatrixnew = np.array(list(zip(pos, vs, ks))) # recombine position, velocity and shell number 
    
    # ions which crossed have to be calculated again recursively
    if crossing_ind[0].size>0:
        # print("Non crossing ind is: ", non_crossing_ind)
        # print("Crossing ind is: ", crossing_ind)
        # print("Remaining time - Crossing time is", Delta_t[crossing_ind]-crossing_t[crossing_ind])
        # print("imatrixnew[crossing_ind] = ", imatrixnew[crossing_ind])
        imatrixnew[crossing_ind] = ionmotion(imatrixnew[crossing_ind], Delta_t[crossing_ind]-crossing_t[crossing_ind], Elist)
    
    # use this line if you want to delete ions which reach outer boundary. This OBC_ind is now accounted for in two ways which should be inefficient, reconsider.
    imatrixnew = np.delete(imatrixnew, OBC_ind, axis=0)
    
    return imatrixnew 

# def iondensity(i_per_shell): Legacy and wrong
#     i_numberdensity = 1/(4*pi*(r_comet*x_k[1:])**2)*(i_per_shell[1:]+i_per_shell[:-1])/(2*x_comet*r_comet)
#     return i_numberdensity

def iondensity(i_per_shell): # calculates unitless density of simulated ions
    i_numberdensity = (i_per_shell[1:]+i_per_shell[:-1])/(4/3*pi*(x_k[2:]**3-x_k[:-2]**3))
    return i_numberdensity

# 2.4 Anders Loop. Supplementary, move to last. 
number_of_loops = int(len(x_k_i)/(u_n*Del_t))
counter = 0
andersfield = ElectricField(phi_anders_log)
for j in range(number_of_loops):
    remaining_time = np.repeat(Del_t, ionmatrix[:,0].shape) # create a remaining time matrix for prior ions
    source_ions = ioncreation(n_per_shell, int(x_k_i[-1])) # birth new ions
    source_remaining_time = np.random.uniform(0, Del_t, source_ions[:,0].shape) # add a uniform random time [0, Del_t) remaining for the newly born ions (Reflects the fact that the ions can be born any time during the time step)
    
    # concatenate prior ions and recently born ions
    ionmatrix = np.concatenate((ionmatrix, source_ions)) # add new ions to ion matrix
    remaining_time = np.concatenate((remaining_time, source_remaining_time))
    
    ionmatrix = ionmotion(ionmatrix, remaining_time, andersfield)
    ionmatrix = ionmatrix[ionmatrix[:,2].argsort()] # sort after column
    counter+=1 # increment the number of loops performed
    
    icount = ioncount(ionmatrix) # calculate number of ions in each shell
    
    idensity = iondensity(icount)
    plt.figure()
    plt.title('Number density of ions')
    plt.xlabel('Shell number '+r'$k$')
    plt.ylabel('Number density [m'+'$^{-3}$]')
    # andersprop = ((1+2*(x_k[1:-1]-1)*phi_at_comet)**(1/2)-1)/(x_k[1:-1]**2*phi_at_comet) # for linear potential
    andersprop = 1/x_k[1:-1]*(pi/2)**(1/2)*np.exp(1/(2*phi_at_comet))*(erf(((1+2*phi_at_comet*np.log(x_k[1:-1]))/(2*phi_at_comet))**(1/2)) - erf(1/(2*phi_at_comet)**(1/2)))# for logarithmic potential
    plt.plot(idensity/andersprop, '.', color='k')
    
    # plt.figure()
    # plt.plot(icount[:-1], '.', color='k')
    # plt.title('Number of ions inside each shell')
    # plt.xlabel('Shell number '+r'$k$')
    # plt.ylabel('Number of ions')
    
    # CALCULATE THIS TO CHECK DEGREE OF MONOMIAL RELATION BETWEEN COUNTS AND CENTRAL X_K OF THE SHELL
    # (np.log(icount[45])-np.log(icount[5]))/(np.log(x_k[45]+1/2)-np.log(x_k[5]+1/2)) # can skip or include