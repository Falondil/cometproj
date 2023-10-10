# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:45:49 2023

@author: Vviik
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import erf
from scipy import optimize

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

n_per_shell = 10 # number of ions generated in each shell
n_ion_sim = n_per_shell # count the number of simulated ions created per shell each timestep
n_ion_real = nu*(r_comet/v_n)**2*Del_t*Q # count the number of ions that they represent
realsimratio = n_ion_real/n_ion_sim # the number of real ions each simulated ion represents

#-----------------------s------------Creation-----------------------------------

# 1.1 General parameter choices
number_of_shells = int(1e2) # number of shells for the spatial discretization
number_of_boundaries = number_of_shells+1 # number of shell boundaries and edgepoints
x_k = np.arange(x_comet, number_of_boundaries+x_comet, dtype=int) # every equally thick shell has an equal number of ionization events per unit time. Unitless
x_k_i = x_k[:-int(len(x_k)/2)] # shells in which we consider ionization

# 1.2 defining functions
def logiondensity(x): # shell density of initial population of ions
    inv2phi = 1/(2*phi_at_comet)
    coef = np.sqrt(pi/2)*np.exp(inv2phi)*np.sqrt(phi_at_comet) # Anders rho is wrong. A factor sqrt(phi0) was missing.
    xerf = x*(erf(np.sqrt(inv2phi+np.log(x)))-erf(np.sqrt(inv2phi)))
    return coef*xerf

def logvelCDF(x, v): # velocity distribution for a known conditional comet distance x
    sqrtinv2phi = 1/np.sqrt(2*phi_at_comet)
    vmax = np.sqrt(1+2*phi_at_comet*np.log(x))
    numer = erf(v*sqrtinv2phi)-erf(sqrtinv2phi)
    denom = erf(vmax*sqrtinv2phi)-erf(sqrtinv2phi)
    return np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0)

def initialcreation(number_of_ions): # for creating the first population of ions (that is self-consistent to a log-potential)
    ionmatrix = np.empty((number_of_ions, 3)) # empty 0-by-3 matrix
    
    poslist = np.linspace(x_k[0], x_k[-1], number_of_boundaries*10) # create linearly spaced list for the discretization of the CDF integral into a cumulative sum
    posCDF = CDF(logiondensity, poslist) # find the CDF for ion comet distance
    print("CDFarray with size:", posCDF.size, posCDF)
    
    uniformlist = np.random.uniform(size=number_of_ions) # uniform distribution of ions in probability space
    print("uniformlist with size:", uniformlist.size, uniformlist)
    
    ionmatrix[:, 0] = np.interp(uniformlist, posCDF, poslist) # evaluate what comet distances correspond to the ions positions' in probability space
    ionmatrix = ionmatrix[ionmatrix[:,0].argsort()] # sort after comet distance
    
    ionmatrix[:, 2] = (ionmatrix[:,0]-1).astype(int) # shell number is one less than the comet distance rounded down. x=1.05 --> shell number=0
    
    for i in range(len(ionmatrix)):
        x = ionmatrix[i, 0]
        vmax = np.sqrt(1+2*phi_at_comet*np.log(x))
        vellist = np.linspace(1, vmax, number_of_boundaries) # calculate the CDF by approximating the integral via linearly spaced v. 
        velCDF = logvelCDF(x, vellist)
        ionmatrix[i, 1] = np.interp(np.random.uniform(), velCDF, vellist)

    # ionmatrix[:, 1] = 1+np.random.uniform(size=number_of_ions) # placeholder
    
    return ionmatrix

def CDF(func, arglist): # calculates CDF of a function at linearly spaced points (arglist)
    funclist = func(arglist) # numpy array of function evaluated at each arg in arglist
    csum = np.cumsum(funclist) # calculates an array consisting of the cumulative sum up to and including arg
    return csum/csum[-1]

def ioncreation(n_per_shell, xfin): # creates n ions, equally many in each shell, up to outer boundary xfin
    ilist = np.empty((0, 3)) # empty 0-by-3 matrix
    for k in range(xfin-1):
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
phi_at_comet = beta

phi_anders = phi_at_comet*(x_k[-1]-x_k)/x_k[-1] # Linear potential. numpy array, endpoint is 0

phi_anders_log = phi_at_comet*(np.log(x_k[-1])-np.log(x_k)) # phi=phi_at_comet at x=1 and phi=0 at x=x_k[-1]

# 1.4 Randomly generating the ions

ionmatrix = ioncreation(n_per_shell, x_k_i[-1]) # Position (sorted) and velocity of all ions.
if n_ion_sim > 1000:
    plt.figure()
    plt.hist(ionmatrix[:, 0], bins = 100, density = True)
    plt.hist(ionmatrix[:, 0], bins = 10, density = True, fill = False)
    plt.xlabel('Distance from comet nucleus [km]')
    plt.ylabel('Number of ions (pdf normalization)') # Normalization: area of all bins = 1
    plt.title('Radial distribution of randomly generated ions')
    
    
# 1.5 Electrons
excess = 10*beta # how many electrontemperatures we consider before truncation
eps, deps = np.linspace(-excess, excess, int(2*excess), retstep = True) # centered on 0.

def loginitialF(number_of_ions, eps, Vmat):
    inv2phi = 1/(2*phi_at_comet)
    # xterm = np.sqrt(eps+phi_at_comet*np.log(x_k[-1]/x_k))*(erf(np.sqrt(inv2phi+np.log(x_k)))-erf(np.sqrt(inv2phi)))
    xterm = np.matrix.transpose(Vmat)*(erf(np.sqrt(inv2phi+np.log(x_k)))-erf(np.sqrt(inv2phi)))
    xterm = np.matrix.transpose(xterm)
    cterm = 4*pi*np.sqrt(2*phi_at_comet)*np.exp(inv2phi)/(phi_at_comet**(3/2))*x_k[-1]*np.exp(eps/phi_at_comet)
    unscaledF = np.sum(xterm*cterm, axis=0) # compute integral, x axis = 0
    F = unscaledF*number_of_ions/electroncounter(unscaledF, Vmat) # rescale such that number_of_ions are created
    return F

def depscalc(epslist): 
    posbool = epslist>0 # boolean of positive energies
    poslist = epslist[posbool] # array of all positive energies
    neglist = epslist[np.invert(posbool)] # array of all negative energies
    
    depspos = poslist-np.append(0, poslist[:-1]) # add 0 to start of list and calculate distance to prior eps.
    depsneg = np.append(neglist[1:], 0)-neglist # add 0 to end of list and calculate distance to prior eps. 
    
    return np.concatenate((depsneg, depspos)) 

# Function definitions
def V(eps, phi): # Calculates unitless sqrt of electron kinetic energy. Either eps or phi can be numpy array.
    x = eps+phi    
    return ((x+abs(x))/2)**(1/2) # returns 0 if V is imaginary

def Vmatrix(epslist, philist): # Calculates matrix of values of V. epslist and philist are both 1D numpy arrays
    eps, phi = np.meshgrid(epslist, philist, sparse=True) # creates mesh of eps, phi values
    return V(eps, phi) # returns 2D array where Vmat[k,i] = V(philist[k], epslist[i])
    
def UI(V0prim): # Maxwell-Boltzmann integrand. upper integrand component for the change in electron density owing to creation of new electrons.
    # V0prim = V0[:len(x_k_i)] # all V where ionization occurs
    return V0prim*np.exp(-V0prim**2/beta)*x_thickness*np.heaviside(2*beta-V0prim, 0.5)
    
def LI(V0): # V0x^2 integrand. lower integrand component for the change in electron density owing to creation of new electrons.
    ret = np.matrix.transpose(V0)*x_k**2*x_thickness 
    return np.matrix.transpose(ret)

def UperL(Vmat): # Integral fraction for calculating the change in electron density owing to creation of new electrons    
    V0prim = Vmat[:len(x_k_i), :] # V0 in region where ionization occurs
    U = np.sum(UI(V0prim), axis=0) # compute upper integral, x axis = 0
    L = np.sum(LI(Vmat), axis=0) # compute lower integral, x axis = 0
    # Ifrac = np.zeros_like(U)
    # nonzero_ind = L.nonzero()
    # Ifrac[nonzero_ind] = U[nonzero_ind]/L[nonzero_ind]
    Ifrac = np.divide(U, L, out=np.zeros_like(U), where=L!=0) # division of U/L except 0 where U/L = 0/0
    return Ifrac

def delF(Vmat):
    # Alt. 1: Maxwell-Boltzmann
    ret = 1/(8*pi**2*(2*pi)**(1/2))*n_per_shell/beta**(3/2)*UperL(Vmat) # eq. 33 (label delF) in overleaf 
    
    # # Alt. 2: Uniform dist. of speeds
    # UI = 1/(2*beta)*np.heaviside(Vmat**2, 0)*np.heaviside(2*beta-Vmat**2, 1)
    # U = np.sum(UI, axis = 0) # computer integral over x axis
    # L = np.sum(LI(Vmat), axis=0)
    # ret = n_per_shell/(16*pi**2*2**(1/2))*np.divide(U, L, out=np.zeros_like(U), where=L!=0) # division of U/L except 0 where U/L = 0/0
    
    electronnumber = electroncounter(ret, Vmat) # count how many electrons are created.
    ret*=(n_per_shell*(x_k_i[-1]-x_k_i[0]))/electronnumber # how many should be created divided by how many are actually created
    return ret

def Fper2V(F0, Vmat): # Calculates integral corresponding to change in unitless potential. (Denominator of RHS in expression for del_phi)
    mat = np.divide(F0*deps, Vmat, out=np.zeros_like(Vmat), where=Vmat!=0) # matrix representation [eps, x] of integrand
    # print("mat", str(mat.shape))
    ret = np.sum(2*pi*2**(1/2)*mat, axis=1) # approximate integral via summation over eps axis = 1
    # mat[:,:int(len(eps)/2)]
    return ret

# Legacy
# def Itilde(Vmat): # Integral expression for calculating the change in electron density owing to creation of new electrons
#     Ifracdeps = UperL(Vmat)*deps 
#     # Ifracdeps = Ifracdeps[:int(len(eps)/2)] # only positive side of eps axis is summed over
#     return Vmat@Ifracdeps  # Vmat[:,:int(len(eps)/2)]

# def delphi(Vmat, F0, del_density): # eq. 46. calculates the change in unitless potential from one timestep to the next
#     numer = del_density-n_per_shell/(2*(pi*beta)**(3/2))*Itilde(Vmat) # numerator
#     denom = Fper2V(F0, Vmat) # denominator
#     ret = np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0) # numerator/denominator if denominator != 0 else 0.
#     return ret

def delphi(Vmat, F0, del_density): # eq. 45. calculates the change in unitless potential from one timestep to the next
    numer = del_density-4*pi*2**(1/2)*Vmat@(delF(Vmat)*deps) # numerator
    denom = Fper2V(F0, Vmat) # denominator
    ret = np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0) # numerator/denominator if denominator != 0 else 0.
    return ret

def dJtildedeps(Vmat): # eq. 30
    L = np.sum(LI(Vmat), axis=0) # compute lower integral, x axis = 0
    return 16*pi**2*2**(1/2)*L

def neweps(eps, phi, del_phi): # eq. 49
    Vmat = Vmatrix(eps, phi) # compute Vmat matrix
    Vmatx2 = LI(Vmat) # matrix with elements Vmat*x^2 
    delphiVmatx2 = np.matrix.transpose(np.matrix.transpose(Vmatx2)*del_phi) # multiply integrand by del_phi
    numer = np.sum(delphiVmatx2, axis=0) # computer upper integral, x axis = 0 
    denom = np.sum(Vmatx2, axis=0) # computer lower integral, x axis = 0
    return eps-np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0) #-W(Vmat, delxb)
    
def Fshift(F0, V0, V, del_phi): # change in F0 from motion of existing electrons (help function for eq. 54)
    numer = F0
    denom = depsdeps0(V0, del_phi)
    fraction = np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0) # numerator/denominator if denominator != 0 else 0

    V0int = np.sum(LI(V0), axis=0) # compute upper integral, summing over x axis
    Vint = np.sum(LI(V), axis=0) # compute lower integral, summing over x axis
    Ifrac = np.divide(V0int, Vint, out=np.zeros_like(V0int), where=Vint!=0)
    
    return fraction*Ifrac

def Fshift2(F0, V0, V, new_deps): # change in F0 from motion of existing electrons (help function for eq. 54)
    numer = F0*deps
    denom = new_deps
    fraction = np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0) # numerator/denominator if denominator != 0 else 0

    V0int = np.sum(LI(V0), axis=0) # compute upper integral, summing over x axis
    Vint = np.sum(LI(V), axis=0) # compute lower integral, summing over x axis
    Ifrac = np.divide(V0int, Vint, out=np.zeros_like(V0int), where=Vint!=0)
    
    return fraction*Ifrac

def newF(F0, V0, V, del_phi): # eq. 54
    Finput = F0 + delF(V0)/2 # add half of new electrons in old potential
    Foutput = Fshift(Finput, V0, V, del_phi) # compute the shift of the old distribution function
    
    return delF(V)/2 + Foutput # add half of new electrons in new potential

# Standalone without Fshift function usage
# def newF(F0, V0, V, del_phi): # eq. 54
#     numer = F0+delF(V0)/2 # add half of new electrons in old potential
#     denom = depsdeps0(V0, del_phi)
#     fraction = np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0) # numerator/denominator if denominator != 0 else 0.
    
#     V0int = np.sum(LI(V0), axis=0) # compute upper integral, summing over x axis
#     Vint = np.sum(LI(V), axis=0) # computer lower integral, summing over x axis
#     Ifrac = np.divide(V0int, Vint, out=np.zeros_like(V0int), where=Vint!=0)
    
#     return delF(V)/2 + fraction*Ifrac # add half of new electrons in new potential

def depsdeps0(Vmat, del_phi): # eq. 57
    deriv = ddelepsdeps0(Vmat, del_phi)
    return abs(1+deriv)

def ddelepsdeps0(Vmat, del_phi): # eq. 60
    Vmatx2 = LI(Vmat) # create matrix with elements Vmat*x^2
    sumVmatx2 = np.sum(Vmatx2, axis=0) # compute integral over x 
    
    delphiVmatx2 = np.matrix.transpose(np.matrix.transpose(Vmatx2)*del_phi) # multiply integrand by del_phi
    
    per2Vmat = np.divide(1, 2*Vmat, out=np.zeros_like(Vmat), where=Vmat!=0) # compute inverse of 2V0
    per2Vmatx2 = LI(per2Vmat) # compute matrix with element values x^2/(2Vmat)
    delphiper2Vmatx2 = np.matrix.transpose(np.matrix.transpose(per2Vmatx2)*del_phi) # multiply integrand by del_phi
    
    firstterm = np.sum(delphiVmatx2, axis=0)*np.sum(per2Vmatx2, axis=0)
    secondterm = np.sum(delphiper2Vmatx2, axis=0)*sumVmatx2
    numer = firstterm-secondterm
    denom = sumVmatx2**2
    return np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0) # - delW(Vmat, delxb)

# def W(Vmat, delxb): # calculates work done by the barrier. Takes Vmat of shape (len(x_e), len(eps)). xb is the position (right before) the barrier
#     xb = int(x_e[-1]) # last x before barrier
#     Vxb = Vmat[xb-1, :] # calculate V before the barrier
    
#     Vmatx2 = LI(Vmat) # create matrix with elements Vmat*x^2
#     sumVmatx2 = np.sum(Vmatx2, axis=0) # compute integral over x 
    
#     numer = 2*delxb*Vxb**3*xb**2
#     denom = 3*sumVmatx2
#     return np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0)

# def delW(Vmat, delxb): # Takes Vmat of shape (len(x_e), len(eps)). xb is the position (right before) the barrier
#     xb = int(x_e[-1]) # barrier
#     Vxb = Vmat[xb-1, :] # calculate V before the barrier, -2 because xb-1 is the index of xb. 
    
#     Vmatx2 = LI(Vmat) # create matrix with elements Vmat*x^2
#     sumVmatx2 = np.sum(Vmatx2, axis=0) # compute integral over x 
    
#     perVmat = np.divide(1, Vmat, out=np.zeros_like(Vmat), where=Vmat!=0) # compute inverse of V0
#     perVmatx2 = LI(perVmat) # compute matrix with element values x^2/Vmat
#     rightint = np.sum(perVmatx2, axis=0) # compute integral over x
    
#     numer = 2/3*delxb*xb**2*(3/2*Vxb*sumVmatx2-Vxb**3*rightint)
#     denom = sumVmatx2**2
#     return np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0)

def electroncounter(F, Vmat):
    Vmatx2 = LI(Vmat) # create matrix with elements Vmat*x^2
    sumVmatx2 = np.sum(Vmatx2, axis=0) # compute integral over x 
    return 16*pi**2*2**(1/2)*sum(F*deps*sumVmatx2)

def averageelectronenergy(F, Vmat):
    Vmatx2 = LI(Vmat) # create matrix with elements Vmat*x^2
    sumVmatx2 = np.sum(Vmatx2, axis=0) # compute integral over x 
    
    numer = (Vmat**2)@(F*deps*sumVmatx2)
    denom = sum((x_k[-1]-1)*F*deps*sumVmatx2)
    frac = np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0)
    return(sum(frac))

def electrondeleter(F, Vmat, ionsvanished):
    Vmatx2 = LI(Vmat) # create matrix with elements Vmat*x^2
    sumVmatx2 = np.sum(Vmatx2, axis=0) # compute integral over x 
    integrandarray = np.flip(16*pi**2*2**(1/2)*deps*F*sumVmatx2)
    
    integral = 0
    for ind in range(len(integrandarray)):
        integrand = integrandarray[ind]
        if integral + integrand >= ionsvanished:
            break
        integral += integrand
    remaining = ionsvanished - integral
    finalFdiff = remaining/(16*pi**2*2**(1/2)*(deps*sumVmatx2)[-ind-1])
    
    if ind > 0:
        F[-ind:] = 0
    F[-ind-1] = F[-ind-1]-finalFdiff
    return F

#----------------------------------Ion motion----------------------------------

# 2.1 Function definitions
def ElectricField(philist): # calculates the unitless electric field
    return -(philist[1:]-philist[:-1])/(x_k[1:]-x_k[:-1]) # ElectricField list has one less element than phi. (Number of shells between points = Number of points - 1)

def arraytimeevaluator(v0, a, s): # Calculates the crossing times for an array of initial velocities, accelerations and the signed shell boundary distance 
    t = np.empty(v0.shape)
    s_used = s # starting assumption
    
    # and overwrite those where the assumption is false
    complex_ind = (2*a*s<-v0**2).nonzero() # find all t that are calculated to be complex
    s_used[complex_ind] = s[complex_ind]-np.sign(s[complex_ind]) # s_inner = s_outer - 1;   s_outer = s_inner + 1 

    taylor_bool = abs(2*a*s_used)<=abs(0.01*v0**2)
    taylor_ind = (abs(2*a*s_used)<=abs(0.01*v0**2)).nonzero() # abs(2*a*s/v0**2)<0.01
    t[taylor_ind] = s_used[taylor_ind]/v0[taylor_ind]-a[taylor_ind]*s_used[taylor_ind]**2/(2*v0[taylor_ind]**3) # 2nd order Taylor expanded
    
    real_bool = 2*a*s_used>=-v0**2
    real_ind = (real_bool*np.invert(taylor_bool)).nonzero() # find all t calculated to be real NOT via Taylor equation
    # real_ind = (2*a*s>=-v0**2).nonzero() # find all t that are calculated to be real
    t[real_ind] = v0[real_ind]/a[real_ind]*(-1+(1+2*a[real_ind]*s[real_ind]/v0[real_ind]**2)**(1/2))

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
    
    # calculate the crossing times and what shell boundary was crossed
    crossing_t, s = arraytimeevaluator(vs, a, s_main)
    
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
    pos[IBC_ind] += 2e-6 # put on opposite side of the IBC boundary. Alternatively pos[IBC_ind] = x_thickness+1e-6
    ks[IBC_ind] = 0 
    vs[IBC_ind] = -vs[IBC_ind] # bounce at comet surface
    
    OBC_ind = (ks>=number_of_shells).nonzero() 
    ks[OBC_ind] = number_of_shells-1 # The outermost shell has infinite extent for now. Largest shell number allowed is number_of_shells-1

    # create a new matrix of ions
    imatrixnew = np.array(list(zip(pos, vs, ks))) # recombine position, velocity and shell number 
    
    # ions which crossed have time remaining. Their motion has to be calculated again recursively
    if crossing_ind[0].size>0:
        # print("Non crossing ind is: ", non_crossing_ind)
        # print("Crossing ind is: ", crossing_ind)
        # print("Remaining time - Crossing time is", Delta_t[crossing_ind]-crossing_t[crossing_ind])
        # print("imatrixnew[crossing_ind] = ", imatrixnew[crossing_ind])
        imatrixnew[crossing_ind] = ionmotion(imatrixnew[crossing_ind], Delta_t[crossing_ind]-crossing_t[crossing_ind], Elist)
    
    return imatrixnew 

def iondensity(i_per_shell): # calculates unitless density of simulated ions
    i_numberdensity = (i_per_shell[1:]+i_per_shell[:-1])/(4/3*pi*(x_k[2:]**3-x_k[:-2]**3))
    return i_numberdensity

def averageionenergy(imatrix):
    vs = imatrix[:,1] # velocity of ions
    return (sum(vs**2)/len(vs))

# 2.4 Loop
neutral_time = int(len(x_k_i)/(u_n*Del_t)) # time for the neutrals to travel x_k_i
number_of_loops = 25# neutral_time
simulation_time = r_comet/v_n*Del_t*number_of_loops # calculate how long a time (in seconds) that is simulated
nstart = 100000

start_time = time.time()
counter = 0
averageenergies = np.zeros((number_of_loops, 3))
ionnumbers = np.zeros(number_of_loops)
electronnumbers = np.zeros(number_of_loops)
totionsvanished = 0

# First timestep values

# zeros-method
# old_phi = np.zeros_like(x_k) # This causes F to only be non-zero for the highest eps (when considering constant phi)
# Vmat = Vmatrix(eps, old_phi)
# old_F = delF(Vmat) # starting F is assumed the delF that results from one burst of ionization
# old_density = np.zeros_like(x_k) # Perhaps change this for Peano eq. I.e. eta = 4*pi*sqrt(2)*integral(F*V deps)

# one step electron ionization-method # start with this guess
ionmatrix=initialcreation(nstart)

old_phi = phi_anders_log # PLACEHOLDER
Vmat = Vmatrix(eps, old_phi) # starting Vmatrix
old_F = loginitialF(nstart, eps, Vmat)
old_density = logiondensity(x_k) # initial ion shell density using a logarithmic initial potential
old_density *= nstart/np.sum(old_density*x_thickness)

# start from counter = 510 method
# counter = 510
# loaded_arrays = np.load('simto510.npz', allow_pickle=True) # ionmatrix=ionmatrix, old_density=old_density, old_phi=old_phi, old_F=old_F
# ionmatrix = loaded_arrays['ionmatrix']
# old_density = loaded_arrays['old_density']
# old_phi = loaded_arrays['old_phi']
# old_F = loaded_arrays['old_F']
# averageenergies[:510] = loaded_arrays['averageenergies'] # fill first 510 values of array with loaded values
# ionnumbers[:510] = loaded_arrays['ionnumbers']
# electronnumbers[:510] = loaded_arrays['electronnumbers']
# number_of_loops -= counter # perform 510 less timesteps
# number_of_loops = 10

for j in range(number_of_loops):
    # 1. Birth ions
    remaining_time = np.repeat(Del_t, ionmatrix[:,0].shape) # create a remaining time matrix for prior ions
    source_ions = ioncreation(n_per_shell, x_k_i[-1]) # birth new ions
    source_remaining_time = np.random.uniform(0, Del_t, source_ions[:,0].shape) # add a uniform random time [0, Del_t) remaining for the newly born ions (Reflects the fact that the ions can be born any time during the time step)
    
    # concatenate prior ions and recently born ions
    ionmatrix = np.concatenate((ionmatrix, source_ions)) # add new ions to ion matrix
    remaining_time = np.concatenate((remaining_time, source_remaining_time))
    
    # 2. Calculate the motion of ions
    Efield = ElectricField(old_phi) # calculate electric field from potential
    ionmatrix = ionmotion(ionmatrix, remaining_time, Efield)
    ionmatrix = ionmatrix[ionmatrix[:,2].argsort()] # sort after shell number
    OBC_ind = (ionmatrix[:,0]>x_k[-1]).nonzero() # find all ions passing outside the system
    ionmatrix = np.delete(ionmatrix, OBC_ind, axis=0) # and delete them
    ionsvanished = len(OBC_ind[0]) # count how many vanish
    totionsvanished += ionsvanished # and add to total 
    # also count how much kinetic energy vanishes
    furthestion = max(ionmatrix[:, 0]) # calculate how far the furthest ion has reached
    
    # 3. Calculate ion densities
    icount = ioncount(ionmatrix) # calculate number of ions in each shell
    idensity = iondensity(icount) # unitless iondensity (Does not contain inner or outer edgepoints)
    
    density_w_inner = np.concatenate((np.array([idensity[0]]), idensity)) # repeats the innermost calculated density value to approximate density at comet)
    new_density = np.append(density_w_inner, idensity[-1]) # repeats the outermost calculated density value to approximate outer boundary density
    
    del_density = (new_density - old_density) # change in density
    
    # 4. Use electrons to solve for change in potential
    Vmat = Vmatrix(eps, old_phi)
    new_F = old_F+delF(Vmat)
    new_F = electrondeleter(new_F, Vmat, ionsvanished) # Alt. 0. removes electrons from highest energy levels equal to number of ions removed using old potential
    new_F *= len(ionmatrix)/electroncounter(new_F, Vmat)
    
    neperni = electroncounter(new_F, Vmat)/len(ionmatrix)
    print('After adding delF and electrondeleter : ' + str(neperni))
    
    numiterations = 25
    for index in range(numiterations):
        del_phi = delphi(Vmat, new_F, del_density) # has same shape as x_k
        # if index<50:
        #     del_phi*=1/10
        # del_phi*=(index+1)/numiterations
        
        # 5. Calculate new potential
        new_phi = old_phi + del_phi
        
        # if index<90:
        #     phitoobigind = (new_phi>excess).nonzero()
        #     new_phi[phitoobigind] = excess
        #     del_phi[phitoobigind] = excess - old_phi[phitoobigind]
            
        #     phitoosmallind = (new_phi<-excess).nonzero()
        #     new_phi[phitoosmallind] = -excess
        #     del_phi[phitoosmallind] = -excess - old_phi[phitoosmallind]
        
        halfnew_Vmat = Vmatrix(eps, new_phi)
        
        # 6. calculate the new distribution function
        new_eps = neweps(eps, old_phi, del_phi)
        # new_Vmat = Vmatrix(new_eps, new_phi)
        # old_F = electrondeleter(old_F, new_Vmat, ionsvanished) # Alt. 1. removes electrons from highest energy levels equal to number of ions removed using new potential and energy levels
        # new_F = newF(new_F, Vmat, new_Vmat, del_phi)
        # new_F = Fshift(new_F, Vmat, new_Vmat, del_phi) 
        
        sortind = np.argsort(new_eps) # find ind that would sort new_eps
        # testeps = new_eps
        # print('Was it already sorted? ' + str(np.all(testeps==new_eps)))
        new_eps = new_eps[sortind] # sort new eps
        new_deps = depscalc(new_eps) # find a new deps
        new_Vmat = Vmatrix(new_eps, new_phi) # new sorting Vmat
        
        new_F = Fshift2(new_F[sortind], Vmat[:, sortind], new_Vmat, new_deps)
        new_F *= len(ionmatrix)/electroncounter(new_F, new_Vmat)
        
        # new_F = new_F[sortind] # sort new F via same sorting
        
        neperni = electroncounter(new_F, new_Vmat)/len(ionmatrix)
        print('After Fshift: ' + str(neperni))
        
        # now resample F
        electronsbeforeinterp = electroncounter(new_F, new_Vmat)
        beforestring = str(electronsbeforeinterp)[:int(np.log10(electronsbeforeinterp)+1)]
        
        # interpolation which preserves the number of electrons
        new_F = np.divide(np.interp(eps, new_eps, new_F*np.sum(LI(new_Vmat), axis=0)*new_deps), deps*np.sum(LI(Vmatrix(eps, new_phi)), axis=0), out=np.zeros_like(new_F), where=np.sum(LI(Vmatrix(eps, new_phi)), axis=0)!=0)     
        new_F *= len(ionmatrix)/electroncounter(new_F, halfnew_Vmat)
        
        electronsafterinterp = electroncounter(new_F, halfnew_Vmat)
        afterstring = str(electronsafterinterp)[:int(np.log10(electronsafterinterp)+1)]
        
        # Plot for each iteration (debugging)
        fig, (ax1, ax2) = plt.subplots(1, 2) 
        # fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2)
        fig.suptitle('Timestep: '+str(counter)+', Iteration: '+str(index))
        ax1.set_title('Potential')
        ax1.semilogy(x_k, new_phi*electrontemperature/beta,'.', color='k')
        ax1.semilogy(x_k, -new_phi*electrontemperature/beta,'.', color='r')
        ax1.axhline(electrontemperature, color='b') # line for electron temperature
        ax1.axvline(furthestion, color='k')
        ax1.set(xlabel = 'Distance from comet center [R'+'$_{C}$]', ylabel = 'Potential [V]')
        
        # change in potential debugging
        ax1.semilogy(x_k, del_phi*electrontemperature/beta,'x', color='k')
        ax1.semilogy(x_k, -del_phi*electrontemperature/beta,'x', color='r')
        
        # ax2.set_title('Electron distribution function')
        # ax2.semilogy(eps*electrontemperature/beta, new_F, '.', color='k')
        # ax2.semilogy(eps*electrontemperature/beta, -new_F, '.', color='r')
        # ax2.set(xlabel='Electron energy [eV]', ylabel= 'F')
        # ax2.yaxis.tick_right()
        
        rho = new_F*np.sum(LI(halfnew_Vmat), axis=0) # energy density
        
        ax2.set_title('Energy density')
        ax2.semilogy(eps*electrontemperature/beta, rho, '.', color='k')
        ax2.semilogy(eps*electrontemperature/beta, -rho, '.', color='r')
        ax2.set(xlabel='Electron energy [eV]', ylabel= '$rho$')
        ax2.yaxis.tick_right()
        
        ax2.text(0.01, 0.37, '#_e before', transform = ax2.transAxes)
        ax2.text(0.01, 0.31, 'interpolation:', transform = ax2.transAxes)
        ax2.text(0.01, 0.25, beforestring, transform = ax2.transAxes)
        
        ax2.text(0.01, 0.13, '#_e after', transform = ax2.transAxes)
        ax2.text(0.01, 0.07, 'interpolation:', transform = ax2.transAxes)
        ax2.text(0.01, 0.01, afterstring, transform = ax2.transAxes)
        
        # ax3.plot(x_k, new_phi,'.', color='k')
        # ax3.axvline(furthestion, color='k')
        # ax3.set(xlabel = 'Distance from comet center [R'+'$_{C}$]', ylabel = 'Potential')
        # ax4.plot(eps, new_F, '.', color='k')
        # ax4.set(xlabel='Electron energy', ylabel= 'F')
        # ax4.yaxis.tick_right()
        
    # new_F = new_F + delF(new_Vmat) # add the created electrons
    # new_F = electrondeleter(new_F, new_Vmat, ionsvanished) # Alt. 2. removes electrons from highest energy levels equal to number of ions removed using new potential and interpolated back to old energy levels
    
    # Calculate current total kinetic energy in the system
    avg_e_energy = averageelectronenergy(new_F, halfnew_Vmat)
    avg_i_energy = averageionenergy(ionmatrix)
    averageenergies[counter, :] = [avg_e_energy, avg_i_energy, (avg_e_energy+avg_i_energy)/2] # avg total energy can be calculated this way since we demand equal number of both electrons and ions
    
    ionnumbers[counter] = len(ionmatrix)
    electronnumbers[counter] = electroncounter(new_F, halfnew_Vmat)
    
    # Legacy
    # new_Vmat = Vmatrix(eps, new_phi)
    # del_F = delF(new_Vmat)
    # new_F = old_F + del_F


    # 7. Plotting
    # plt.figure()
    # plt.title('Number density of ions. Timestep: '+str(counter))
    # plt.xlabel('Distance from comet center [R'+'$_{C}$]')
    # plt.ylabel('Number density [R'+'$_{C}^{-3}$]')
    # plt.plot(x_k, new_density, '.', color='k')
    
    # plt.figure()
    # plt.title('Number of ions inside each shell. Timestep: '+str(counter))
    # plt.xlabel('Shell number')
    # plt.ylabel('Number of ions')
    # plt.plot(icount, '.', color='k')
    
    # plt.figure()
    # plt.title('Potential. Timestep: '+str(counter))
    # plt.xlabel('Distance from comet center [R'+'$_{C}$]')
    # plt.ylabel('Potential')
    # plt.plot(x_k, new_phi,'.', color='k')
    # plt.plot(x_k, phi_anders, '-', color = 'C0')
    
    # plt.figure()
    # plt.title('Change in potential. Timestep: '+str(counter))
    # plt.xlabel('Distance from comet center [R'+'$_{C}$]')
    # plt.ylabel('Change in potential')
    # plt.plot(x_k, del_phi,'.', color='k')
    
    # plt.figure()
    # plt.title('Electron distribution function. Timestep: '+str(counter))
    # plt.xlabel('Electron energy')
    # plt.ylabel('F')
    # plt.plot(new_eps, new_F, '.', color='k')
    
    # # Both potential and dist. function
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle('Timestep: '+str(counter))
    # ax1.set_title('Potential')
    # ax1.semilogy(x_k, new_phi,'.', color='k')
    # ax1.set(xlabel = 'Distance from comet center [R'+'$_{C}$]', ylabel = 'Potential')
    # ax2.set_title('Electron distribution function')
    # ax2.semilogy(eps, new_F, '.', color='k')
    # ax2.set(xlabel='Electron energy', ylabel= 'F')
    # ax2.yaxis.tick_right()
    
    # 8. Overwrite all old values
    old_density = new_density
    old_phi = new_phi
    old_F = new_F
    # eps = new_eps # This is wrong when resampling.
    
    counter+=1 # increment the number of loops performed
    # if counter > neutral_time: 
    #     if counter < neutral_time+10:   
    #         Del_t = 0.01
    #     else:
    #         Del_t = 0.1
    
    # if counter == 510:
    #     np.savez('simto'+str(counter)+'.npz', ionmatrix=ionmatrix, old_density=old_density, old_phi=old_phi, old_F=old_F, averageenergies=averageenergies, ionnumbers=ionnumbers, electronnumbers=electronnumbers)
    
    # np.mean(electronnumbers[300:-1]-electronnumbers[299:-2]) = -53 
    # ions increase by 520 each timestep before the knee, electrons increase by 466.8. 520-466.8 = 53.2
    
    
    # Relics
    # andersprop = ((1+2*(x_k[1:-1]-1)*phi_at_comet)**(1/2)-1)/(x_k[1:-1]**2*phi_at_comet) # for linear potential
    # # andersprop = 1/x_k[1:-1]*(pi/2)**(1/2)*np.exp(1/(2*phi_at_comet))*(erf(((1+2*phi_at_comet*np.log(x_k[1:-1]))/(2*phi_at_comet))**(1/2)) - erf(1/(2*phi_at_comet)**(1/2)))# for logarithmic potential
    # plt.plot(idensity/andersprop, '.', color='k')

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Timestep: '+str(counter)+', Iteration: '+str(index))
ax1.set_title('Potential')
ax1.plot(x_k, new_phi,'.', color='k')
ax1.axvline(furthestion, color='k')
ax1.set(xlabel = 'Distance from comet center [R'+'$_{C}$]', ylabel = 'Potential')
ax2.set_title('Electron distribution function')
ax2.plot(eps, new_F, '.', color='k')
ax2.set(xlabel='Electron energy', ylabel= 'F')
ax2.yaxis.tick_right()

plt.figure()
plt.title('Average energies in the system')
plt.xlabel('Timestep number')
plt.ylabel('Unitless average kinetic energy')
plt.plot(averageenergies[:,2], color='k', label='Total')
plt.plot(averageenergies[:,0], color='k', linestyle='--', label='Electron')
plt.plot(averageenergies[:,1], color='k', linestyle=':', label='Ion')
plt.axvline(10.07860256446754/Del_t, color='k')
plt.legend()

plt.figure()
plt.title('Number of particles')
plt.xlabel('Timestep number')
plt.ylabel('Number of particles')
plt.plot(electronnumbers, color='k', linestyle='--', label='Electrons')
plt.plot(ionnumbers, color='k', linestyle=':', label='Ions')
plt.axvline(10.07860256446754/Del_t, color='k')
plt.legend()

end_time = time.time()

elapsed_time = end_time-start_time
print('Simulated time: '+str(simulation_time)+' s')
print('Elapsed time: '+str(elapsed_time)+' s')