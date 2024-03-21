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
# from scipy.optimize import nnls
from scipy.linalg import solve_triangular
from scipy.interpolate import interp1d
    
# constants
pi = np.pi
r_comet = 1e3  # [m]
v_n = 6e2  # m/s, all ions are born with 600 m/s radial velocity
electrontemperature = 10 # eV
beta = 147.8215  # source electron temperature to source ion energy ratio: kT/(Mv^2)
# ionization frequency [number/s], Vigren2013a (Table 5. 9.2e-7 at solar maximum)
nu = 1e-6
Q = 1e25  # [s-1], number of neutrals per second leaving the comet surface
N_R = Q/(4*pi*r_comet**2*v_n)  # [m-3], neutral density at the comet surface
phi0 = beta

# normalization, tbd.
u_n = 1
x_comet = 1
x_thickness = 1
# timestep between ionization bursts, unitless. 1 is time for unaccelerated ions to traverse one shell width.
Del_t = 0.1

n_per_shell = max(int(Del_t*100), 1)  # number of ions generated in each shell
# count the number of simulated ions created per shell each timestep
n_ion_sim = n_per_shell
# count the number of ions that they represent
n_ion_real = nu*(r_comet/v_n)**2*Del_t*Q
# the number of real ions each simulated ion represents
realsimratio = n_ion_real/n_ion_sim

# -----------------------s------------Creation-----------------------------------

# 1.1 General parameter choices
number_of_shells = int(1e2)  # number of shells for the spatial discretization
# number of shell boundaries and edgepoints
number_of_boundaries = number_of_shells+1
# every equally thick shell has an equal number of ionization events per unit time. Unitless
x_k = np.arange(x_comet, number_of_boundaries+x_comet, dtype=int)
# spatial discretization for the electrons, can be more precise than for the ions
xe, dxe = np.linspace(x_comet, number_of_boundaries,
                      number_of_shells*10+1, retstep=True)
excess = 4*beta  # how many electrontemperatures we consider before truncation
eps, deps = np.linspace(-excess, excess, int(4*excess), retstep=True) # centered on 0.

# 1.2 defining functions

def depscalc(epslist):
    posbool = epslist > 0  # boolean of positive energies
    poslist = epslist[posbool]  # array of all positive energies
    neglist = epslist[np.invert(posbool)]  # array of all negative energies

    # add 0 to start of list and calculate distance to prior eps.
    depspos = poslist-np.append(0, poslist[:-1])
    # add 0 to end of list and calculate distance to prior eps.
    depsneg = np.append(neglist[1:], 0)-neglist

    return np.concatenate((depsneg, depspos))

# Linear potential. numpy array, endpoint is 0
def phi_anders_lin(xlist):
    return phi0*(number_of_boundaries-xlist)/number_of_boundaries # 0 when x equals 101 (number_of_boundaries)

def phi_anders_log(xlist): # logarithmic potential evaluated at the points xlist
    return phi0*(np.log(number_of_boundaries)-np.log(xlist)) # 0 when x equals 101 (number_of_boundaries)

# sigmoid parameters
L = 50
b = 10
c = 1+np.exp((1-L)/b)

def phi_anders_sigmoid(xlist): # sigmoid potential evaluated at the points xlist
    return phi0*(np.exp(L/b)+np.exp(1/b))/(np.exp(L/b)+np.exp(xlist/b))

def xenergyspacinglog(eps, tinyfactor): # returns the positions where the potential equals the negative of the total energy
    return number_of_boundaries*np.exp(eps/phi0)*np.exp(-deps*tinyfactor/phi0)

# epsminind = (eps>-phi_anders_log(x_comet)).nonzero()[0][0] # find minimal value for which an electron with that total energy can exist
# epsmaxind = (eps<0).nonzero()[0][-1] # set maximal energy to exclude all positive value for total electron energy
# goodeps = eps[epsminind:epsmaxind+1] # create new list of eps for use in creating a non-singular matrix of velocities
# goodeps, dgoodeps = np.linspace(eps[epsminind], eps[epsmaxind+1], 10*(epsmaxind-epsminind), retstep=True)

# xenergy = xenergyspacinglog(goodeps, 0.707)

def shelllogiondensity(x):  # shell density of initial population of ions with logarithmic potential
    inv2phi = 1/(2*phi0)
    coef = np.sqrt(pi*inv2phi)*np.exp(inv2phi)
    xerf = 4*pi*x*(erf(np.sqrt(inv2phi+np.log(x)))-erf(np.sqrt(inv2phi)))
    scaling = n_per_shell/(4*pi*Del_t)
    return scaling*coef*xerf

def shellsigmoidiondensity(x): # shell density of initial population of ions with sigmoid potential
    phi = phi_anders_sigmoid(x) # potential at the point x
    cterm = n_per_shell/Del_t*2*b # constant multiplying factor
    u1 = np.sqrt(1+2*phi0-2*phi) # maximal ion speed
    u2 = np.sqrt(1+2*c*phi0-2*phi) # same as u1 but with c term multiplying phi0
    u3 = np.sqrt(abs(1-2*phi)) # same as u2 but with c = 0, absolute value handles imaginary roots
    xterm = np.where(1-2*phi>0, np.arctanh(u1/u2)/u2-np.arctanh(u3/u1)/u3-np.arctanh(1/u2)/u2+np.arctanh(u3/1)/u3, np.arctanh(u1/u2)/u2-np.arctan(u3/u1)/u3-np.arctanh(1/u2)/u2+np.arctan(u3/1)/u3) # check if u3 is imaginary, use arctan over arctanh if it is.
    return cterm*xterm

# rectangular integration of shelliondensity between x = [1, xmax]
def initialionnumber(number_of_points, shelliondensityfunction):
    xlin, step = np.linspace(1, x_k[-1], number_of_points, retstep=True)
    return np.sum(shelliondensityfunction(xlin)*step)

def logvelCDF(x, v):  # velocity distribution for a known conditional comet distance x
    sqrtinv2phi = 1/np.sqrt(2*phi0)
    vmax = np.sqrt(1+2*phi0*np.log(x))
    numer = erf(v*sqrtinv2phi)-erf(sqrtinv2phi)
    denom = erf(vmax*sqrtinv2phi)-erf(sqrtinv2phi)
    return np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0)

# for creating the first population of ions (that is self-consistent to a log-potential)
def initialcreation(number_of_ions, shelliondensityfunction):
    ionmatrix = np.empty((number_of_ions, 3))  # empty 0-by-3 matrix

    # create linearly spaced list for the discretization of the CDF integral into a cumulative sum
    poslist = np.linspace(x_k[0], x_k[-1], number_of_boundaries*10)
    posCDF = CDF(shelliondensityfunction, poslist)  # find the CDF for ion comet distance
    # print("CDFarray with size:", posCDF.size, posCDF) # debugging

    # uniform distribution of ions in probability space
    uniformlist = np.random.uniform(size=number_of_ions)
    # print("uniformlist with size:", uniformlist.size, uniformlist) # debugging

    # evaluate what comet distances correspond to the ions positions' in probability space
    ionmatrix[:, 0] = np.interp(uniformlist, posCDF, poslist)
    # sort after comet distance
    ionmatrix = ionmatrix[ionmatrix[:, 0].argsort()]

    # shell number is one less than the comet distance rounded down. x=1.05 --> shell number=0
    ionmatrix[:, 2] = (ionmatrix[:, 0]-1).astype(int)

    for i in range(len(ionmatrix)):
        x = ionmatrix[i, 0]
        vmax = np.sqrt(1+2*phi0*np.log(x))
        # calculate the CDF by approximating the integral via linearly spaced v.
        vellist = np.linspace(1, vmax, number_of_boundaries)
        velCDF = logvelCDF(x, vellist)
        ionmatrix[i, 1] = np.interp(np.random.uniform(), velCDF, vellist)

    # ionmatrix[:, 1] = 1+np.random.uniform(size=number_of_ions) # placeholder

    return ionmatrix

def CDF(func, arglist):  # calculates normalized CDF of a function at linearly spaced points (arglist)
    cs = csum(func, arglist) 
    return cs/cs[-1]

def csum(func, arglist): # calculates the cumulative sum of a function at the points arglist
    funclist = func(arglist) # numpy array of function evaluated at each arg in arglist
    return np.cumsum(funclist) # calculates an array consisting of the cumulative sum up to and including arg

# creates n ions, equally many in each shell, up to outer boundary xfin
def ioncreation(n_per_shell, xfin):
    ilist = np.empty((0, 3))  # empty 0-by-3 matrix
    for k in range(xfin-1):
        ilist = np.concatenate(
            (ilist, ionshellcreation(n_per_shell, k)), axis=0)
    return ilist

def ionshellcreation(n, k):  # creates n ions uniformly distributed in the k-th shell
    ilist = np.empty((n, 3))  # creates an empty n-by-3 matrix
    ilist[:, 1] = u_n  # 1st column stores velocity
    ilist[:, 2] = k  # 2nd column stores shell number

    xmin = k+x_comet
    x_ion = xmin+np.array([np.random.rand(n)])
    x_ion.sort()
    x_ion.reshape((n, 1))  # make the x_ion array a column vector
    ilist[:, 0] = x_ion  # 0th column stores position
    return ilist

# 1.3 Electrons

# 1.3.1 Initialization of electrons

def loginitialrho(eps, Vmat):
    inv2phi = 1/(2*phi0)
    xterm = xe**2 * \
        np.matrix.transpose(
            Vmat)*(erf(np.sqrt(inv2phi+np.log(xe)))-erf(np.sqrt(inv2phi)))*dxe
    xterm = np.matrix.transpose(xterm)
    xterm[0] /= 2
    xterm[-1] /= 2
    cterm = 2**(1/2)/(phi0**2)*n_per_shell/Del_t * \
        np.exp(inv2phi)*np.exp(-eps/phi0)*1/xe[-1]
    ret = np.sum(xterm*cterm, axis=0)  # compute integral, x axis = 0
    return ret

def maxwell(energy, temp): # computes the maxwell boltzman distribution in energy space
    cterm = 2/(np.sqrt(pi)*beta**(3/2))
    xterm = np.sqrt(abs(energy))*np.exp(-abs(energy)/temp)
    return cterm*xterm

def invsqrtdist(energy,minenergy): # computes a square root distribution in energy space
    return 1/(2*np.sqrt(energy*minenergy))

def expinvsqrtdist(energy, temp): # computes an exponential divided by square root distribution in energy space
    return np.exp(energy/temp)*np.sqrt(-pi*temp*energy)

def doublezerodist(energy, fi0): # computes a ...
    return (fi0+energy)**2*np.exp(-(energy)/(1/10*fi0))

def integrandsigmoidandersrho(xlist):
    phi = phi_anders_sigmoid(xlist)
    u1 = np.sqrt(1+2*phi0-2*phi) # maximal ion speed
    u2 = np.sqrt(1+2*c*phi0-2*phi) # same as u1 but with c term multiplying phi0
    u3 = np.sqrt(abs(1-2*phi)) # same as u2 but with c = 0, absolute value handles imaginary roots
    integrand = 1/phi*np.where(1-2*phi>0, np.arctanh(u1/u2)/u2-np.arctanh(u3/u1)/u3-np.arctanh(1/u2)/u2+np.arctanh(u3/1)/u3, np.arctanh(u1/u2)/u2-np.arctan(u3/u1)/u3-np.arctanh(1/u2)/u2+np.arctan(u3/1)/u3) # check if u3 is imaginary, use arctan over arctanh if it is.
    integrand[0] /= 2 # division such that summing the elements is equivalent with trapezoid summation approximation
    integrand[-1] /= 2 # same as above
    cterm = 2*b*4*pi*n_per_shell/Del_t # constant term in front of integral including scaling
    return integrand*cterm

# 1.3.2 General electron functions

# Calculates unitless sqrt of electron kinetic energy. Either eps or phi can be numpy array.
def V(eps, phi):
    x = eps+phi
    return ((x+abs(x))/2)**(1/2)  # returns 0 if V is imaginary


# Calculates matrix of values of V. epslist and philist are both 1D numpy arrays
def Vmatrix(epslist, philist):
    # creates mesh of eps, phi values
    eps, phi = np.meshgrid(epslist, philist, sparse=True)
    # returns 2D array where Vmat[k,i] = V(philist[k], epslist[i])
    return V(eps, phi)


# Maxwell-Boltzmann integrand. upper integrand component for the change in electron density owing to creation of new electrons.
def UI(V0):
    return V0*np.exp(-V0**2/beta)*dxe*np.heaviside(2*beta-V0, 0.5)


def LI(V0):  # V0x^2 integrand. lower integrand component for the change in electron density owing to creation of new electrons.
    ret = np.matrix.transpose(V0)*xe**2*dxe
    return np.matrix.transpose(ret)


def UperL(Vmat):  # Integral fraction for calculating the change in electron density owing to creation of new electrons
    UpperIntegrand = UI(Vmat)
    LowerIntegrand = LI(Vmat)
    UpperIntegrand[0] /= 2
    UpperIntegrand[-1] /= 2
    LowerIntegrand[0] /= 2
    LowerIntegrand[-1] /= 2
    U = np.sum(UpperIntegrand, axis=0)  # compute upper integral, x axis = 0
    L = np.sum(LowerIntegrand, axis=0)  # compute lower integral, x axis = 0
    # Ifrac = np.zeros_like(U)
    # nonzero_ind = L.nonzero()
    # Ifrac[nonzero_ind] = U[nonzero_ind]/L[nonzero_ind]
    # division of U/L except 0 where U/L = 0/0
    Ifrac = np.divide(U, L, out=np.zeros_like(U), where=L != 0)
    return Ifrac


def delF(Vmat):
    # Alt. 1: Maxwell-Boltzmann
    ret = 1/(8*pi**2*(2*pi)**(1/2))*n_per_shell/beta**(3/2) * \
        UperL(Vmat)  # eq. 33 (label delF) in overleaf

    # # Alt. 2: Uniform dist. of speeds
    # UI = 1/(2*beta)*np.heaviside(Vmat**2, 0)*np.heaviside(2*beta-Vmat**2, 1)
    # U = np.sum(UI, axis = 0) # computer integral over x axis
    # L = np.sum(LI(Vmat), axis=0)
    # ret = n_per_shell/(16*pi**2*2**(1/2))*np.divide(U, L, out=np.zeros_like(U), where=L!=0) # division of U/L except 0 where U/L = 0/0

    # count how many electrons are created.
    electronnumber = electroncounter(ret, Vmat)
    # how many should be created divided by how many are actually created
    ret *= (n_per_shell*(xe[-1]-xe[0]))/electronnumber
    return ret


def Fper2V(F0, Vmat):  # Calculates integral corresponding to change in unitless potential. (Denominator of RHS in expression for del_phi)
    # matrix representation [eps, x] of integrand
    mat = np.divide(F0*deps, Vmat, out=np.zeros_like(Vmat), where=Vmat != 0)
    # print("mat", str(mat.shape))
    # approximate integral via summation over eps axis = 1
    ret = np.sum(2*pi*2**(1/2)*mat, axis=1)
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

# def delphi(Vmat, F0, del_density): # eq. 45. calculates the change in unitless potential from one timestep to the next
#     numer = del_density-4*pi*2**(1/2)*Vmat@(delF(Vmat)*deps) # numerator
#     denom = Fper2V(F0, Vmat) # denominator
#     ret = np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0) # numerator/denominator if denominator != 0 else 0.
#     return ret

# def delphi(Vmat, ionVmat, F0, del_density): # eq. 45. calculates the change in unitless potential from one timestep to the next
#     numer = del_density-4*pi*2**(1/2)*ionVmat@(delF(Vmat)*deps) # numerator
#     denom = Fper2V(F0, ionVmat) # denominator
#     ret = np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0) # numerator/denominator if denominator != 0 else 0.
#     return ret

def etae(F, V0, del_phi):
    coef = 4*pi*2**(1/2)
    transmat = np.matrix.transpose(V0**2) # transpose matrix with elements V0^2 such that del_phi can be added
    transmat += del_phi # add del_phi
    Vsquared = np.matrix.transpose(transmat) # transpose back to original orientation
    Vsquared = (Vsquared + abs(Vsquared))/2 # all negative values become 0
    integrand = F*Vsquared**(1/2)*deps
    integrand[0] /= 2
    integrand[-1] /= 2
    ret = coef*np.sum(integrand, axis=1) # approximate integral via summation over eps axis = 1
    return ret

def delphitaylor(ionVmat, F, density):
    numer = density-4*pi*2**(1/2)*ionVmat@(F*deps)
    denom = Fper2V(F, ionVmat)
    # numerator/denominator if denominator != 0 else 0.
    ret = np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0)
    return ret

def delphilimit(ionVmat, F, iondensity):
    coef = 4*pi*2**(1/2) 
    electrondensity = coef*np.sum(F*ionVmat*deps, axis=1) # calculate the electron density if there is no change in potential
    numer = iondensity - electrondensity # calculate difference in densities
    denom = coef*sum(F*deps) # approximate integral via summation
    sqrtlimit = np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0) # find the limit for the square root of the change in potential
    return sqrtlimit**2 # return the limit for the change in potential

def densitydifferencealt(del_phi, ionVmat, F, iondensity): # calculate the difference in densities as a function of the guess for del_phi
    return etae(F, ionVmat, del_phi) - iondensity

# def delphitrial(ionVmat, F, iondensity, limit):
#     def densitydifference(delp): # calculate the difference in densities as a function of the guess for del_phi
#         return etae(F, ionVmat, delp) - iondensity
#     sol = optimize.bisect(densitydifference, -limit, limit)#, args=(ionVmat, F, iondensity))
#     return sol

def etaesingle(F, V0, del_phi): # calculates the electron density at a single position
    coef = 4*pi*2**(1/2)
    Vsquared = V0**2+del_phi
    Vsquared = (Vsquared + abs(Vsquared))/2 # all negative values become 0
    integrand = F*Vsquared**(1/2)*deps
    integrand[0] /= 2
    integrand[-1] /= 2
    ret = coef*np.sum(integrand) # approximate integral via summation
    return ret

# def delphitrial2(ionVmat, F, iondensity):
#     def densitydifference(delp): # calculate the difference in densities as a function of the guess for del_phi
#         return etae(F, ionVmat, delp) - iondensity
#     sol = optimize.fsolve(densitydifference, 0)
#     return sol

def G0func(eps, F, depsilon=deps): # creates the function G(eps)
    return interp1d(eps, 4*pi*2**(1/2)*F*depsilon, bounds_error=False, fill_value=0)
    
def delphisingle(V0, F, iondensity):
    def densitydifferencesingle(delp):
        return etaesingle(F, V0, delp) - iondensity
    sol = optimize.newton_krylov(densitydifferencesingle, 0)
    return sol

def delphimulti(eps, F0, V0, density): # caclculates the change in potential using F(eps0) = F0(eps0) approximation
    def Z(delp): # create subfunction for the function that we want root solution to
        Vsquared = np.matrix.transpose(np.matrix.transpose(V0**2)+delp)
        Vsquared = (Vsquared + abs(Vsquared))/2 # all negative values become 0
        V = Vsquared**(1/2)
        return 4*pi*2**(1/2)*np.sum(F0*V, axis=1)*deps-density
    
    def dZddelphi(delp): # create subfunction for the derivative of the function that we want root solution to
        Vsquared = np.matrix.transpose(np.matrix.transpose(V0**2)+delp)
        Vsquared = (Vsquared + abs(Vsquared))/2 # all negative values become 0
        invV = np.divide(1, Vsquared**(1/2), out=np.zeros_like(Vsquared), where=Vsquared!=0)
        return pi*2**(3/2)*np.sum(F0*invV, axis=1)*deps
    
    sol = optimize.newton(Z, np.zeros_like(density), fprime=dZddelphi)
    return sol

def delphiG0(eps, G0, V0, density): # calculates the change in potential using G0 as an interpolated function
    def subfunc(delp):
        Vsquared = np.matrix.transpose(np.matrix.transpose(V0**2)+delp)
        Vsquared = (Vsquared + abs(Vsquared))/2 # all negative values become 0
        V = Vsquared**(1/2)
        return V@G0(eps+np.divide(np.sum(np.matrix.transpose(delp*np.matrix.transpose(V)), axis=0), np.sum(V, axis=0), out=np.zeros_like(eps), where=np.sum(V, axis=0)!=0))-density
    sol = optimize.newton_krylov(subfunc, np.zeros_like(density)) # find which change in potential delp solves the subfunction
    return sol
    
def Jtilde(Vmat):  # unitless ergodic invariant
    # 16*pi**2*2**(3/2)/3 # does not matter if coef is accurate since they are just compared with eachother
    coef = 1
    # transpose Vmat matrix to be order to element-wise multiply by the factor x^3/2 before returning to original untransposed format.
    integrand = np.matrix.transpose((np.matrix.transpose(Vmat))**3*xe**2*dxe)
    integrand[0] /= 2
    integrand[-1] /= 2
    return coef*np.sum(integrand, axis=0)

def dJtildedeps(Vmat):  # eq. 30
    Vmatx2 = LI(Vmat)
    Vmatx2[0] /= 2
    Vmatx2[-1] /= 2
    L = np.sum(Vmatx2, axis=0)  # compute lower integral, x axis = 0
    return 16*pi**2*2**(1/2)*L

# Legacy
# def neweps(eps, phi, del_phi): # eq. 49
#     Vmat = Vmatrix(eps, phi) # compute Vmat matrix
#     Vmatx2 = LI(Vmat) # matrix with elements Vmat*x^2*dx
#     delphiVmatx2 = np.matrix.transpose(np.matrix.transpose(Vmatx2)*del_phi) # multiply integrand by del_phi
#     numer = np.sum(delphiVmatx2, axis=0) # computer upper integral, x axis = 0
#     denom = np.sum(Vmatx2, axis=0) # computer lower integral, x axis = 0
#     return eps-np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0) #-W(Vmat, delxb)

# Legacy
# def Fshift(F0, V0, V, del_phi): # change in F0 from motion of existing electrons (help function for eq. 54)
#     numer = F0
#     denom = depsdeps0(V0, del_phi)
#     fraction = np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0) # numerator/denominator if denominator != 0 else 0

#     V0int = np.sum(LI(V0), axis=0) # compute upper integral, summing over x axis
#     Vint = np.sum(LI(V), axis=0) # compute lower integral, summing over x axis
#     Ifrac = np.divide(V0int, Vint, out=np.zeros_like(V0int), where=Vint!=0)

#     return fraction*Ifrac

# Legacy
# def Fshift2(F0, V0, V, new_deps): # change in F0 from motion of existing electrons (help function for eq. 54)
#     numer = F0*deps
#     denom = new_deps
#     fraction = np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0) # numerator/denominator if denominator != 0 else 0

#     V0int = np.sum(LI(V0), axis=0) # compute upper integral, summing over x axis
#     Vint = np.sum(LI(V), axis=0) # compute lower integral, summing over x axis
#     Ifrac = np.divide(V0int, Vint, out=np.zeros_like(V0int), where=Vint!=0)

#     return fraction*Ifrac

# Legacy
# def newF(F0, V0, V, del_phi): # eq. 54
#     Finput = F0 + delF(V0)/2 # add half of new electrons in old potential
#     Foutput = Fshift(Finput, V0, V, del_phi) # compute the shift of the old distribution function

#     return delF(V)/2 + Foutput # add half of new electrons in new potential

# Standalone without Fshift function usage
# def newF(F0, V0, V, del_phi): # eq. 54
#     numer = F0+delF(V0)/2 # add half of new electrons in old potential
#     denom = depsdeps0(V0, del_phi)
#     fraction = np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0) # numerator/denominator if denominator != 0 else 0.

#     V0int = np.sum(LI(V0), axis=0) # compute upper integral, summing over x axis
#     Vint = np.sum(LI(V), axis=0) # computer lower integral, summing over x axis
#     Ifrac = np.divide(V0int, Vint, out=np.zeros_like(V0int), where=Vint!=0)

#     return delF(V)/2 + fraction*Ifrac # add half of new electrons in new potential


def epschange(del_phi, Vmat):
    Vmatx2delphi = np.matrix.transpose(np.matrix.transpose(
        Vmat)*del_phi*xe**2*dxe)  # create matrix with elements Vmat*x^2*dx
    Vmatx2delphi[0] /= 2
    Vmatx2delphi[-1] /= 2
    numer = -np.sum(Vmatx2delphi, axis=0)  # compute integral over x
    Vmatx2 = LI(Vmat)
    Vmatx2[0] /= 2
    Vmatx2[-1] /= 2
    denom = np.sum(Vmatx2, axis=0)  # compute integral over x
    ret = np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0)
    return ret

# Legacy
# def depsdeps0(Vmat, del_phi): # eq. 57
#     deriv = ddelepsdeps0(Vmat, del_phi)
#     return abs(1+deriv)

# def ddelepsdeps0(Vmat, del_phi): # eq. 61
#     Vmatx2 = LI(Vmat) # create matrix with elements Vmat*x^2*dx
#     sumVmatx2 = np.sum(Vmatx2, axis=0) # compute integral over x

#     delphiVmatx2 = np.matrix.transpose(np.matrix.transpose(Vmatx2)*del_phi) # multiply integrand by del_phi

#     per2Vmat = np.divide(1, 2*Vmat, out=np.zeros_like(Vmat), where=Vmat!=0) # compute inverse of 2V0
#     per2Vmatx2 = LI(per2Vmat) # compute matrix with element values x^2/(2Vmat)*dx
#     delphiper2Vmatx2 = np.matrix.transpose(np.matrix.transpose(per2Vmatx2)*del_phi) # multiply integrand by del_phi

#     firstterm = np.sum(delphiVmatx2, axis=0)*np.sum(per2Vmatx2, axis=0)
#     secondterm = np.sum(delphiper2Vmatx2, axis=0)*sumVmatx2
#     numer = firstterm-secondterm
#     denom = sumVmatx2**2
#     return np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0)


def electroncounter(F, Vmat, depsilon=deps):
    Vmatx2 = LI(Vmat)  # create matrix with elements Vmat*x^2*dx
    sumVmatx2 = np.sum(Vmatx2, axis=0)  # compute integral over x
    integrand = F*depsilon*sumVmatx2
    # halve the integrand at the end points to make summation approximate trapezoid method
    integrand[0] /= 2
    integrand[-1] /= 2
    return 16*pi**2*2**(1/2)*sum(integrand)

# def averageelectronenergy(F, Vmat):
#     Vmatx2 = LI(Vmat) # create matrix with elements Vmat*x^2*dx
#     sumVmatx2 = np.sum(Vmatx2, axis=0) # compute integral over x

#     numer = (Vmat**2)@(F*deps*sumVmatx2)
#     denom = sum((xe[-1]-1)*F*deps*sumVmatx2)
#     frac = np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0)
#     return(sum(frac*dxe))


def electronenergy(F, Vmat):
    # matrix with integrand V^3 x^2 dxe as elements.
    Vmat3x2 = np.matrix.transpose((np.matrix.transpose(Vmat))**3*xe**2*dxe)
    sumFVmat3x2 = np.sum(F*Vmat3x2*deps, axis=1)  # compute integral over eps
    sumFVmat3x2[0] /= 2
    sumFVmat3x2[-1] /= 2
    sumsumFVmat3x2 = np.sum(sumFVmat3x2, axis=0)  # compute integral over x
    coef = 16*pi**2*2**(1/2)  # coefficient for the integral
    ret = coef*sumsumFVmat3x2  # calculate the total kinetic energy
    return ret


def electrondeleter(F, Vmat, ionsvanished, depsilon=deps):
    Vmatx2 = LI(Vmat)  # create matrix with elements Vmat*x^2*dx
    sumVmatx2 = np.sum(Vmatx2, axis=0)  # compute integral over x
    integrandarray = np.flip(16*pi**2*2**(1/2)*deps*F*sumVmatx2)

    integral = 0
    for ind in range(len(integrandarray)):
        integrand = integrandarray[ind]
        if integral + integrand >= ionsvanished:
            break
        integral += integrand
    remaining = ionsvanished - integral
    finalFdiff = remaining/(16*pi**2*2**(1/2)*(depsilon*sumVmatx2)[-ind-1])

    if ind > 0:
        F[-ind:] = 0
    F[-ind-1] = F[-ind-1]-finalFdiff
    return F

# ----------------------------------Ion motion----------------------------------

# 2.1 Function definitions


def ElectricField(philist):  # calculates the unitless electric field
    # ElectricField list has one less element than phi. (Number of shells between points = Number of points - 1)
    return -(philist[1:]-philist[:-1])/(x_k[1:]-x_k[:-1])


# Calculates the crossing times for an array of initial velocities, accelerations and the signed shell boundary distance
def arraytimeevaluator(v0, a, s):
    t = np.empty(v0.shape)
    s_used = s  # starting assumption

    # and overwrite those where the assumption is false
    # find all t that are calculated to be complex
    complex_ind = (2*a*s < -v0**2).nonzero()
    # s_inner = s_outer - 1;   s_outer = s_inner + 1
    s_used[complex_ind] = s[complex_ind]-np.sign(s[complex_ind])

    taylor_bool = abs(2*a*s_used) <= abs(0.01*v0**2)
    taylor_ind = (abs(2*a*s_used) <= abs(0.01*v0**2)
                  ).nonzero()  # abs(2*a*s/v0**2)<0.01
    t[taylor_ind] = s_used[taylor_ind]/v0[taylor_ind]-a[taylor_ind] * \
        s_used[taylor_ind]**2 / \
        (2*v0[taylor_ind]**3)  # 2nd order Taylor expanded

    real_bool = 2*a*s_used >= -v0**2
    # find all t calculated to be real NOT via Taylor equation
    real_ind = (real_bool*np.invert(taylor_bool)).nonzero()
    # real_ind = (2*a*s>=-v0**2).nonzero() # find all t that are calculated to be real
    t[real_ind] = v0[real_ind]/a[real_ind] * \
        (-1+(1+2*a[real_ind]*s[real_ind]/v0[real_ind]**2)**(1/2))

    negative_ind = (t <= 0).nonzero()
    t[negative_ind] = -t[negative_ind]-2*v0[negative_ind]/a[negative_ind]

    return t, s_used


def ioncount(imatrix):  # counts the number of ions inside each shell number k
    ks = imatrix[:, 2]
    counts = np.empty((number_of_shells,), dtype=int)
    for k in range(number_of_shells):
        counts[k] = np.count_nonzero(ks == k)
    return counts


# calculates motion for every ion in ionmatrix. Each ion is a row in the n-by-3 ionmatrix [x_ion, u_ion, k_ion]
def ionmotion(imatrix, Delta_t, Elist):
    pos = imatrix[:, 0]  # position of ions
    vs = imatrix[:, 1]  # velocity of ions
    # shell number of ions, astype may be unnecessary
    ks = imatrix[:, 2].astype(int)

    counts = ioncount(imatrix)  # count number of ions in each shell
    # calculate the electric field that each ion experiences (repeating the value of the Elist elements as many times as there are ions in each shell)
    a = np.repeat(Elist, counts[:len(Elist)])

    pos_in_shell = pos % 1  # calculates position inside each shell from [0, 1)
    s_inner = -pos_in_shell  # distance (negative) to inner shell for each ion
    s_outer = 1-pos_in_shell  # distance (positive) to outer shell for each ion

    # s_main: New array where s has the same direction as the velocity
    positive_vs_ind = (vs > 0).nonzero()
    s_main = s_inner  # (Read line below first) rest go to inner s
    # those with positive velocity go to outer s
    s_main[positive_vs_ind] = s_outer[positive_vs_ind]

    # calculate the crossing times and what shell boundary was crossed
    crossing_t, s = arraytimeevaluator(vs, a, s_main)

    # check for which indices a crossing happens/does not happen
    non_crossing_ind = (crossing_t >= Delta_t).nonzero()
    crossing_ind = (crossing_t < Delta_t).nonzero()

    # calculate new position and velocity for ions where crossing does not happen
    pos[non_crossing_ind] = pos[non_crossing_ind] + vs[non_crossing_ind] * \
        Delta_t[non_crossing_ind] + a[non_crossing_ind] * \
        Delta_t[non_crossing_ind]**2/2
    vs[non_crossing_ind] = vs[non_crossing_ind] + \
        a[non_crossing_ind]*Delta_t[non_crossing_ind]

    # calculate new position and velocity and shell number for ions where crossing happens
    pos[crossing_ind] = np.sign(s[crossing_ind])*1e-6+pos[crossing_ind] + vs[crossing_ind] * \
        crossing_t[crossing_ind] + a[crossing_ind] * \
        crossing_t[crossing_ind]**2/2
    vs[crossing_ind] = vs[crossing_ind] + \
        a[crossing_ind]*crossing_t[crossing_ind]
    ks[crossing_ind] = ks[crossing_ind]+np.sign(s[crossing_ind])

    # handle inner and outer BC
    IBC_ind = (ks == -1).nonzero()
    # put on opposite side of the IBC boundary. Alternatively pos[IBC_ind] = x_thickness+1e-6
    pos[IBC_ind] += 2e-6
    ks[IBC_ind] = 0
    vs[IBC_ind] = -vs[IBC_ind]  # bounce at comet surface

    OBC_ind = (ks >= number_of_shells).nonzero()
    # The outermost shell has infinite extent for now. Largest shell number allowed is number_of_shells-1
    ks[OBC_ind] = number_of_shells-1

    # create a new matrix of ions
    # recombine position, velocity and shell number
    imatrixnew = np.array(list(zip(pos, vs, ks)))

    # ions which crossed have time remaining. Their motion has to be calculated again recursively
    if crossing_ind[0].size > 0:
        # print("Non crossing ind is: ", non_crossing_ind)
        # print("Crossing ind is: ", crossing_ind)
        # print("Remaining time - Crossing time is", Delta_t[crossing_ind]-crossing_t[crossing_ind])
        # print("imatrixnew[crossing_ind] = ", imatrixnew[crossing_ind])
        imatrixnew[crossing_ind] = ionmotion(
            imatrixnew[crossing_ind], Delta_t[crossing_ind]-crossing_t[crossing_ind], Elist)

    return imatrixnew


def iondensity(i_per_shell):  # calculates unitless density of simulated ions
    i_numberdensity = (i_per_shell[1:]+i_per_shell[:-1]) / \
        (4/3*pi*(x_k[2:]**3-x_k[:-2]**3))
    return i_numberdensity


def averageionenergy(imatrix):
    vs = imatrix[:, 1]  # velocity of ions
    return (sum(vs**2)/(2*len(vs)))


# 2.4 Loop
neutral_time = int(len(x_k)/(u_n*Del_t))  # time for the neutrals to travel x_k
number_of_loops = 100  # neutral_time
# calculate how long a time (in seconds) that is simulated
simulation_time = r_comet/v_n*Del_t*number_of_loops

start_time = time.time()
counter = 0
averageenergies = np.zeros((number_of_loops, 6))
ionnumbers = np.zeros(number_of_loops)
electronnumbers = np.zeros(number_of_loops)
totionsvanished = 0

# First timestep values

potentialtype = input('Potential profile to be used. Type 1 for logarithmic or 2 for sigmoid: ')
if potentialtype == '1':
    shelliondensityfunc = shelllogiondensity
    ionphi = phi_anders_log(x_k) # potential at the points x_k
    electronphi = phi_anders_log(xe) # potential at the points xe
else:
    shelliondensityfunc = shellsigmoidiondensity
    ionphi = phi_anders_sigmoid(x_k) # potential at the points x_k
    electronphi = phi_anders_sigmoid(xe) # potential at the points xe

# one step electron ionization-method # start with this guess
# initial ion shell density using a logarithmic initial potential is rescaled to a 3d density.
old_density = np.divide(shelliondensityfunc(x_k), 4*pi*x_k**2)
nionstart = int(initialionnumber(number_of_boundaries*10000+1, shelliondensityfunc)) # 49012
ionmatrix = initialcreation(nionstart, shelliondensityfunc) # create the initial matrix of ions containing position, velocity, and shell number

Vmat = Vmatrix(eps, electronphi)  # starting Vmatrix

# find the energies which are bound and whose kinetic energy equation allow a non-imaginary velocity at some point in the simulation box
negativeepsind = (eps<=0).nonzero()
negativeeps = eps[negativeepsind]
existingepsind = ((eps<=0)*(-phi0<eps)).nonzero() 
existingeps = eps[existingepsind] 

global_rho = np.zeros_like(eps)

# Alternative 0. Use the legacy loginitialrho func.
# global_rho = loginitialrho(eps, Vmat)  # calculate density in energy space
# Alternative 1. Setting the local density in kinetic energy space to rectangular dist. for bound energies.
# xuppers = 1+b*np.log(phi0/-existingeps+(phi0/-existingeps-1)*np.exp((L-1)/b)) # upper limits for the integral. Function of eps for the highest value for x where an electron with the energy eps is still bound. 
# xuppers = np.where(xuppers<number_of_boundaries, xuppers, number_of_boundaries)
# cumsumintegranddx = dxe*csum(integrandsigmoidandersrho, xe)
# global_rho[existingepsind] = np.interp(xuppers, xe, cumsumintegranddx) # evaluate the partial integral as part of the full sum. Interpolation used to more precisely calculate the partial sum at the given upper integration limit of xupper(eps).     
# Alternative 2. Setting the global density in energy space to equal a specific function then scaling to have the same number of electrons as ions.
# global_rho[negativeepsind] = maxwell(negativeeps,1/2*beta) # 2.1 use maxwell dist.
# global_rho[existingepsind] = invsqrtdist(existingeps, existingeps[0]) # 2.2 use inv square root dist.
# global_rho[negativeepsind] = expinvsqrtdist(negativeeps, beta) # 2.3 use inv square root times exponential
global_rho[existingepsind] = doublezerodist(existingeps, phi0) # 2.4 use ...

global_rho*=nionstart/np.sum(global_rho*deps)
    
old_F = np.divide(global_rho, dJtildedeps(Vmat), out=np.zeros_like(
    global_rho), where=dJtildedeps(Vmat) != 0)  # calculate phase space density
# rescale the phase space density such that the total number of electrons = total number of ions
print('Number of ions/number of electrons before first norm.: ', nionstart/electroncounter(old_F, Vmat))
# old_F *= nionstart/electroncounter(old_F, Vmat)

# Vexact = Vmatrix(goodeps, energyphi) # true values for the velocity matrix
# plt.matshow(Vexact)
# a,_ = nnls(Vexact*deps, old_gooddensity)
# a,_,_,_ = np.linalg.lstsq(Vexact*deps, old_gooddensity)
# a = solve_triangular(4*pi*np.sqrt(2)*Vexact*deps, old_gooddensity)

# create a matrix that saves results every ? timesteps
savematrix = np.empty((number_of_loops, 3))

ionizationbool = True  # do you want ionization?
print('Is there ionization?: '+str(ionizationbool))

for j in range(number_of_loops):
    # 1. Birth ions
    # create a remaining time matrix for prior ions
    remaining_time = np.repeat(Del_t, ionmatrix[:, 0].shape)
    if ionizationbool:  # if ionization is turned on, create new ions and give them time to move
        source_ions = ioncreation(n_per_shell, x_k[-1])  # birth new ions
        # add a uniform random time [0, Del_t) remaining for the newly born ions (Reflects the fact that the ions can be born any time during the time step)
        source_remaining_time = np.random.uniform(
            0, Del_t, source_ions[:, 0].shape)

        # concatenate prior ions and recently born ions
        # add new ions to ion matrix
        ionmatrix = np.concatenate((ionmatrix, source_ions))
        remaining_time = np.concatenate(
            (remaining_time, source_remaining_time))

    # 2. Calculate the motion of ions
    # calculate electric field from potential for ions
    Efield = ElectricField(ionphi)
    ionmatrix = ionmotion(ionmatrix, remaining_time, Efield)
    ionmatrix = ionmatrix[ionmatrix[:, 2].argsort()]  # sort after shell number
    # find all ions passing outside the system
    OBC_ind = (ionmatrix[:, 0] > x_k[-1]).nonzero()
    ionmatrix = np.delete(ionmatrix, OBC_ind, axis=0)  # and delete them
    ionsvanished = len(OBC_ind[0])  # count how many vanish
    totionsvanished += ionsvanished  # and add to total

    # count number of ions with negative velocity
    nbackwardsions = len((ionmatrix[:, 1] < 0).nonzero()[0])
    frbackwardsions = nbackwardsions/len(ionmatrix)
    # strfrbackwardsions = str(frbackwardsions)[:int(np.log10(frbackwardsions)+6)]

    # 3. Calculate ion densities
    icount = ioncount(ionmatrix)  # calculate number of ions in each shell
    # unitless iondensity (Does not contain inner or outer edgepoints)
    idensity = iondensity(icount)

    # 4. Calculate new potential using neutrality
    Vmat = Vmatrix(eps, electronphi)
    ionVmat = Vmatrix(eps, ionphi)
    if ionizationbool:  # only add electrons if ionization occurs
        old_F += delF(Vmat)
    # Alt. 0. removes electrons from highest energy levels equal to number of ions removed using old potential
    new_F = old_F
    # new_F = electrondeleter(old_F, Vmat, ionsvanished) # remove equal amount of electrons as the number of ions that disappeared in the past time step
    # new_F[(eps>0).nonzero()] = 0 # remove all electrons with positive energy

    neperni0 = electroncounter(new_F, Vmat)/len(ionmatrix)
    print('After electrondeleter: ' + str(neperni0))
    # new_F *= len(ionmatrix)/electroncounter(new_F, Vmat)

    # matrix of velocities using spatial discretization of ions

    # calculate change in potential at points x_k
    # del_phi = delphitaylor(ionVmat[1:-1, :], new_F, idensity)
    
    # delphi_limit = delphilimit(ionVmat[1:-1], new_F, idensity)
    # del_phi = delphitrial2(ionVmat[1:-1, :], new_F, idensity)
    del_phi = np.zeros_like(idensity)
    # for i in range(len(idensity)):
    #     del_phi[i] = delphisingle(ionVmat[i+1,:], new_F, idensity[i])
    
    Gf = G0func(eps, new_F) # create the function for G
    halfeps = eps[negativeepsind]
    halfF = new_F[negativeepsind]
    del_phi = delphiG0(eps, Gf, ionVmat[1:-1,:], idensity) # solve for the change in potential
    
    # del_phi = delphimulti(eps, new_F, ionVmat[1:-1], idensity) 
    
    new_ionphi = ionphi[1:-1]+del_phi  # calculate new potential at points x_k

    # assuming same Efield in innermost as in second innermost shell
    innermost_phi = 2*new_ionphi[0]-new_ionphi[1]
    # assuming same Efield in outermost as in second outermost shell
    outermost_phi = 2*new_ionphi[-1]-new_ionphi[-2]
    # add innermost point to new_ionphi
    new_ionphi = np.concatenate((np.array([innermost_phi]), new_ionphi))
    # assuming same Efield in outermost and second outermost shell
    new_ionphi = np.append(new_ionphi, outermost_phi)

    # lowphi_index = (new_ionphi<-highestenergy).nonzero() # find index of potential where the potential is below the threshold set by remaining highest energy electrons
    # new_ionphi[lowphi_index] = -highestenergy # set these elements to the value where the highest energy electrons existing are bound
    # rewrite again with correct length and accounting for lower limit
    del_phi = new_ionphi-ionphi
    new_electronphi = np.interp(xe, x_k, new_ionphi) # calculate new potential at points xe
    del_electronphi = new_electronphi-electronphi # calculate change in potential at points xe
    
    new_F[(eps>0).nonzero()] = 0 # remove all electrons with positive energy
    electroneta = etae(new_F, ionVmat, del_phi)
    densityfraction = np.divide(electroneta[1:-1], idensity, out=np.zeros_like(idensity), where=idensity!=0)
    ratiovalue = np.exp(abs(np.log(abs(densityfraction))))

    halfnew_Vmat = Vmatrix(eps, new_electronphi)

    # alt. 6. calculate the new distribution function via ergodic invariant

    J0 = Jtilde(Vmat) # calculate the old ergodic invariant from the old Vmat (old potential)
    J = Jtilde(halfnew_Vmat) # calculate the old ergodic invariant from the new Vmat (new potential)

    epsinterp = np.interp(J0, J, eps) # linear interpolation to evaluate at what eps* J(eps*) = J0(eps). We know the values of J at eps.
    depsinterp = depscalc(epsinterp) # calculating the new deps
    
    # Simple linear interpolation
    # new_F = np.interp(eps, epsinterp, new_F) # pretend that new_F is the values of the distribution function at epsinterp. Then the values at the desired list of energies eps is linearly interpolated.
    # new_F *= len(ionmatrix)/electroncounter(new_F, halfnew_Vmat) # normalize

    # Interpolation preserving the number of electrons
    new_Vmat = Vmatrix(epsinterp, new_electronphi)
    Vx2 = LI(halfnew_Vmat)
    Vx2[0] /= 2
    Vx2[-1] /= 2
    Vx2int = np.sum(Vx2, axis=0) # approximate integral over x axis
    new_F = np.divide(np.interp(eps, epsinterp, new_F*np.sum(LI(new_Vmat), axis=0)*depsinterp), deps*Vx2int, out=np.zeros_like(new_F), where=Vx2int!=0)     
    print(len(ionmatrix)/electroncounter(new_F, halfnew_Vmat))
    new_F *= len(ionmatrix)/electroncounter(new_F, halfnew_Vmat)

    # alt.alt. 6
    # # calculate the new energy levels directly
    # new_eps = eps+epschange(del_electronphi, halfnew_Vmat)
    # # move the previous values of the distribution function to the new energy levels then interpolate back to find the new distribution function at the old energy levels
    # new_F = np.interp(eps, new_eps, new_F)
    # new_F *= len(ionmatrix)/electroncounter(new_F, halfnew_Vmat)  # normalize

    
    # Plot for each iteration (debugging)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, )
    fig.suptitle('Timestep: '+str(counter))
    
    ax1.set_title('Potential')
    ax1.semilogy(xe, new_electronphi*electrontemperature/beta, '.', color='k')
    ax1.semilogy(xe, -new_electronphi*electrontemperature/beta, '.', color='r')
    # line for electron temperature
    ax1.axhline(electrontemperature, color='b')
    ax1.set(xlabel='Distance from comet center [R'+'$_{C}$]', ylabel='Potential [V]')
    ax1.set_ylim(1e-3, 1e2)
    ax1.text(0, 5e1, 'Backwards ion frac.: '+str(frbackwardsions)[:5])
    # change in potential debugging
    ax1.semilogy(x_k, del_phi*electrontemperature/beta, 'x', color='k')
    ax1.semilogy(x_k, -del_phi*electrontemperature/beta, 'x', color='r')
    
    ax2.set_title('Electron distribution function')
    ax2.semilogy(eps*electrontemperature/beta, new_F, '.', color='k')
    ax2.semilogy(eps*electrontemperature/beta, -new_F, '.', color='r')
    ax2.set(xlabel='Electron energy [eV]', ylabel= 'F')
    ax2.yaxis.tick_right()
    
    # integrand = LI(halfnew_Vmat)
    # integrand[0] /= 2
    # integrand[-1] /= 2
    # rho = new_F*np.sum(integrand, axis=0)  # density in energy space
    # ax2.set_title('Density in energy space')
    # ax2.semilogy(eps*electrontemperature/beta, rho, '.', color='k')
    # ax2.semilogy(eps*electrontemperature/beta, -rho, '.', color='r')
    # ax2.set(xlabel='Electron energy [eV]', ylabel='$rho$')
    # ax2.yaxis.tick_right()
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, )
    fig.suptitle('Timestep: '+str(counter))
    
    ax1.set_title('Electron vs. ion density')
    ax1.semilogy(x_k, electroneta, '.', color='r')
    ax1.semilogy(x_k[1:-1], idensity, '.', color='b')
    ax1.set(xlabel='Distance from comet center [R'+'$_{C}$]', ylabel='Density')
    
    ax2.set_title('Electron vs. ion density')
    ax2.semilogy(x_k[1:-1], ratiovalue, '.', color='k')
    
    

    # ax2.text(0.01, 0.37, '#_e before', transform = ax2.transAxes)
    # ax2.text(0.01, 0.31, 'interpolation:', transform = ax2.transAxes)
    # ax2.text(0.01, 0.25, beforestring, transform = ax2.transAxes)

    # ax2.text(0.01, 0.13, '#_e after', transform = ax2.transAxes)
    # ax2.text(0.01, 0.07, 'interpolation:', transform = ax2.transAxes)
    # ax2.text(0.01, 0.01, afterstring, transform = ax2.transAxes)

    # ax3.plot(x_k, new_phi,'.', color='k')
    # ax3.axvline(furthestion, color='k')
    # ax3.set(xlabel = 'Distance from comet center [R'+'$_{C}$]', ylabel = 'Potential')
    # ax4.plot(eps, new_F, '.', color='k')
    # ax4.set(xlabel='Electron energy', ylabel= 'F')
    # ax4.yaxis.tick_right()

    # new_F = new_F + delF(new_Vmat) # add the created electrons
    # new_F = electrondeleter(new_F, new_Vmat, ionsvanished) # Alt. 2. removes electrons from highest energy levels equal to number of ions removed using new potential and interpolated back to old energy levels

    # Calculate current total kinetic energy in the system
    ionnumbers[counter] = len(ionmatrix)
    electronnumbers[counter] = electroncounter(new_F, halfnew_Vmat)

    # total kinetic energy of electrons
    tot_e_energy = electronenergy(new_F, halfnew_Vmat)
    # average kinetic energy of electrons
    avg_e_energy = tot_e_energy/electroncounter(new_F, halfnew_Vmat)
    # average kinetic energy of ions
    avg_i_energy = averageionenergy(ionmatrix)
    tot_i_energy = avg_i_energy*len(ionmatrix)  # total kinetic energy of ions
    # total kinetic energy of electrons all particles
    tot_energy = tot_e_energy+tot_i_energy

    # avg total energy can be calculated this way since we demand equal number of both electrons and ions
    averageenergies[counter, :] = [avg_e_energy, avg_i_energy,
                                   (avg_e_energy+avg_i_energy)/2, tot_e_energy, tot_i_energy, tot_energy]

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
    # potential evaluated at the space discretization used for ions (x_k)
    ionphi = new_ionphi
    # potential evaluated at the space discretization used for electrons (xe)
    electronphi = new_electronphi
    old_F = new_F
    # eps = new_eps # This is wrong when resampling.

    counter += 1  # increment the number of loops performed
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
    # andersprop = ((1+2*(x_k[1:-1]-1)*phi0)**(1/2)-1)/(x_k[1:-1]**2*phi0) # for linear potential
    # # andersprop = 1/x_k[1:-1]*(pi/2)**(1/2)*np.exp(1/(2*phi0))*(erf(((1+2*phi0*np.log(x_k[1:-1]))/(2*phi0))**(1/2)) - erf(1/(2*phi0)**(1/2)))# for logarithmic potential
    # plt.plot(idensity/andersprop, '.', color='k')

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Timestep: '+str(counter))
ax1.set_title('Potential')
ax1.plot(xe, new_electronphi/beta*electrontemperature, '.', color='k')
ax1.axhline(electrontemperature, color='k', linewidth=1)
ax1.set(
    xlabel='Distance from comet center [R'+'$_{C}$]', ylabel='Potential [V]')
ax1.set_ylim([0, max(new_electronphi/beta*electrontemperature)*1.05])
ax2.set_title('Electron distribution function')
ax2.plot(eps/beta*electrontemperature, new_F, '.', color='k')
ax2.set(xlabel='Electron energy [eV]', ylabel='F')
ax2.yaxis.tick_right()

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Velocity distribution of ions')
ax1.set_title('Histogram')
ax1.hist(ionmatrix[:, 1], bins=100)
ax1.set(xlabel='Velocity', ylabel='Number of ions')
ax1.axvline(0, color='k', linewidth=1)
ax2.set_title('CDF')
ax2.hist(ionmatrix[:, 1], bins=100, cumulative=True)
ax2.set(xlabel='Velocity', ylabel='Number of ions')
ax2.yaxis.tick_right()

plt.figure()
plt.title('Number of particles')
plt.xlabel('Timestep number')
plt.ylabel('Number of particles')
plt.plot(electronnumbers, color='k', linestyle='--', label='Electrons')
plt.plot(ionnumbers, color='k', linestyle=':', label='Ions')
plt.ylim([0, max(electronnumbers)*1.05])
plt.legend()

plt.figure()
plt.title('Total energies in the system')
plt.xlabel('Timestep number')
plt.ylabel('Unitless average kinetic energy')
plt.plot(averageenergies[:, 3], color='k', linestyle='--', label='Electron')
plt.plot(averageenergies[:, 4], color='k', linestyle=':', label='Ion')
plt.plot(averageenergies[:, 5], color='k', label='Both')
plt.ylim([0, max(averageenergies[:, 5])*1.05])
plt.legend()

plt.figure()
plt.title('Average energies in the system')
plt.xlabel('Timestep number')
plt.ylabel('Unitless average kinetic energy')
plt.plot(averageenergies[:, 0], color='k', linestyle='--', label='Electron')
plt.plot(averageenergies[:, 1], color='k', linestyle=':', label='Ion')
plt.plot(averageenergies[:, 2], color='k', label='Both')
plt.ylim([0, max(max(averageenergies[:, 1]), max(averageenergies[:, 0]))*1.05])
plt.legend()

end_time = time.time()

elapsed_time = end_time-start_time
print('Simulated time: '+str(simulation_time)+' s')
print('Elapsed time: '+str(elapsed_time)+' s')
