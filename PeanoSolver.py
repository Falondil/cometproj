# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:27:07 2022

@author: Vviik

Solve plasma distribution of continuously ionized plasma from initially neutral coma

via the method of Peano et al. (2006): Smooth electrons, finite protons

"""

import numpy as np

pi = np.pi
q = 1 # normalization, tbd. 


def deltar

def inv_sq(r):
    return 1/r**2


def ergodic_inv(eps, phi): # eq. 75. phi is a vector of all phi_r_k


        
def dJdeps(eps, phi, rvec) # eq. 74. rvec is a vector of all r_k up to R(eps)



def sum_dJdphirk(eps, phi, delta_phi, rvec) # eq. 76. rvec is a vector of all r_k. 
    
    return sum([(eps-q*phi[i])**(1/2)*rvec[i]^2 * (rvec)[i]])


def del_phi(new_phi, old_phi):
    return [new_phi[i]-old_phi[i] for i in range(len(new_phi))]


def del_eps(eps, phi, del_phi, rvec): # eq. 77. all but eps are vectors
    return -sum_dJdphirk(eps, phi, delta_phi, rvec)/dJdeps(eps, phi, rvec)



