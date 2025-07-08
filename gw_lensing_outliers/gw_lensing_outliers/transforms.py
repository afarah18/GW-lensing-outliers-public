import numpy as np

H0 = 67.66
Om0 = 0.30966
##################################################
## COSMOLOGY
##################################################

def Phi(x):
    num = 1 + 1.320*x + 0.4415* np.power(x,2) + 0.02656*np.power(x,3)
    den = 1 + 1.392*x + 0.5121* np.power(x,2) + 0.03944*np.power(x,3)
    return num/den
def xx(z,Om0):
    return (1.0-Om0)/Om0/np.power(1.0+z,3)

def dL_approx(z,H0=H0,Om0=Om0):
    D_H = 3e5 / H0 / 1000. # Gpc
    return 2.*D_H * (1.+z) * (Phi(xx(0.,Om0)) - Phi(xx(z,Om0))/np.sqrt(1.+z))/np.sqrt(Om0)

_z_arr = np.logspace(-4,2,num=500,base=10)
def z_at_dl_approx(dl,H0=H0,Om0=Om0,zmin=1e-3,zmax=100):
    #dl in Gpc
    return np.interp(dl, dL_approx(_z_arr,H0,Om0),_z_arr, left=zmin, right=zmax, period=None)

def Ez_inv(z,Om0):
    return 1./np.sqrt((1.-Om0) + Om0*np.power((1.+z),3))

def diff_comoving_volume_approx(z,H0=H0,Om0=Om0):
    dL = dL_approx(z,H0,Om0) #Gpc
    Ez_i = Ez_inv(z,Om0)
    D_H = 3e5 / H0 / 1000. #Gpc
    return (4.*np.pi) * np.power(dL,2) * D_H * Ez_i / np.power(1.+z,2.) # Gpc**3

def dDLdz_approx(z, H0=H0, Om0=Om0):
    dL = dL_approx(z,H0,Om0) #Gpc
    Ez_i = Ez_inv(z,Om0)
    D_H = 3e5 / H0 / 1000. #Gpc
    return np.abs(dL/(1.+z) + (1.+z)*D_H * Ez_i)

##################################################
## LENSING
##################################################

def inferred_quantities_from_lensing(mags, det_masses, true_distances, H0=H0, Om0=Om0):
    inferred_distances = true_distances/np.sqrt(mags)
    inferred_redshifts = z_at_dl_approx(inferred_distances, H0, Om0)
    inferred_src_masses = det_masses/(1 + inferred_redshifts)

    return inferred_distances, inferred_redshifts, inferred_src_masses