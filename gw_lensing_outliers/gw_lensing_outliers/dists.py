import numpy as np
from scipy.special import erf, expit
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import Planck15
from colossus.cosmology import cosmology as colcosmo
from colossus.lss import mass_function
colcosmo.setCosmology('planck15')
import astropy.units as u

from gw_lensing_outliers import transforms as trans

H0 = trans.H0
Om0 = trans.Om0
ZMAX = 10
BETA = 3.4 # redshift distribution slope after peak
ZPEAK = 2.4 # redshift distribution peak
Z_TEST = np.linspace(1e-4,ZMAX,num=100)

MU_MAX = 2500

# SIS default parameters
M200_MAX = 16
M200_MIN = 6
COSMO = colcosmo.setCosmology('planck15')

###########################################################################
## GENERAL
###########################################################################
def powerlaw(xx, alpha, high, low):
    norm = np.where(alpha==-1,
                     1 / np.log(high / low),
                     (1 + alpha) / (high ** (1 + alpha) - low ** (1 + alpha))
                    )
    prob = np.power(xx, alpha)
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob

def truncnorm(xx, mu, sigma, high, low):
    norm = 2 ** 0.5 / np.pi ** 0.5 / sigma
    norm /= erf((high - mu) / 2 ** 0.5 / sigma) + erf((mu - low) / 2 ** 0.5 / sigma)
    prob = np.exp(-np.power(xx - mu, 2) / (2 * sigma ** 2))
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob

###########################################################################
## HALO MASS FUNCTIONS
###########################################################################

def Tinker08_mine(mass,redshift):
    raise NotImplementedError

    rho_crit0 = 2.77536627245708E11 # solar masses * h^2 / Mpc^3
    rho_m0 = rho_crit0 * Om0 
    rho_m = rho_m0*(1+redshift)**3
    R = (3.0 * mass / 4.0 / np.pi / rho_m0 )**(1.0 / 3.0)

    Delta = mass /((4/3) * np.pi * rho_m * R**3)

    fit_Delta = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])
    fit_A0 = np.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
    fit_a0 = np.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
    fit_b0 = np.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
    fit_c = np.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])

    A0 = np.interp(Delta, fit_Delta, fit_A0)
    a0 = np.interp(Delta, fit_Delta, fit_a0)
    b0 = np.interp(Delta, fit_Delta, fit_b0)
    c = np.interp(Delta, fit_Delta, fit_c)

    alpha = np.power(10,-(0.75/(np.log(Delta/75)))**1.2)

    A = A0 * (1+redshift)**(-0.14)
    a = a0 * (1+redshift)**(-0.06)
    b = b0 * (1+redshift)**(-alpha)
    
    sigma = np.sqrt(
        np.trapz(y=Pk * WkR * k**2,x=k)
    )
    
    f_of_sigma = A * ((b/sigma)**a + 1)*np.exp(-c/sigma**2)
    return f_of_sigma * rho_m / mass * dln_siginv_dm

def Tinker08(mass,redshift):
    """Halo mass function from Tinker et al. 2008"""
    
    rho_crit0 = 2.77536627245708E11 # solar masses * h^2 / Mpc^3
    rho_m0 = rho_crit0 * Om0 
    R = (3.0 * mass / 4.0 / np.pi / rho_m0 )**(1.0 / 3.0)

    Ez_squared =(1.-Om0) + Om0*np.power((1.+redshift),3)
    Delta_m = 200 * Ez_squared/Om0/(1+redshift)**3 #mass /((4/3) * np.pi * rho_m * R**3)
    
    sigma = COSMO.sigma(R, redshift)
    
    fit_Delta = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])
    fit_A0 = np.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
    fit_a0 = np.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
    fit_b0 = np.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
    fit_c0 = np.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])
		
	# Compute fit parameters and f-function
    # if Delta_m < fit_Delta[0]:
    #     raise Exception('Delta_m %d is too small, minimum %d.' % (Delta_m, fit_Delta[0]))
    # if Delta_m > fit_Delta[-1]:
    #     raise Exception('Delta_m %d is too large, maximum %d.' % (Delta_m, fit_Delta[-1]))

    A0 = np.interp(Delta_m, fit_Delta, fit_A0)
    a0 = np.interp(Delta_m, fit_Delta, fit_a0)
    b0 = np.interp(Delta_m, fit_Delta, fit_b0)
    c0 = np.interp(Delta_m, fit_Delta, fit_c0)

    alpha = 10**(-(0.75 / np.log10(Delta_m / 75.0))**1.2)
    A = A0 * (1.0 + redshift)**-0.14
    a = a0 * (1.0 + redshift)**-0.06
    b = b0 * (1.0 + redshift)**-alpha
    c = c0
    f_T08 = A * ((sigma / b)**-a + 1.0) * np.exp(-c / sigma**2)

    d_ln_sigma_d_ln_R = COSMO.sigma(R, redshift, derivative = True)
    rho_Mpc = COSMO.rho_m(0.0) * 1E9
    dn_dlnM = -(1.0 / 3.0) * f_T08 * rho_Mpc / mass * d_ln_sigma_d_ln_R

    return dn_dlnM

###########################################################################
## MAGNIFICATIONS
###########################################################################

# Phenomenological
## From Dai+17 appendix:
# z_vals = np.array([0.7,1,2,3,5,10,20])
# sigma_vals = np.array([0.008,0.010,0.028,0.050,0.078,0.110,0.150])
# expnegdelta_vals = np.array([1.0380,1.0465,1.0700,1.0859,1.1065,1.1327,1.1649])
# delta_vals = -np.log(expnegdelta_vals)
# t0_vals = np.array([0.365,0.399,0.471,0.511,0.557,0.609,0.666])
## Calibrated to Oguri 2017 for low-z and using above for high z:
z_vals = [0.01,0.1,0.2,0.4, 1,2,3,5,10,20]
sigma_vals = [0.005,  0.00728209,  0.00756015,  0.00616215, 0.010,0.028,0.050,0.078,0.110,0.150]
t0_vals = [0.12,   0.30726794,  0.32334431,  0.37662823,  0.399, 0.471,0.511,0.557,0.609,0.666]
delta_vals = [-0.03929844, -0.03929844, -0.03900541, -0.03867672, -0.04545126, -0.06765865, -0.08240914, -0.10120188, -0.12460416, -0.15263525]

t_arr = np.linspace(0,100,num=10000)

def _invA(t0,lam=5):
    return np.trapz(y=np.exp(lam/(t_arr+t0)-2*t_arr),x=t_arr)
_invA = np.vectorize(_invA)
def logmag_pdf(logmu,z):
    """Magnification distribution from Dai et al. 2017"""
    sigma=np.interp(z,z_vals,sigma_vals)
    t0=np.interp(z,z_vals,t0_vals)
    delta=np.interp(z,z_vals,delta_vals)
    lam=5
    
    A = 1./_invA(t0,lam)
    if type(logmu) in [type(1), type(1.),type(np.log(1))]:
        return np.trapz(y=np.exp(lam/(t_arr+t0)-2*t_arr)*np.exp(-(logmu - delta - t_arr)**2/(2*sigma**2)),
                        x=t_arr) * A / np.sqrt(2*np.pi) / sigma
    else:
        p = np.zeros_like(logmu)
        for i in range(len(p)):
            p[i] = np.trapz(y=np.exp(lam/(t_arr+t0)-2*t_arr)*np.exp(-(logmu[i] - delta - t_arr)**2/(2*sigma**2)),
                        x=t_arr) * A / np.sqrt(2*np.pi) / sigma
        return p

def logmag_pdf_Lambda(logmu,sigma,t0,delta,lam=5):
    """Magnification distribution from Dai et al. 2017, adapted to take in hyperparameters Lambda={sigma,t0,delta} rather than redshifts"""
    # assert logmu.shape == sigma.shape and logmu.shape == t0.shape and logmu.shape == delta.shape
    
    tgrid, t0grid = np.meshgrid(t_arr,t0)
    tgrid, sigmagrid = np.meshgrid(t_arr,sigma)
    tgrid, deltagrid = np.meshgrid(t_arr,delta)
    tgrid, logmugrid = np.meshgrid(t_arr,logmu)

    A = 1./np.trapz(y=np.exp(lam/(tgrid+t0grid)-2*tgrid),x=tgrid,axis=1)
    p = np.trapz(y=np.exp(lam/(tgrid+t0grid)-2*tgrid)*np.exp(-(logmugrid - deltagrid - tgrid)**2/(2*sigmagrid**2)),
                    x=tgrid,axis=1) * A / np.sqrt(2*np.pi) / sigma
    return p

def logmag_pdf_t0const(logmu,z,t0_const):
    """Magnification distribution from Dai et al. 2017 with additive constant to t0"""
    sigma=np.interp(z,z_vals,sigma_vals)
    t0=np.interp(z,z_vals,t0_vals) + t0_const
    delta=np.interp(z,z_vals,delta_vals)
    lam=5
    
    A = 1./_invA(t0,lam)
    
    if type(logmu) in [type(1), type(1.),type(np.log(1))]:
        return np.trapz(y=np.exp(lam/(t_arr+t0)-2*t_arr)*np.exp(-(logmu - delta - t_arr)**2/(2*sigma**2)),
                        x=t_arr) * A / np.sqrt(2*np.pi) / sigma
    else:
        p = np.zeros_like(logmu)
        for i in range(len(p)):
            p[i] = np.trapz(y=np.exp(lam/(t_arr+t0)-2*t_arr)*np.exp(-(logmu[i] - delta - t_arr)**2/(2*sigma**2)),
                            x=t_arr) * A / np.sqrt(2*np.pi) / sigma
        return p

# Singular Isothermal Sphere

def einstein_radius(velocity_dispersion, z_lens, z_source):
    """ analytic form for the case of an SIS. """
    d_s = Planck15.angular_diameter_distance(z_source).to(u.Mpc)
    d_ls = Planck15.angular_diameter_distance_z1z2(z_lens,z_source).to(u.Mpc)
    return 4 * np.pi * np.power(velocity_dispersion/c.c,2) * d_ls/d_s

def magnification_cross_section(M200,z_lens,z_source,mu_thresh):
    vel_disp = velocity_dispersion(M200,z_lens)
    return 2 * np.pi * einstein_radius(vel_disp,z_lens,z_source)**2 \
        * (mu_thresh**2 + 1) / (mu_thresh**2 - 1)**2

def dsigma_dmu(mu, M200,z_lens,z_source):
    "analytic expression for the magnification cross section as a function of magnification"
    theta_E = einstein_radius(velocity_dispersion=velocity_dispersion(M200,z_lens),z_lens=z_lens,z_source=z_source)
    return 2*np.pi*theta_E**2/(mu-1)**3

def velocity_dispersion(M200,z):
    """ Velocity dispersion in meters/s of halos with masses defined by 
    M200 and redshifts of z. This is the for the case of a singular isothermal sphere.
    parameters:

    M200 (float): M200 in solar masses
    z (float): redshift

    returns: velocity dispersion, sigma_v in meters/s
    """
    M200 = M200 * u.M_sun
    rho_crit = Planck15.critical_density(z).to(u.kg/u.m**3)
    rho_200 = 200*rho_crit

    return np.power(np.sqrt(np.pi * c.G**3 * rho_200/6.) * M200 , 1/3.).to(u.m/u.s)

def dtau_dmu(mu,z_source_true,log10m200_min,log10m200_max):
    z_lens_dummy = np.linspace(0,z_source_true,50)
    m200_dummy = np.logspace(log10m200_min,log10m200_max,100,base=10)
    m200_grid, z_lens_grid, mu_grid = np.meshgrid(m200_dummy,z_lens_dummy,mu)
    d2n_dlogmdv = (Planck15.H0/(100*u.km/u.s/u.Mpc)) **3 \
            * Tinker08(m200_grid,z_lens_grid)
    dVc_dz = Planck15.differential_comoving_volume(z_lens_grid) # already divided by 4pi
    ds_dmu = dsigma_dmu(mu_grid, m200_grid,z_lens_grid,z_source_true)
    # marginalize over halo masses
    integrand = dVc_dz * ds_dmu * (d2n_dlogmdv / u.Mpc**3 / m200_grid) # last factor is the jacobian to go from dn/dlnM -> dn/dM
    return np.trapz(y=np.trapz(integrand,m200_dummy,axis=1),x=z_lens_dummy,axis=0) 

dtau_dmu_sis_10 = np.zeros_like(Z_TEST)
for i in range(len(Z_TEST)):
    dtau_dmu_sis_10[i] = dtau_dmu(10,Z_TEST[i],M200_MIN,M200_MAX).value
from scipy.interpolate import interp1d
scaling = (10-1)**3 # the tau interpolant is defined at mu=10 so we have to apply this to get it at mu=muthr
dtau_dmu_sis=interp1d(Z_TEST,dtau_dmu_sis_10 * scaling)

def dtau_dmu_interp():
    raise NotImplementedError
    lnmu_dummy = np.log(np.linspace(mu_thr,1e5))
    z_arr = np.logspace(-1,np.log10(5),base=10,num=100)
    dtau_dz_dai = np.zeros_like(z_arr)
    for i in range(len(dtau_dz_dai)):
        dtau_dz_dai[i] = np.trapz(x=lnmu_dummy,y=logmag_pdf(lnmu_dummy,z_arr[i]))

def differential_optical_depth(M200,z_lens,z_source,mu_thresh,model='tinker08'):
    """d tau / dlog(M) dz_lens"""
    dVcdzdskyangle = Planck15.differential_comoving_volume(z_lens) # this is divided by 4pi
    dNdlogmass_tinker = (Planck15.H0/(100*u.km/u.s/u.Mpc)) **3 \
        * mass_function.massFunction(M200,z_lens,mdef='200c',
                                     model=model,q_out='dndlnM') / u.Mpc**3
    return (magnification_cross_section(M200,z_lens,z_source,mu_thresh) 
        * dVcdzdskyangle * dNdlogmass_tinker)*u.sr

def dtau_dzldzs(z_lens, z_source, mu_thresh,log10m200_min=M200_MIN,log10m200_max=M200_MAX,model='tinker08'):       
    m200_arr = np.logspace(log10m200_min,log10m200_max,base=10)
    return np.trapz(x=np.log(m200_arr),y=differential_optical_depth(m200_arr,z_lens,z_source,mu_thresh,model))
dtau_dzldzs = np.vectorize(dtau_dzldzs)

def dtau_dzs(z_source,mu_thresh,**kwargs):
    z_lens_arr = np.linspace(0,z_source)
    return np.trapz(x=z_lens_arr, y=dtau_dzldzs(z_lens=z_lens_arr,z_source=z_source,mu_thresh=mu_thresh,**kwargs),axis=0)
# dtau_dzs = np.vectorize(dtau_dzs)

# generic, SIS-like model but with no assumed HMF
def dtau_dmu_generic(mu,t):
    return t/(mu-1)**3 
def t_volume_pl(z,t0,alpha_t):
    return t0 * trans.diff_comoving_volume_approx(z) * (1+z)**alpha_t
def t_volume_bpl(z,t0,a,b,brk):
    return t0 * trans.diff_comoving_volume_approx(z) * ((1+z)/(1+brk))**np.where(z<brk,a,b)
def t_const(z,t0):
    return t0

###########################################################################
## MASSES
###########################################################################    
def powerlaw_peak(m1,mMin,mMax,alpha,sig_m1,mu_m1,f_peak):
    tmp_min = 2.
    tmp_max = 150.
    dmMax = 2
    dmMin = 1

    # Define power-law and peak
    p_m1_pl = powerlaw(m1,low=tmp_min,high=tmp_max,alpha=alpha)
    p_m1_peak = np.exp(-(m1-mu_m1)**2/(2.*sig_m1**2))/np.sqrt(2.*np.pi*sig_m1**2)

    # Compute low- and high-mass filters
    low_filter = np.exp(-(m1-mMin)**2/(2.*dmMin**2))
    low_filter = np.where(m1<mMin,low_filter,1.)
    high_filter = np.exp(-(m1-mMax)**2/(2.*dmMax**2))
    high_filter = np.where(m1>mMax,high_filter,1.)

    # Apply filters to combined power-law and peak
    return (f_peak*p_m1_peak + (1.-f_peak)*p_m1_pl)*low_filter*high_filter

def two_component_single(
    mass, alpha, mmin, mmax, lam, mpp, sigpp, delta_m, gaussian_mass_maximum=100
):
    r"""
    gwpopulation version of power law + peak.

    Parameters
    ----------
    mass: array-like
        Array of mass values (:math:`m`).
    alpha: float
        Negative power law exponent for the black hole distribution (:math:`\alpha`).
    mmin: float
        Minimum black hole mass (:math:`m_\min`).
    mmax: float
        Maximum black hole mass (:math:`m_\max`).
    lam: float
        Fraction of black holes in the Gaussian component (:math:`\lambda_m`).
    mpp: float
        Mean of the Gaussian component (:math:`\mu_m`).
    sigpp: float
        Standard deviation of the Gaussian component (:math:`\sigma_m`).
    delta_m: float
        Rise length of the low end of the mass distribution.
    gaussian_mass_maximum: float, optional
        Upper truncation limit of the Gaussian component. (default: 100)
    """
    p_pow = powerlaw(mass, alpha=-alpha, high=mmax, low=mmin)
    p_norm = truncnorm(mass, mu=mpp, sigma=sigpp, high=gaussian_mass_maximum, low=mmin)
    prob = (1 - lam) * p_pow + lam * p_norm
    window = smoothing(mass,mmin,mmax,delta_m)
    return prob * window

def smoothing(mass,mmin,mmax,delta_m):
    shifted_mass = np.nan_to_num((mass - mmin) / delta_m, nan=0)
    shifted_mass = np.clip(shifted_mass, 1e-6, 1 - 1e-6)
    exponent = 1 / shifted_mass - 1 / (1 - shifted_mass)
    window = expit(-exponent)
    window *= (mass >= mmin) * (mass <= mmax)
    return window

###########################################################################
## DISTANCES AND COSMOLOGY
###########################################################################


def unif_comoving_rate(z,H0=H0,Om0=Om0):
    dvc_dz = trans.diff_comoving_volume_approx(z,H0,Om0) # Gpc**3
    dtsrc_dtdet = (1 + z)
    return dvc_dz / dtsrc_dtdet

def shouts_murmurs(z,zp,alpha,beta,H0=H0,Om0=Om0):
    unif_comov = unif_comoving_rate(z, H0, Om0) # Gpc**3
    c0 = 1 + (1 + zp)**(-alpha - beta)
    return unif_comov * c0  * (1 + z)**alpha / ( 1 + (np.power((1.+z)/(1.+zp),(alpha+beta)))) 

def powerlaw_redshift_distribution(z,alpha,H0=H0,Om0=Om0,norm=False):
    unif_comov = unif_comoving_rate(z, H0, Om0) # Gpc**3
    p = unif_comov * (1 + z)**alpha
    if norm:
        normalization = np.trapz(y= unif_comoving_rate(Z_TEST, H0, Om0) * (1 + Z_TEST)**alpha,x=Z_TEST)
    else:
        normalization=1 
    return p / normalization

###########################################################################
## FULL CBC POPULATION
###########################################################################

def threeDpopulation(m1,q,z,m1_kwargs,q_beta,z_alpha,normalize=False):
    pm1 = two_component_single(m1,**m1_kwargs)
    m2 = m1*q
    pq = powerlaw(q,q_beta,1,m1_kwargs['mmin']/m1) * smoothing(m2,m1_kwargs['mmin'],m1,m1_kwargs['delta_m'])
    # deal with edge cases
    pq = np.where(m1_kwargs['mmin']==m1,1,pq)
    pz = powerlaw_redshift_distribution(z=z,alpha=z_alpha)
    if normalize:
        m1_test = np.linspace(1,100,num=100)
        q_test = np.linspace(0,1,num=25)
        m1_grid,q_grid = np.meshgrid(m1_test,q_test)
        norm = np.trapz(y=np.trapz(x=m1_test,y=two_component_single(m1_grid,**m1_kwargs)
                                * powerlaw(q_grid,q_beta,1,m1_kwargs['mmin']/m1_grid) * smoothing(m1_grid*q_grid,m1_kwargs['mmin'],m1_grid,m1_kwargs['delta_m']),
                                axis=1),x=q_test
                        )* np.trapz(x=Z_TEST,y=powerlaw_redshift_distribution(z=Z_TEST,alpha=z_alpha))
    else:
        norm = 1.
    return pm1 * pq * pz / norm