import numpy as np
from scipy.stats import truncnorm
from scipy.interpolate import RegularGridInterpolator
from tqdm import trange
import os
import sys
import h5py

from GWMockCat.posterior_utils import generate_obs_from_true_list, m1m2_from_mceta, mchirp, etafunc
from GWMockCat.vt_utils import draw_thetas

from gw_lensing_outliers import dists, sampling
from gw_lensing_outliers import transforms as trans

OUTLIER_THRESH=0.01
N_EVS_O3 = 69
MASS_KEYS = ["alpha", "mmin", "mmax", "lam", "mpp", "sigpp", "delta_m","gaussian_mass_maximum"]

# TODO: write docstrings

def draw_true(hyperpe,N,rng,z_bounds=[0.0001,100],mass_bounds=[1,500],
              mass_keys=["alpha", "mmin", "mmax", "lam", "mpp", "sigpp", "delta_m","gaussian_mass_maximum"],
              normalize=False
              ):
    src_masses = sampling.inverse_transform_sample(dists.two_component_single,mass_bounds,rng=rng,N=N,
                                                   num_interp_points=int(1e4), **hyperpe[mass_keys])
    q = sampling.inverse_transform_sample(dists.powerlaw,[1/mass_bounds[1],1],rng=rng,N=N,low=1/mass_bounds[1],high=1,alpha=hyperpe['beta'])
    true_redshifts = sampling.inverse_transform_sample(dists.shouts_murmurs,z_bounds,rng=rng,N=N,
                                                    alpha=hyperpe['lamb'],zp=dists.ZPEAK,beta=dists.BETA,) 
    true_distances = trans.dL_approx(true_redshifts)
    thetas = draw_thetas(N,rng)

    pdraw = dists.two_component_single(src_masses,**hyperpe[mass_keys]) 
    pdraw *= dists.shouts_murmurs(true_redshifts,alpha=hyperpe['lamb'],zp=dists.ZPEAK,beta=dists.BETA)
    pdraw *= dists.powerlaw(q,hyperpe['beta'],high=1,low=1/mass_bounds[1]) # dists.powerlaw(q,0,1,hyperpe['mmin']/src_masses) * dists.smoothing(src_masses*q,hyperpe['mmin'],src_masses,hyperpe['delta_m'])
    if normalize:
        m1_arr = np.linspace(mass_bounds[0],mass_bounds[1],num=100)
        norm = np.trapz(x=m1_arr,y=dists.two_component_single(m1_arr,**hyperpe[mass_keys])
                        )#* np.trapz(Z_TEST,dists.shouts_murmurs(z=Z_TEST,alpha=hyperpe['lamb'],zp=dists.ZPEAK,beta=dists.BETA))
        pdraw/=norm
    return src_masses, src_masses * (1+true_redshifts), q, true_redshifts, true_distances, thetas, pdraw

def create_ppds(rng, hyperparam_samples, optimal_snr_interpolator, distance_unit, uncert_dict, cosmo_dict, n_evs=N_EVS_O3, outlier_thresh=OUTLIER_THRESH,num_injs=int(1e7)):
    
    # create semianalytic injections selected based off of their maximum likelihood values    
    ## first, generate true values from a distribution that is at least close to most hyperposterior samples
    characteristic_pop = hyperparam_samples.mean()
    characteristic_pop['delta_m'] = 4.95
    characteristic_pop['mmax'] = hyperparam_samples['mmax'].max() + 20
    src_masses, det_masses, q, true_redshifts, true_distances, thetas, pdraw = draw_true(characteristic_pop,num_injs,rng,z_bounds=[0.001,2],mass_bounds=[1,characteristic_pop['mmax']])
    ## then jitter them and select them
    observed_detected, true_detected  = generate_obs_from_true_list(src_masses,
                                                                    src_masses*q,
                                                                    true_redshifts,
                                                                    optimal_snr_interpolator,
                                                                    cosmo_dict,
                                                                    rng,
                                                                    distance_unit,
                                                                    uncert=uncert_dict)
    obs_m1det, obs_m2det = m1m2_from_mceta(observed_detected['Mc_det_obs'],observed_detected['eta_obs'])
    true_m1s = true_detected['m1']
    true_q = true_detected['m2']/true_detected['m1']
    true_z = true_detected['z']
    obs_dL = observed_detected['t_obs'] * optimal_snr_interpolator(obs_m1det, obs_m2det,grid=False)/observed_detected['rho_obs']
    obs_z = trans.z_at_dl_approx(obs_dL)
    obs_m1s = obs_m1det / (1 + obs_z)
    obs_pdraw = pdraw[true_detected['detected_index']]
    ndet = len(obs_m1s)

    # find the distribution of maximum masses and minimum distances
    n_hyperpe = len(hyperparam_samples['alpha'])
    m1max = np.zeros(n_hyperpe)
    Dmin = np.zeros(n_hyperpe)
    ## for each population draw from PLP...
    for i, row in hyperparam_samples.iterrows(): 
        ## ...draw many injections, weighted by this population draw
        expanded_row = row
        expanded_row['delta_m']=characteristic_pop['delta_m']
        p_target = dists.threeDpopulation(true_m1s,true_q,true_z,
                                    m1_kwargs=expanded_row[MASS_KEYS],
                                    q_beta=row['beta'],z_alpha=row['lamb'],normalize=False)

        p_ratio = p_target / obs_pdraw
        if np.isnan(np.sum(p_ratio)) or np.any(p_ratio<0):
            print(row)
            print(p_target[p_target<0],p_target[np.isnan(p_target)],obs_pdraw[obs_pdraw<=0],obs_pdraw[np.isnan(obs_pdraw)])
            print(true_m1s[obs_pdraw<=0])
        idxs = np.random.choice(ndet,size=n_evs,replace=True,
                                p=p_ratio/np.sum(p_ratio))
        rwed_m1s = obs_m1s[idxs]
        rwed_dl = obs_dL[idxs]

        ## find maximum predicted mass and minimum distance
        m1max[i] = rwed_m1s.max()
        Dmin[i] = rwed_dl.min()
    
    m1max_thresh = np.quantile(m1max,1-outlier_thresh)
    Dmin_thresh = np.quantile(Dmin,outlier_thresh)

    return m1max_thresh, Dmin_thresh

def make_injections(rng, num_injections, mass_redshift_pop_param_dict, z_bounds, mass_bounds, mu_bounds):
    true_src_masses, det_masses, q, true_redshifts, true_distances, thetas, mass_z_pdraw = draw_true(mass_redshift_pop_param_dict,
                                                                                               num_injections,
                                                                                               rng,z_bounds=z_bounds,
                                                                                               mass_bounds=mass_bounds)
    logmags, logmu_pdraw = rng.uniform(np.log(mu_bounds[0]),np.log(mu_bounds[1]),size=num_injections), np.ones(num_injections)/np.diff(np.log(mu_bounds))
    mags = np.exp(logmags)
    
    # # sample from a power law in mu. This does mu^-2, which is like the tail of the target mag distribution. It starts at mu=(1-0.7)=0.3
    # shift = mu_bounds[0]-1
    # mags = pareto(1,loc=shift).rvs(size=num_injections)
    # mu_pdraw = pareto(1,loc=shift).pdf(mags)
    # logmu_pdraw = mu_pdraw * mags
    # # replace the ones that are above interpolation ranges
    # too_high = mags>mu_bounds[1]
    # n_too_high = too_high.sum()
    # while n_too_high>0:
    #     mags[too_high] = pareto(1,loc=shift).rvs(size=n_too_high)
    #     too_high = mags>mu_bounds[1]
    #     n_too_high = too_high.sum()

    inferred_distances, inferred_redshifts, inferred_src_masses = trans.inferred_quantities_from_lensing(mags, det_masses, true_distances)
    
    injections = dict(true_src_mass=true_src_masses,
                      det_mass=det_masses,
                      mass_ratio=q,
                      true_redshift=true_redshifts,
                      true_distance=true_distances,
                      sky_angle=thetas,
                      magnification=mags,
                      inferred_distance=inferred_distances,
                      inferred_redshift=inferred_redshifts,
                      inferred_src_mass=inferred_src_masses,
                      mass_z_pdraw=mass_z_pdraw,
                      lnmu_pdraw=logmu_pdraw,
                      num_injections=num_injections,
                      )
    return injections

def find_injections(rng, injections, m1max_thresh, Dmin_thresh, optimal_snr_interpolator, distance_unit, uncert_dict, cosmo_dict):
    # unpack injection dictionary
    q = injections['mass_ratio']
    mags = injections['magnification']
    inferred_redshifts = injections['inferred_redshift']
    inferred_src_masses = injections['inferred_src_mass']

    if True: # do this to ensure consistency with PPD calculation
        # detect events and get their observed parameters
        observed_detected, true_detected = generate_obs_from_true_list(inferred_src_masses,
                                                                    inferred_src_masses*q,
                                                                    inferred_redshifts,
                                                                    optimal_snr_interpolator,
                                                                    cosmo_dict,
                                                                    rng,
                                                                    distance_unit,
                                                                    uncert=uncert_dict
        )
        found = true_detected['detected_index']
        N_found = len(found)
        print(f"{N_found} found injections")
        ## unsure if we want to return anything from true_detected? Seems important for reweighting later,
        ## but as we are only reweighing based on magnifications, true masses and distances are probably 
        ## never going to be used
        
        # transform to useful parameters
        obs_det_masses_found, obs_det_mass2_found = m1m2_from_mceta(observed_detected['Mc_det_obs'],observed_detected['eta_obs'])
        obs_inferred_distances_found = observed_detected['t_obs'] * optimal_snr_interpolator(obs_det_masses_found, obs_det_mass2_found,grid=False)/observed_detected['rho_obs']
        obs_inferred_redshifts_found = trans.z_at_dl_approx(obs_inferred_distances_found)
        obs_inferred_src_masses_found = obs_det_masses_found / (1+ obs_inferred_redshifts_found)
        mags_found = mags[found]

    else:
        det_masses = injections['det_mass']
        thetas = injections['sky_angle']
        Ninj = injections['num_injections']
        inferred_distances = injections['inferred_distance']

        # detect events
        true_SNR = optimal_snr_interpolator(det_masses,det_masses*q,grid=False) * thetas / inferred_distances 
        obs_SNR = true_SNR + rng.normal(size=Ninj)
        found = obs_SNR > 8.
        N_found = np.sum(found)
        print(f"{N_found} found injections")
        true_det_masses_found = det_masses[found]
        true_q_found = q[found]
        true_thetas_found = thetas[found]
        mags_found = mags[found]
        obs_SNR_found = obs_SNR[found]

        # get their observed values
        mc = mchirp(true_det_masses_found, true_det_masses_found*true_q_found)
        eta = etafunc(true_det_masses_found, true_det_masses_found*true_q_found)
        smc = uncert_dict['threshold_snr']/obs_SNR_found*uncert_dict['mc']
        mcobs = rng.lognormal(mean=np.log(mc), sigma=smc)
        seta = uncert_dict['threshold_snr']/obs_SNR_found*uncert_dict['eta']
        etaobs = eta+seta*truncnorm.rvs((0.0-eta)/seta,(0.25-eta)/seta,size=N_found,random_state=rng)
        st = uncert_dict['threshold_snr']/obs_SNR_found*uncert_dict['Theta']
        tobs = true_thetas_found+st*truncnorm.rvs((0.0-true_thetas_found)/st, (1.0-true_thetas_found)/st, size=N_found,random_state=rng)

        # transform to useful parameters
        obs_det_masses_found, obs_det_mass2_found = m1m2_from_mceta(mcobs,etaobs)
        obs_inferred_distances_found = tobs * optimal_snr_interpolator(obs_det_masses_found, obs_det_mass2_found,grid=False)/obs_SNR_found
        obs_inferred_redshifts_found = trans.z_at_dl_approx(obs_inferred_distances_found)
        obs_inferred_src_masses_found = obs_det_masses_found / (1+ obs_inferred_redshifts_found)
    
    # save result
    obs_found_dict = dict(det_mass=obs_det_masses_found,
                          inferred_src_mass=obs_inferred_src_masses_found,
                          mass_ratio=obs_det_mass2_found/obs_det_masses_found,
                          inferred_redshift=obs_inferred_redshifts_found,
                          inferred_distance=obs_inferred_distances_found,
                          true_redshift=injections['true_redshift'][found],
                          magnification=mags_found,
                          lnmu_pdraw=injections['lnmu_pdraw'][found],
                          snr=observed_detected['rho_obs'],
                        # snr = obs_SNR_found,
                          num_found=N_found,
                          num_generated_total=injections['num_injections'],
                          found_bool=found,
    )

    # see which events are identified as outliers
    # identified = np.logical_or(obs_inferred_src_masses_found>m1max_thresh, obs_inferred_distances_found<Dmin_thresh)
    identified = obs_inferred_src_masses_found>m1max_thresh
    N_identified = np.sum(identified)
    print(f"{N_identified} injections found and identified as outliers")

    identified_dict = dict(det_mass=obs_det_masses_found[identified],
                           inferred_src_mass=obs_inferred_src_masses_found[identified],
                           magnification=mags_found[identified],
                           mass_ratio=obs_det_mass2_found[identified]/obs_det_masses_found[identified],
                           inferred_redshift=obs_inferred_redshifts_found[identified],
                           inferred_distance=obs_inferred_distances_found[identified],
                           snr=observed_detected['rho_obs'][identified],
                           Theta=observed_detected['t_obs'][identified],
                        # snr = obs_SNR_found[identified],
                        # Theta=tobs[identified],
                           identified_bool=identified,
    )                     

    return obs_found_dict, identified_dict

def make_Dai_interpolant(grid_filename, make_new=False, return_grids=False,
                         ln_mu_space=np.linspace(np.log(0.1),np.log(2500),num=5000),
                         z_space=np.linspace(0.001,10,num=100),
                         t_const_space=np.linspace(0,5,num=10)):
    if os.path.isfile(grid_filename) and not make_new:
        with h5py.File(grid_filename,'r') as f:
            ln_mu_space = f['ln_mu'][()]
            z_space = f['z'][()]
            t_const_space = f['additive_t_const'][()]
            pdf_grid = f['pdf_grid'][()]
    elif not make_new:
        raise ValueError("No grid file found. Check the filename.")
    elif make_new:
        pdf_grid = np.zeros((len(ln_mu_space),len(z_space),len(t_const_space)))

        #TODO: parallelize this
        for j in trange(len(z_space)):
            for k in range(len(t_const_space)):
                pdf_grid[:,j,k] = dists.logmag_pdf_t0const(ln_mu_space,z_space[j],t_const_space[k])
        
        # save grid. This should overwrite any previous file with the same name
        with h5py.File(grid_filename,'w') as f:
            f.create_dataset('ln_mu',data=ln_mu_space)
            f.create_dataset('z',data=z_space)
            f.create_dataset('additive_t_const',data=t_const_space)
            f.create_dataset('pdf_grid',data=pdf_grid)
            f.attrs['function'] = """def logmag_pdf_t0const(logmu,z,t0_const):
            sigma=np.interp(z,z_vals,sigma_vals)
            t0=np.interp(z,z_vals,t0_vals) + t0_const
            delta=np.interp(z,z_vals,delta_vals)
            lam=5
            
            A = 1./dists._invA(t0,lam)
            
            return np.trapz(y=np.exp(lam/(t_arr+t0)-2*t_arr)*np.exp(-(logmu - delta - t_arr)**2/(2*sigma**2)),
                            x=t_arr) * A / np.sqrt(2*np.pi) / sigma"""

    mu_interp = RegularGridInterpolator((ln_mu_space,z_space,t_const_space),pdf_grid)
    if return_grids:
        return mu_interp, ln_mu_space, z_space, t_const_space
    else:
        return mu_interp

def calc_Nexp(found_injections, identified_injections, mu_interp, t_const, rate_evo_index, local_rate=15, obs_time=1):
    Ninj = found_injections['num_generated_total']
    mags_found = found_injections['magnification']
    true_redshifts = found_injections['true_redshift']
    identified = identified_injections['identified_bool']
    found_weights = mu_interp((np.log(mags_found),true_redshifts,t_const)) / found_injections['lnmu_pdraw']
    identified_weights = found_weights[identified] 

    z_dummy = np.linspace(0.001,dists.ZMAX,num=100)
    total_rate = local_rate * np.trapz(x=z_dummy,y=dists.shouts_murmurs(
        z_dummy,alpha=rate_evo_index,beta=dists.BETA,zp=dists.ZPEAK))

    Nexp_lensed = identified_weights.sum() * total_rate * obs_time / Ninj # should we be multiplying this by OUTLIER_THRESH or something to account for the expected fraction of false positives?
    Nexp_traditional = found_weights.sum() * total_rate * obs_time / Ninj
    return Nexp_lensed, Nexp_traditional

def calc_binom_f(found_injections, identified_injections, mu_interp, t_const, rate_evo_index, local_rate=15, obs_time=1):
    numerator, denominator = calc_Nexp(found_injections, identified_injections, mu_interp, t_const, rate_evo_index, local_rate=local_rate, obs_time=obs_time)
    return numerator/denominator

def tau_from_t0c(muprime, z, t_const, mu_interp, dlogmu=0.01):
    lnmu_dummy = np.linspace(np.log(muprime),np.log(dists.MU_MAX),num=int(-(np.log(muprime)-np.log(dists.MU_MAX))/dlogmu))
    lnmu_for_norm = np.linspace(np.log(0.3),np.log(dists.MU_MAX),num=int(-(np.log(0.3)-np.log(dists.MU_MAX))/dlogmu))
    return np.trapz(x=lnmu_dummy,y=mu_interp((lnmu_dummy,z,t_const)))/np.trapz(x=lnmu_for_norm,y=mu_interp((lnmu_for_norm,z,t_const)))

def plot_likelihood(x_vals,rate_vs_x, xlabel=None,ylabel=None,ax=None,**plot_kwargs):
    from scipy import stats
    from matplotlib import pyplot as plt
    if ylabel is None:
        if xlabel is None:
            ylabel = r"$p(k|t_0^c)$"
        else:
            ylabel = r"$p(k|$"+ xlabel + r"$)$"
    if ax is None:
        fig, ax=plt.subplots()
    ax.plot(x_vals,stats.poisson(rate_vs_x).pmf(0),**plot_kwargs,label=f"k = {0} outliers")
    ax.plot(x_vals,stats.poisson(rate_vs_x).pmf(1),**plot_kwargs,label=f"k = {1} outlier")
    ax.plot(x_vals,stats.poisson(rate_vs_x).pmf(2),**plot_kwargs,label=f"k = {2} outliers")
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig

def plot_likelihood_binom(x_vals, binom_frac, n_trials, xlabel=None,ylabel=None,ax=None,**plot_kwargs):
    from scipy import stats
    from matplotlib import pyplot as plt
    if ylabel is None:
        if xlabel is None:
            ylabel = r"$p(k|t_0^c)$"
        else:
            ylabel = r"$p(k|$"+ xlabel + r"$)$"
    if ax is None:
        fig, ax=plt.subplots()
    ax.plot(x_vals,stats.binom(n_trials,binom_frac).pmf(0),**plot_kwargs,label=f"k = {0} outliers")
    ax.plot(x_vals,stats.binom(n_trials,binom_frac).pmf(1),**plot_kwargs,label=f"k = {1} outlier")
    ax.plot(x_vals,stats.binom(n_trials,binom_frac).pmf(2),**plot_kwargs,label=f"k = {2} outliers")
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig