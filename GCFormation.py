import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from astropy import constants as const
from astropy.cosmology import LambdaCDM, z_at_value
from scipy.optimize import fsolve
import illustris_python as il
import requests

baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key":"ac9d38def7dd515283db64d231e4978f"}
cosmo = LambdaCDM(H0=67.74, Om0=0.3089, Ode0=0.6911, Ob0=0.0486)
# TODO: Need to update these columns to reflect all the additional information I want to record
cols = ['Mass', 'Radius', 'Metallicity', 'Merger Time', 'Formation Time' 'Subhalo Mass', 'Galactocentric Radius']

def get(path, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically
    return r

def get_snap_to_redshift(sim_name='TNG100-1'):
    """Create dictionary to map snap number to redshift.
    """
    r = get(baseUrl)
    names = [sim['name'] for sim in r['simulations']]
    i = names.index(sim_name)
    sim = get( r['simulations'][i]['url'] )
    snaps = get( sim['snapshots'] )
    snap_to_redshift = {snaps[i]['number']:snaps[i]['redshift'] for i in range(100)}
    # snap_times = [cosmo.age(snaps[i]['redshift']) for i in range(100)]
    return snap_to_redshift

snap_to_redshift = get_snap_to_redshift()

def mass_dist(M_0, M):
    """Distribution function of globular cluster masses. 
    """
    return M_0 * (M**(-2))

def max_mass_eqn(M_max, M_GC, M_min=10**5):
    """Equation used to find maximum mass for globular cluster distribution.
    """
    return M_max * np.log(M_max / M_min) - M_GC

def get_M_max(M_GCs, M_min=10**5):
    """Solves for the maximum possible globular cluster mass formed from a merger event.
    """
    return fsolve(max_mass_eqn, 10**6, args=(M_GCs, M_min))[0]


def get_merger_timescale(r_disk, m, r):
    """Merger timescale is roughly the dynamical time of the galaxy
    
    Parameters
    ----------
    r_disk : float
        Scale length of gas disk in kpc.
    m : float
        Mass of subhalo in Msun
    r : float
        Radius of subhalo in kpc

    Returns
    -------
    float
        Approximate merger timescale in Gyr
    """
    G = ((const.G).to(u.pc**3 / u.Msun / u.Gyr**2))
    m = m * u.Msun
    r = r * u.kpc
    r_disk = r_disk * u.kpc
    vel = np.sqrt(G * m / r)
    tau = (2 * r_disk / vel).to(u.Gyr)
    return tau.value

def get_formation_times(merge_time, merger_timescale, num_gcs):
    """Randomly sample a cluster formation time.

    Parameters
    ----------
    merge_time : float
        Time of subhalo merger in Gyr.
    merger_timescale : float
        Timescale of merger in Gyr.
    num_gcs : int
        Number of times to sample. 

    Returns
    -------
    array of floats
        Formation time for a specified number of globular clusters. Evenly sampled between (merge_time, merge_time + merger_timescale)
    """
    sample = np.random.sample(size=num_gcs)
    return merge_time + merger_timescale * sample

def get_GC_masses(M_min, M_max, size=None):
    """Randomly sample a specified number of globular cluster masses.

    Parameters
    ----------
    M_min
        Minimum of globular cluster mass sampling range in units of Msun. Default value is 10^5 Msun.
    M_max
        Maximum of globular cluster mass sampling range in units of Msun.
    size : int
        Number of masses to sample. Default value is None.
    
    Returns
    -------
    int or array
        If size=None, returns single globular cluster mass in units of Msun. If size is specified, returns an array of cluster masses.
    """
    r = np.random.random_sample(size)
    return M_min / (1 - r * (1 - M_min/M_max))


def get_GC_radii(M_cl, scatter):
    """Sampling function for globular cluster radius.
    
    Parameters
    ----------
    M_cl
        Mass of globular cluster in units of Msun.
    sigma
        Width of log-normal distribution centered around r_exp. Default is 0.4 dex.

    Returns
    -------
    int or array
        The radius of a globular cluster in pc, given its mass, including intrinsic random scatter.
    """
    
    M_cl = M_cl / 10000.
    r_exp = 1.4 * np.power(M_cl, 0.25) # in pc
    log_r = np.random.normal(np.log10(r_exp), scatter, size=len(r_exp))
    gc_radii = np.power(10, log_r)
    return gc_radii


def get_M_GCs(dm, dt, m1, r1, m_gas, alpha_eta, beta_eta, alpha_gamma, beta_gamma, sigma_crit, sigma_gmc_coeff):
    """Total mass in globular clusters formed during a merger.

    Parameters
    ----------
    dm : float
        Amount of gas inflow in Msun.
    dt : float
        Merger timescale in Gyr.
    m1 : float
        Mass of subhalo at time 1 in Msun.
    r1 : float
        Radius of subhalo at time 1 in kpc.
    m_gas : float
        Amount of gas in disk in Msun. Includes dm.
    
    Returns
    -------
    float
        Total mass formed in globular clusters in Msun. 
    """
    mdot = dm / dt # units of Msun / Gyr
    
    eta = alpha_eta * (m1 / (10**12))**(-beta_eta)
    sfr = mdot / (1 + eta) # in Msun / Gyr
    
    surface_area = np.pi * ((r1*1000)**2) # in pc^2
    surf_gas_dens = m_gas / surface_area # in Msun/pc^2
    mdot_gcs = sfr * alpha_gamma / (1 + ((sigma_crit/(sigma_gmc_coeff*surf_gas_dens))**-beta_gamma)) # in Msun/Gyr
    
    M_GC = mdot_gcs * dt
    return M_GC


def get_N_GCs(M_max, M_GCs, M_min=10**5):
    """The number of globular clusters formed given the total mass in globular clusters and a mass range.

    Parameters
    ----------
    M_max
        Maximum of globular cluster mass sampling range in units of Msun.
    M_GCs
        Total amount of mass in globular clusters in units of Msun.
    M_min
        Minimum of globular cluster mass sampling range in units of Msun. Default value is 10^5 Msun.

    Returns
    -------
    int
        Number of globular clusters expected to form, rounded to an integer.
    """
    if M_GCs <= M_min:
        return 0
    M_avg = np.log(M_max / M_min) / (1 / M_min - 1 / M_max)
    N_GCs = M_GCs / M_avg

    if N_GCs < 0.5:
        return 0
    
    rounding = np.random.random_sample()
    if rounding < N_GCs - np.trunc(N_GCs):
        return int(np.ceil(N_GCs))
    else:
        return int(np.floor(N_GCs)) 


def process_merger(dm, m1, r1, m_gas, merge_time, met, saveFile='globularclusters.h5', alpha_eta=1, beta_eta=1/3, alpha_gamma=0.0021, beta_gamma=1/3, sigma_crit=3000, sigma_gmc_coeff=5, M_min=10**5, scatter=0.4, disk_factor = 0.025): 
    """Create a population of globular clusters based on a given merger history.

    Parameters
    ----------
    merger: str
        HDF5 file with merger information.

    saveFile : str
        If a file name is given, the cluster data will be appended to the file. Must be an hdf5 (.h5) file.
    
    Returns
    -------
    Pandas dataframe
        Dataframe containing all globular clusters formed and their properties. Properties stored are [].
    """
    # TODO: update docstring once columns are changed to reflect new saved values
    
    # Get merger timescale
    dt = get_merger_timescale(r1*disk_factor, m1, r1)

    # Get M_GCs
    M_GCs = get_M_GCs(dm, dt, m1, r1, m_gas, alpha_eta=alpha_eta, beta_eta=beta_eta, alpha_gamma=alpha_gamma, beta_gamma=beta_gamma, sigma_crit=sigma_crit, sigma_gmc_coeff=sigma_gmc_coeff)

    # Get N_GCs
    M_max = get_M_max(M_GCs, M_min=M_min)
    N_GCs = get_N_GCs(M_max, M_GCs, M_min=M_min)

    # Sample stuff
    gc_masses = get_GC_masses(M_min, M_max, size=N_GCs)
    gc_radii = get_GC_radii(gc_masses, scatter=scatter)
    gc_times = get_formation_times(merge_time, dt ,size=N_GCs)
    gc_mets = np.full(N_GCs, met)
    
    gcs_formed = np.column_stack((gc_masses, gc_radii, gc_times, gc_mets))
    gcs_df = pd.DataFrame(data=gcs_formed, columns=cols)
    return gcs_df