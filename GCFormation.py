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
cols = ['Mass', 'Radius', 'Metallicity', 'Merger Time', 'Subhalo Mass', 'Galactocentric Radius']

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


def get_M_GCs(dm, r_disk, m1, r1):
    """Total mass in globular clusters formed during a merger.

    Parameters
    ----------
    dm : float
        Amount of gas inflow in Msun.
    r_disk : float
        Scale length of gas disk in kpc.
    m1 : float
        Mass of subhalo at time 1 in Msun.
    r1 : float
        Radius of subhalo at time 1 in kpc.
    
    Returns
    -------
    float
        Total mass formed in globular clusters in Msun. 
    """
    dt = get_merger_timescale(r_disk, m1, r1)
    mdot = dm / dt # units of Msun / Gyr
    
    eta = (m1 / (10**12))**(-1/3)
    sfr = mdot / (1 + eta) # in Msun / Gyr
    
    surface_area = np.pi * (r1**2) # in kpc^2
    surf_gas_dens = 1200 * np.sqrt((sfr / surface_area) / (10**10))
    mdot_gcs = sfr * 0.0021 / (1 + (3000/(5*surf_gas_dens))) # in Msun/Gyr
    
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

def get_GC_masses(M_max, M_min=10**5, size=None):
    """Randomly sample a specified number of globular cluster masses.

    Parameters
    ----------
    M_max
        Maximum of globular cluster mass sampling range in units of Msun.
    M_min
        Minimum of globular cluster mass sampling range in units of Msun. Default value is 10^5 Msun.
    size : int
        Number of masses to sample. Default value is None.
    
    Returns
    -------
    int or array
        If size=None, returns single globular cluster mass in units of Msun. If size is specified, returns an array of cluster masses.
    """
    r = np.random.random_sample(size)
    return M_min / (1 - r * (1 - M_min/M_max))

def get_GC_radii(M_cl, scatter=0.4):
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

def get_merger_time(z1, z2, size=None):
    """Sampling function for subhalo merger times.
    
    Parameters
    ----------
    z1
        Redshift at first timestep.
    z2 
        Redshift at second timestep.
    size
        Number of merger times to sample.
    
    Returns
    -------
    float
        Time of merger in Gyr. 
    """
    t1 = cosmo.age(z1).value
    t2 = cosmo.age(z2).value
    dt = np.abs(t2-t1)
    
    merge_time = np.fmin(t1, t2) + dt * np.random.random_sample()
    # merge_redshifts = z_at_value(cosmo.age, merge_times)
    # merge_redshifts = np.array([z.value for z in merge_redshifts])
    return merge_time


def process_merger(tree, primaryIndex, secondaryIndex, descIndex, gas_in_progenitors, M_min=10**5):
    """Create a population of globular clusters based on a single merger event.

    Parameters
    ----------
    tree
        Full merger tree of subhalo at z=0
    primaryIndex : int
        Index of first progenitor of merger event in tree
    secondaryIndex : int
        Index of secondary progenitor of merger event in tree
    gas_in_progenitors : float
        Total gas mass in all progenitors of a merger.
    descIndex : int
        Index of descendant (merger product) in tree

    Returns
    -------
    array
        Array of globular cluster data with each row corresponding to a cluster and the columns corresponding to ['Mass', 'Radius', 'Metallicity', 'Metallicity Bin', 'Formation Time', 'Subhalo Mass', 'Galactocentric Radius'].
    """
    
    # Grab descendent (post-merger) subhalo info
    descMass = tree['SubhaloMassInMaxRad'][descIndex] * (10**10) * cosmo.h #Msun
    descMass_gas = tree['SubhaloMassInMaxRadType'][descIndex][0] * (10**10) * cosmo.h #Msun
    descSnap = tree['SnapNum'][descIndex]
    descRedshift = snap_to_redshift[descSnap]
    descRadius = tree['SubhaloVmaxRad'][descIndex] / (1 + descRedshift) * cosmo.h #kpc

    # Grab first progenitor subhalo info (we assume all other progenitor subhalos fall into this one)
    primaryMass = tree['SubhaloMassInMaxRad'][primaryIndex] * (10**10) * cosmo.h #Msun
    primaryMass_gas = tree['SubhaloMassInMaxRadType'][primaryIndex][0] * (10**10) * cosmo.h #Msun
    primarySnap = tree['SnapNum'][primaryIndex]
    primaryRedshift = snap_to_redshift[primarySnap]
    primaryRadius = tree['SubhaloVmaxRad'][primaryIndex] / (1 + primaryRedshift) * cosmo.h #kpc
    primaryRadius_disk = 0.025 * primaryRadius #kpc

    # Merger timescale depends on first progenitor subhalo properties
    dt = get_merger_timescale(primaryMass_gas, primaryMass, primaryRadius)

    # Total change in mass (gas only) 
    dm_gas_total = descMass_gas - primaryMass_gas

    # Produce GCs independently for each merger
    secondaryMass_gas = tree['SubhaloMassInMaxRadType'][secondaryIndex][0] * (10**10) * cosmo.h #Msun
    secondaryMet = tree['SubhaloGasMetallicityMaxRad'][secondaryIndex]
    secondarySnap = tree['SnapNum'][secondaryIndex]
    secondaryRedshift = snap_to_redshift[secondarySnap]

    if dm_gas_total > 0:
        dm_merger = dm_gas_total * secondaryMass_gas / gas_in_progenitors
        merge_time = get_merger_time(secondaryRedshift, descRedshift)
        M_GCs = get_M_GCs(dm_merger, primaryRadius_disk, primaryMass, primaryRadius)
        if M_GCs > M_min:
            M_max = get_M_max(M_GCs)
            N_GCs = get_N_GCs(M_max, M_GCs, M_min=M_min)
            if N_GCs > 0:
                masses = get_GC_masses(M_max, size=N_GCs, M_min=M_min)
                radii = get_GC_radii(M_cl=masses)
                metallicities = np.full(shape=N_GCs, fill_value=secondaryMet) 
                merger_times = np.full(shape=N_GCs, fill_value=merge_time)
                subhalo_mass = np.full(shape=N_GCs, fill_value=descMass)
                gal_rad = np.full(shape=N_GCs, fill_value=descRadius)
            
                # TODO: update what info gets saved
                gcs_formed = np.column_stack((masses, radii, metallicities, merger_times, subhalo_mass, gal_rad))
                gcs_formed_df = pd.DataFrame(data=gcs_formed, columns=cols)
                return gcs_formed_df

    return None


def process_subhalo(basePath, tree, saveFile='globularclusters.h5'): # TODO: add assertions to check that the necessary properties are loaded for the tree
    """Create a population of globular clusters formed in a subhalo based on its merger history.

    Parameters
    ----------
    basePath : str
        Path to Illustris data.
    tree
        The full merger tree of the subhalo. 
    saveFile : str
        If a file name is given, the cluster data will be appended to the file. Must be an hdf5 (.h5) file.
    
    Returns
    -------
    Pandas dataframe
        Dataframe containing all globular clusters formed and their properties. Properties stored are ['Mass', 'Radius', 'Metallicity', 'Merger Time', 'Subhalo Mass', 'Galactocentric Radius', 'Formation Subhalo ID'].
    """
    # TODO: update docstring once columns are changed to reflect new saved values
    gcs_df = pd.DataFrame()

    hasNextProgenitor = tree['NextProgenitorID'] > 0
    subhalo2_index = tree['NextProgenitorID'][hasNextProgenitor] - tree['SubhaloID'][0]
    descendant_index = tree['DescendantID'][hasNextProgenitor] - tree['SubhaloID'][0]

    unique_desc = np.unique(descendant_index)

    for descendant in unique_desc:
        subhalo1 = tree['FirstProgenitorID'][descendant] - tree['SubhaloID'][0]
        subhalo2_list = subhalo2_index[descendant_index == descendant]

        # Need this for ratio of mass that comes from each infalling subhalo
        gas_in_progenitors = np.sum(tree['SubhaloMassInMaxRadType'][subhalo2_list, 0]) * (10**10) * cosmo.h #Msun
        if gas_in_progenitors > 0:
            for subhalo2 in subhalo2_list:
                merger_df = process_merger(tree, subhalo1, subhalo2, descendant, gas_in_progenitors)
                if merger_df is not None:
                    gcs_df = pd.concat([gcs_df, merger_df], ignore_index=True)

    gcs_df.to_hdf(saveFile, key='subhalo{}'.format(tree['SubfindID'][0]), append=True)
    return gcs_df