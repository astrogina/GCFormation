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
cols = ['mass1', 'gas_mass1', 'radius1', 'met1', 'subhalo_id1', 'snap1', 'mass2', 'gas_mass2', 'radius2', 'met2', 'subhalo_id2', 'snap2', 'dm_gas', 'gas_in_progenitors', 'desc_id', 'desc_snap', 'merge_time']

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

# TODO: add assertions to check that the necessary properties are loaded for the tree
def get_mergers_in_tree(basePath, tree): 
    """Record information about all mergers within a single merger tree.

    Parameters
    ----------
    basePath : str
        Path to Illustris data.
    tree
        The full merger tree of the subhalo. 
    
    Returns
    -------
    array
        Array of merger data with each row corresponding to a merger event and the columns corresponding to ['mass1', 'gas_mass1', 'radius1', 'met1', 'subhalo_id1', 'snap1', 'mass2', 'gas_mass2', 'radius2', 'met2', 'subhalo_id2', 'snap2', 'dm_gas', 'gas_in_progenitors', 'desc_id', 'desc_snap', 'merge_time'].
    """
    # TODO: update docstring once columns are changed to reflect new saved values
    mergers_df = pd.DataFrame()

    hasNextProgenitor = tree['NextProgenitorID'] > 0
    subhalo2_index = tree['NextProgenitorID'][hasNextProgenitor] - tree['SubhaloID'][0]
    descendant_index = tree['DescendantID'][hasNextProgenitor] - tree['SubhaloID'][0]
    
    # Grab info about all descendent subhalos
    descID = tree['SubhaloID'][descendant_index]
    descMass = tree['SubhaloMassInMaxRad'][descendant_index] * (10**10) * cosmo.h #Msun
    descMass_gas = tree['SubhaloMassInMaxRadType'][descendant_index][0] * (10**10) * cosmo.h #Msun
    descSnap = tree['SnapNum'][descendant_index]
    descRedshift = snap_to_redshift[descSnap] # Need to convert snap_to_redshift to an array
    descRadius = tree['SubhaloVmaxRad'][descendant_index] / (1 + descRedshift) * cosmo.h #kpc
    
    # Grab info about all subhalo1 subhalos, we assume all corresponding subhalo2s fall into this one
    subhalo1_index = tree['FirstProgenitorID'][descendant_index] - tree['SubhaloID'][0]
    primaryID = tree['SubhaloID'][subhalo1_index]
    primaryMass = tree['SubhaloMassInMaxRad'][subhalo1_index] * (10**10) * cosmo.h #Msun
    primaryMass_gas = tree['SubhaloMassInMaxRadType'][subhalo1_index][0] * (10**10) * cosmo.h #Msun
    primaryMet = tree['SubhaloGasMetallicityMaxRad'][subhalo1_index]
    primarySnap = tree['SnapNum'][subhalo1_index]
    primaryRedshift = snap_to_redshift[primarySnap]
    primaryRadius = tree['SubhaloVmaxRad'][subhalo1_index] / (1 + primaryRedshift) * cosmo.h #kpc

    # Total change in mass (gas only) 
    dm_gas_total = descMass_gas - primaryMass_gas

    # Grab info about all subhalo2 subhalos
    secondaryID = tree['SubhaloID'][subhalo2_index]
    secondaryMass = tree['SubhaloMassInMaxRad'][subhalo2_index] * (10**10) * cosmo.h #Msun
    secondaryMass_gas = tree['SubhaloMassInMaxRadType'][subhalo2_index][0] * (10**10) * cosmo.h #Msun
    secondaryMet = tree['SubhaloGasMetallicityMaxRad'][subhalo2_index]
    secondarySnap = tree['SnapNum'][subhalo2_index]
    secondaryRedshift = snap_to_redshift[secondarySnap]
    secondaryRadius = tree['SubhaloVmaxRad'][subhalo2_index] / (1 + secondaryRedshift) * cosmo.h #kpc

    merge_time = get_merger_time(min(primaryRedshift, secondaryRedshift), descRedshift)
    
    # Only save the ones that have nonzero gas mass -- those are the only ones that can form new GCs
    unique_desc = np.unique(descendant_index)
    mask = np.full(len(descendant_index), False)
    gas_in_prog = np.zeros(len(descendant_index))

    for descendant in unique_desc:
        # Need this for ratio of mass that comes from each infalling subhalo
        subhalo2_list = subhalo2_index[descendant_index == descendant]
        gas_in_progenitors = np.sum(tree['SubhaloMassInMaxRadType'][subhalo2_list, 0]) * (10**10) * cosmo.h #Msun
        
        if gas_in_progenitors > 0:
            mask[descendant_index == descendant] = True
            gas_in_prog[descendant_index == descendant] = gas_in_progenitors
    
    merger_info_array = np.column_stack((primaryMass[mask], primaryMass_gas[mask], primaryRadius[mask], primaryMet[mask], primaryID[mask], primarySnap[mask], secondaryMass[mask], secondaryMass_gas[mask], secondaryRadius[mask], secondaryMet[mask], secondaryID[mask], secondarySnap[mask], dm_gas_total[mask], gas_in_prog[mask], descID[mask], descSnap[mask], merge_time[mask]))
    mergers_df = pd.DataFrame(data=merger_info_array, columns=cols)

    return mergers_df