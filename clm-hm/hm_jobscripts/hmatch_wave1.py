import numpy as np
import pandas as pd
import xarray as xr
import os, json
import sys
import tensorflow as tf
from scipy.stats import qmc
from hm_pyfunctions import *

# input to script
idx = sys.argv[1]

#############################################
# user mods
#############################################
n_universal_samples = 10000
n_pft_samples = 1000
n_pfts = 14

I_tolerance = 3.0 # implausibility tolerance

out_dir = '/glade/work/linnia/CLM6-PPE/ctsm53065_lhc/NROY/'

#############################################
# Setup
#############################################
# get config information
utils_dir = '/glade/u/home/linnia/CLM6-PPE/utils/'

with open(os.path.join(utils_dir, "param_info.json"), "r") as f:
    param_info = json.load(f)

with open(os.path.join(utils_dir, "biome_info.json"), "r") as f:
    biome_info = json.load(f)

universal_param_names = param_info['u_params']
pft_param_names = param_info['pft_params']
n_universal_params = len(param_info['u_params'])
n_pft_params = len(param_info['pft_params']) # Number of parameters per PFT

# load observational data
obs = xr.open_dataset('../obs_statistics_byBiome.nc')
#somc_obs = xr.open_dataset('../SOMC_statistics_PgC.nc')

#############################################
# stack up all your observational data and emulators for each input metric (variable and biome)
paths = {
    'LAI': '../emulators/biome_lai_',
    'GPP': '../emulators/biome_gpp_',
    'TVC': '../emulators/biome_tvc_',
    'LH' : '../emulators/biome_lh_'
}

biomes = [1,2,3,4,5,6,7,8,9,10,11,12,13]
obs_mean = []
obs_var = []
emulators = [] 
biome_list = [] 

var_list = ['LAI','GPP','TVC','LH']
for v in var_list:
    for b in biomes:
        biome_list.append(str(b))
    
        obs_mean.append(obs[v+'_mean'].isel(biome=b).values)
        obs_var.append(obs[v+'_stdev'].isel(biome=b).values**2)
    
        emulators.append(tf.saved_model.load(paths[v]+str(b)))

# SOMC
#biome_list.append(str(14))
#obs_mean.append(somc_obs['nonarctic_mean'].values)
#obs_var.append(somc_obs['nonarctic_stdev'].values**2)
#emulators.append(tf.saved_model.load('../emulators/somc_PgC_region0'))

#biome_list.append(str(15))
#obs_mean.append(somc_obs['arctic_mean'].values)
#obs_var.append(somc_obs['arctic_stdev'].values**2)
#emulators.append(tf.saved_model.load('../emulators/somc_PgC_region1'))

#############################################
# History Matching
#############################################
# create universal sample

sampler = qmc.LatinHypercube(d=n_universal_params)

s = sampler.random(n=n_universal_samples)
universal_parameter_samples = pd.DataFrame(s,columns=universal_param_names)

#############################################

all_nroy_param_sets_from_universal = [] # List to collect NROY sets from each universal sample

# Loop over each universal parameter sample
for u_idx in range(len(universal_parameter_samples)):
    # create sample (dataframe) for the current universal parameter set
    usample_values = np.tile(universal_parameter_samples.iloc[[u_idx]].values,(n_pft_samples,1))
    psample_values = np.random.rand(n_pft_samples,n_pfts*n_pft_params)
    combined_values = np.concatenate([usample_values,psample_values],axis=1)

    all_pft_param_column_names = []
    for pft_id in range(1, n_pfts + 1):
        for param_name in pft_param_names:
            all_pft_param_column_names.append(f"pft{pft_id}_{param_name}")
    all_columns = universal_param_names + all_pft_param_column_names
    sample_df = pd.DataFrame(combined_values, columns=all_columns)


    # History matching for the current universal parameter set
    num_metrics = len(emulators)
    all_implausibilities_for_samples = np.empty((n_pft_samples, num_metrics))

    for metric_ix in range(num_metrics):
        biome_id = str(biome_list[metric_ix]) # Ensure biome_id is an integer for dict lookup
        emulator_object = emulators[metric_ix] # access emulator from a list/array

        y_pred, y_pred_var = call_biome_emulator(sample_df, biome_id, emulator_object, biome_info)

        # calc_Implausibility 
        metric_I = calc_Implausibility(y_pred, y_pred_var, obs_mean[metric_ix], obs_var[metric_ix])
        
        # Store the implausibility values for the current biome/metric
        all_implausibilities_for_samples[:, metric_ix] = metric_I.squeeze()

    # identify parameter sets that are Not Ruled Out Yet for the current universal sample
    is_nroy_mask = (all_implausibilities_for_samples < I_tolerance).all(axis=1)
    current_u_nroy_param_sets = sample_df[is_nroy_mask]
    
    # Append the NROY sets found for this universal parameter to the master list
    all_nroy_param_sets_from_universal.append(current_u_nroy_param_sets)

# Concatenate all NROY_param_sets from all universal samples
NROY_param_sets = pd.concat(all_nroy_param_sets_from_universal).reset_index(drop=True)
out_file = out_dir+'NROY_param_sets_'+str(idx)+'.csv'
NROY_param_sets.to_csv(out_file,index=False)
all_nroy_param_sets_from_universal = []

