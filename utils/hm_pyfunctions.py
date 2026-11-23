import os
import numpy as np
import xarray as xr
import cftime
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import glob
import dask
import pickle

import gpflow
import tensorflow as tf
from sklearn.metrics import r2_score


utils_dir = os.path.dirname(__file__)

# ==============================================================
# ===================== Emulation ==============================

def build_kernel_dict(num_params):
    kernel_noise = gpflow.kernels.White(variance=1e-3)
    kernel_matern32 = gpflow.kernels.Matern32(active_dims=range(num_params), variance=10, lengthscales = np.tile(10,num_params))
    kernel_matern52 = gpflow.kernels.Matern52(active_dims=range(num_params),variance=1,lengthscales=np.tile(1,num_params))
    kernel_bias = gpflow.kernels.Bias(active_dims = range(num_params))
    kernel_linear = gpflow.kernels.Linear(active_dims=range(num_params),variance=[1.]*num_params)
    kernel_poly = gpflow.kernels.Polynomial(active_dims = range(num_params),variance=[1.]*num_params)
    kernel_RBF = gpflow.kernels.RBF(active_dims = range(num_params), lengthscales=np.tile(1,num_params))
    
    kernel_dict = {
        0: kernel_linear + kernel_noise,
        1: kernel_RBF * kernel_linear + kernel_noise,
        2:kernel_matern32 + kernel_noise,
        3:kernel_matern32 + kernel_linear + kernel_noise,
        4:kernel_linear*kernel_RBF + kernel_matern52 + kernel_noise
    }
    return kernel_dict

def select_kernel(kernel_dict, X_train, X_test, y_train, y_test):
    scores = []
    for k, kernel in kernel_dict.items():
        emulator = train_model(X_train, y_train, kernel)
        y_pred, sd, current_r2 = validate_model(emulator, X_test, y_test)
        
        # How many points fall within the 95% confidence interval?
        lower = y_pred - 1.96 * sd
        upper = y_pred + 1.96 * sd
        within_interval = np.mean((y_test >= lower) & (y_test <= upper))
        uncertainty_error = abs(within_interval - 0.95)
        
        # Combined Score: Maximize R2, Minimize uncertainty Error
        combined_score = current_r2 - uncertainty_error
        scores.append(combined_score)

    best_kernel = kernel_dict[np.argmax(scores)]
    return best_kernel

def train_model(X_train, y_train, kernel):
    model = gpflow.models.GPR(data=(X_train, np.float64(y_train)), kernel=kernel, mean_function=None)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=30))
    return model

def validate_model(model, X_test, y_test):
    y_pred, y_pred_var = model.predict_y(X_test)
    sd = y_pred_var.numpy().flatten()**0.5
    coef_deter = r2_score(y_test, y_pred.numpy())
    return y_pred, sd, coef_deter

def save_model(model, savedir, num_params):
    model.compiled_predict_f = tf.function(
        lambda X: model.predict_f(X, full_cov=False),
        input_signature=[tf.TensorSpec([None, num_params], tf.float64)],
    )
    tf.saved_model.save(model, savedir)

def plot_emulator_val(y_test, y_pred, sd, coef_deter, outfile):
    plt.figure()
    plt.errorbar(y_test, y_pred.numpy().flatten(), yerr=2*sd, fmt="o")
    plt.text(0.02, 0.98, f'R² = {np.round(coef_deter, 2)}',fontsize=10,transform=plt.gca().transAxes,va='top',ha='left')
    plt.text(0.02, 0.93, f'Emulator stdev ≈ {np.round(np.mean(sd), 2)}',fontsize=10,transform=plt.gca().transAxes,va='top',ha='left')
    plt.plot([0,np.max(y_test)],[0,np.max(y_test)],linestyle='--',c='k')
    plt.xlabel('CLM')
    plt.ylabel('Emulated')
    plt.xlim([np.min(y_test),np.max(y_test)])
    plt.ylim([np.min(y_test),np.max(y_test)])
    plt.tight_layout()
    plt.savefig(outfile)


def build_biome_X(ds_params, u_params, pft_params, biome_pfts):
    # get universal parameters
    u = ds_params.sel(param=u_params, pft=1).values
    X_arrays = [u]

    # Add each pft param for this biome
    for pft in biome_pfts:
        p = ds_params.sel(param=pft_params, pft=pft).values
        X_arrays.append(p)

    X = np.concatenate(X_arrays, axis=1)

    return X


def call_biome_emulator(sample_df, biome_id, emulator_object, biome_info_dict):

    # Subset the sample DataFrame by selecting columns by name
    biome_params = biome_info_dict[biome_id]['biome_parameters']
    biome_specific_sample_df = sample_df[biome_params]

    # Call the emulator and predict the sample
    y_pred, y_pred_var = emulator_object.compiled_predict_f(biome_specific_sample_df.values)

    return y_pred, y_pred_var


def call_biome_emulator_tf(params_tensor, biome_id, emulator_object, biome_info_dict, biome_params):
    """
    TensorFlow-native version of call_biome_emulator.
    Accepts a 1D tensor of parameters and returns predictions that support gradients.
    
    Args:
        params_tensor: 1D tf.Tensor of shape (n_params,) with float32 dtype
        biome_id: Biome ID string
        emulator_object: Loaded emulator SavedModel
        biome_info_dict: Biome information dictionary
        biome_params: List of parameter names in the correct order
    
    Returns:
        y_pred: tf.Tensor prediction value
        y_pred_var: tf.Tensor prediction variance
    """
    import tensorflow as tf
    
    # Ensure input is a tensor and has correct shape
    params_tensor = tf.cast(params_tensor, tf.float64)
    
    # Reshape to 2D if needed (emulator expects batch dimension)
    if len(params_tensor.shape) == 1:
        params_tensor = tf.reshape(params_tensor, [1, -1])
    
    # Call the emulator with the tensor directly
    y_pred, y_pred_var = emulator_object.compiled_predict_f(params_tensor)
    
    # Flatten to scalars (assuming single sample)
    y_pred = tf.reshape(y_pred, [])
    y_pred_var = tf.reshape(y_pred_var, [])
    
    return y_pred, y_pred_var
    
    
# ===================================================================
# ====================    History Matching   ========================

def calc_Implausibility(model_mean,model_var,obs_mean,obs_var):
        # implausibility score
        I = np.abs(obs_mean-model_mean) / np.sqrt(obs_var + model_var)
        return I


def sample_paramset_indices(df, universal_param_names, n):
    # choose n unique global parameter sets
    # pick one random PFT row for each
    
    global_unique = df[universal_param_names].drop_duplicates()
    chosen_global = global_unique.sample(n)

    selected_ix = []
    for _, g in chosen_global.iterrows():
        # find rows matching this global parameter set
        mask = (df[universal_param_names] == g.values).all(axis=1)
        # choose one random row belonging to that global set
        selected_ix.append(df[mask].sample(1).index[0])

    return np.array(selected_ix)


def sample_multiple_draws(df, universal_param_names, N, n):
    """
    Returns draws_ix of shape (N, n), where each row
    is one set of indices from sample_paramset_indices.
    """
    return np.vstack([
        sample_paramset_indices(df, universal_param_names, n)
        for _ in range(N)
    ])


def subset_NROY(df, N, n, universal_param_names):

    num_params = np.shape(df)[1]
    # take N random samples of size n
    draws_ix = sample_multiple_draws(df, universal_param_names, N, n)

    # Compute scores for each sampled subset
    nbins = 20 # for LHC
    L = np.zeros(N) 
    for i in range(N):
        s = df.iloc[draws_ix[i, :], :].values
        L[i] = LHC_score(n, nbins, num_params, s)
    
    # Find the best subset (minimum score)
    min_score_ix = np.argmin(L)
    
    return draws_ix[min_score_ix, :]
    

def LHC_score(n,nbins,num_params,sample):
    # sample is np array with rows as ensemble members and columns as parameters
    # zero is a perfect Latin Hypercube
    Pb = n/nbins
    dim_count = []
    for di in range(num_params):
        data = sample[:,di]
        bin_count = []
        for bi in range(nbins):
            bin_width = 1/20
            bin_min = bi/nbins
            bin_max = bi/nbins+bin_width
            Ab = np.sum((data>bin_min)&(data<bin_max))
    
            bin_count.append(np.abs(Pb - Ab))
        dim_count.append(np.sum(bin_count))
    
    return np.sum(dim_count)


def write_paramset_ds(default_ds, param_ranges_file, paramset_csv, pft_param_names, outfile):
    """
    Create and save a NetCDF file of normalized parameter settings for ensemble.

    Parameters:
    -----------
    default_ds : xarray dataset
        dataset default parameter values (currently unused).
    param_ranges_file : str
        Path to the CSV file with parameter names and inclusion flags.
    paramset_csv : str
        Path to the CSV file containing parameter values for each ensemble member.
    outfile : str
        Output NetCDF file path (should end in .nc).
    """

    # Select parameters we are perturbing only
    df = pd.read_csv(param_ranges_file)
    param_names = df[df['include'] == 1]['param'].values

    # Read ensemble values
    paramsets = pd.read_csv(paramset_csv)

    # Initialize 
    pft_ids = [1,2,3,4,5,6,7,8,10,11,12,13,14]
    extra_pfts = [0,9]
    c3_crop_pft = 15
    c3_grass_pft = 13
    
    data_vars = {}
    n_ens = len(paramsets)
    n_pfts = 79

    for param in param_names:
        if param in pft_param_names:
            default_vals = default_ds[param].values
            if len(default_vals) < n_pfts:
                padded_defaults = np.full(n_pfts, 0.5)
                padded_defaults[:len(default_vals)] = default_vals
            else:
                padded_defaults = default_vals
                
            arr = np.tile(padded_defaults, (n_ens, 1))

            # overwrite with the calibrated values
            cols = [f'pft{i}_{param}' for i in pft_ids]
            arr[:, pft_ids] = paramsets[cols].values

            # Patch extra PFTs
            for p in extra_pfts:
                arr[:, p] = np.tile(default_ds[param].values[p], (n_ens))
            arr[:, c3_crop_pft] = arr[:, c3_grass_pft]

            data_vars[param] = (["ens", "pft"], arr)
        else:
            arr = paramsets[param].values
            data_vars[param] = (["ens"], arr)

    ds_out = xr.Dataset(
        data_vars,
        coords={
            "ens": np.arange(n_ens),
            "pft": np.arange(n_pfts)
        }
    )

    ds_out.to_netcdf(outfile)
