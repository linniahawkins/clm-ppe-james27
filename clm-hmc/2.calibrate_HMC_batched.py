"""
HMC Calibration with Batched Chains for GPU Efficiency

This script runs Hamiltonian Monte Carlo sampling using TensorFlow Probability,
with multiple chains batched together to exploit GPU parallelism and emulator
batch efficiency.
"""

import numpy as np
import pandas as pd
import xarray as xr
import json
import os
import sys
import tensorflow as tf
import tensorflow_probability as tfp

# Ensure GPU memory growth (prevents OOM)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Using {len(gpus)} GPU(s)")

sys.path.append('/glade/u/home/linnia/CLM6-PPE/')

# =============================================================================
# Configuration
# =============================================================================

UTILS_PATH = '/glade/u/home/linnia/CLM6-PPE/utils/'
EMULATOR_PATHS = {
    'lai': '/glade/u/home/linnia/CLM6-PPE/clm6-calibration/emulators/biome_lai_',
    'gpp': '/glade/u/home/linnia/CLM6-PPE/clm6-calibration/emulators/biome_gpp_',
    'biomass': '/glade/u/home/linnia/CLM6-PPE/clm6-calibration/emulators/biome_tvc_',
    'lh': '/glade/u/home/linnia/CLM6-PPE/clm6-calibration/emulators/biome_lh_',
    'somc': '/glade/u/home/linnia/CLM6-PPE/clm6-calibration/emulators/somc_PgC_'
}
BIOMES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
N_PFTS = 14

# HMC settings
N_CHAINS = 30
NUM_RESULTS = 5000
NUM_BURNIN = 1000
NUM_LEAPFROG_STEPS = 30
STEP_SIZE_INIT = 0.01
TARGET_ACCEPT_PROB = 0.85
USE_NUTS=False
CHUNK_SIZE = 50  # Save a checkpoint every CHUNK_SIZE samples

OUTPUT_DIR = '/glade/u/home/linnia/CLM6-PPE/clm6-calibration/hmc_results_longrun/'
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'checkpoint.npz')

# =============================================================================
# Load metadata and observations
# =============================================================================

with open(os.path.join(UTILS_PATH, "param_info.json"), "r") as f:
    param_info = json.load(f)

with open(os.path.join(UTILS_PATH, "biome_info.json"), "r") as f:
    biome_info = json.load(f)

universal_param_names = param_info['u_params']
pft_param_names = param_info['pft_params']

def get_biome_param_names(biome_id):
    return biome_info[str(biome_id)]['biome_parameters']

# Load default parameters and reorder
default = pd.read_csv('default_params_norm.csv')

all_pft_param_column_names = []
for pft_id in [1,2,3,4,5,6,7,8,10,11,12,13,14]:
    for param_name in pft_param_names:
        all_pft_param_column_names.append(f"pft{pft_id}_{param_name}")
all_columns = universal_param_names + all_pft_param_column_names
default_params = default[all_columns]
x_default = tf.constant(default_params.iloc[[0]].to_numpy(), dtype=tf.float64)

n_params = len(all_columns)

# Load observations
obs_biome = xr.open_dataset('obs_statistics_byBiome.nc', engine='h5netcdf')
somc_obs = xr.open_dataset(UTILS_PATH + 'SOMC_statistics_PgC.nc', engine='h5netcdf')

# =============================================================================
# Load emulators and build target/index arrays
# =============================================================================
print("Loading emulators...", flush=True)

param_indices = []
targets = []
stdevs = []
emulators = []

for b in BIOMES:
    param_names = get_biome_param_names(b)
    param_ix = [default_params.columns.get_loc(p) for p in param_names]
    
    for metric, path_key in [('LAI', 'lai'), ('GPP', 'gpp'), ('TVC', 'biomass'), ('LH', 'lh'),]:
        param_indices.append(param_ix)
        targets.append(obs_biome[f'{metric}_mean'].isel(biome=b).values)
        stdevs.append(obs_biome[f'{metric}_stdev'].isel(biome=b).values)
        emulators.append(tf.saved_model.load(f"{EMULATOR_PATHS[path_key]}{b}"))

# SOMC
param_names = get_biome_param_names(14)
param_ix = [default_params.columns.get_loc(p) for p in param_names]
param_indices.append(param_ix)
targets.append(somc_obs['nonarctic_mean'].values)
stdevs.append(somc_obs['nonarctic_stdev'].values)
emulators.append(tf.saved_model.load(f"{EMULATOR_PATHS['somc']}{'non-arctic'}"))

param_names = get_biome_param_names(15)
param_ix = [default_params.columns.get_loc(p) for p in param_names]
param_indices.append(param_ix)
targets.append(somc_obs['arctic_mean'].values)
stdevs.append(somc_obs['arctic_stdev'].values)
emulators.append(tf.saved_model.load(f"{EMULATOR_PATHS['somc']}{'arctic'}"))

# Convert to tensors
targets_tf = tf.constant(np.array(targets), dtype=tf.float64)
stdevs_tf = tf.constant(np.array(stdevs), dtype=tf.float64)

# Compile emulator prediction functions
print("Compiling emulator functions...", flush=True)
emulator_fns = []
for emulator, indices in zip(emulators, param_indices):
    input_dim = len(indices)
    sig = tf.TensorSpec([None, input_dim], tf.float64)
    fn = tf.function(
        lambda x, m=emulator: m.compiled_predict_f(x)[0],
        input_signature=[sig]
    )
    emulator_fns.append(fn)

# =============================================================================
# Log probability function (batched across chains)
# =============================================================================

@tf.function
def unnormalized_log_prob(x):
    """
    Compute unnormalized log probability for batched chains.
    
    Args:
        x: Parameter tensor of shape [n_chains, n_params]
    
    Returns:
        Log probability tensor of shape [n_chains]
    """
    x = tf.cast(x, tf.float64)
    
    # Handle both single sample and batched input
    if len(x.shape) == 1:
        x = tf.expand_dims(x, 0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    n_chains = tf.shape(x)[0]
    log_likelihood = tf.zeros([n_chains], dtype=tf.float64)
    
    # Sum likelihood over all emulators
    for i, (emulator_fn, indices) in enumerate(zip(emulator_fns, param_indices)):
        # Extract relevant parameters: [n_chains, n_biome_params]
        x_biome = tf.gather(x, indices, axis=1)
        
        # Predict: [n_chains, 1] -> [n_chains]
        y_pred = tf.squeeze(emulator_fn(x_biome), axis=-1)
        
        # Gaussian log-likelihood per chain
        target_i = targets_tf[i]
        stdev_i = stdevs_tf[i]
        residual = (y_pred - target_i) / stdev_i
        log_lik_i = -0.5 * residual**2 - tf.math.log(stdev_i)
        
        log_likelihood = log_likelihood + log_lik_i
    
    # Soft bounds prior: penalize parameters outside [0, 1]
    bounds_penalty = -10000.0 * (
        tf.reduce_sum(tf.nn.relu(-x)**2, axis=1) +
        tf.reduce_sum(tf.nn.relu(x - 1.0)**2, axis=1)
    )
    
    log_prob = log_likelihood + bounds_penalty
    
    if squeeze_output:
        log_prob = tf.squeeze(log_prob, axis=0)
    
    return log_prob

# =============================================================================
# Kernel factory
# =============================================================================

def make_kernel(step_size, num_adaptation_steps):
    """Build an adaptive HMC/NUTS kernel. Set num_adaptation_steps=0 for fixed step size."""
    if USE_NUTS:
        inner = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=unnormalized_log_prob,
            step_size=step_size,
            max_tree_depth=6
        )
    else:
        inner = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_log_prob,
            num_leapfrog_steps=NUM_LEAPFROG_STEPS,
            step_size=step_size
        )
    return tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=inner,
        num_adaptation_steps=num_adaptation_steps,
        target_accept_prob=TARGET_ACCEPT_PROB
    )

def trace_fn(_, pkr):
    inner = pkr.inner_results
    result = {
        'step_size': pkr.new_step_size,
    }
    if USE_NUTS:
        result['target_log_prob'] = inner.target_log_prob
        result['leapfrogs_taken'] = inner.leapfrogs_taken
    else:
        result['target_log_prob'] = inner.accepted_results.target_log_prob
        result['is_accepted'] = inner.is_accepted
    return result

# =============================================================================
# Initialize or resume from checkpoint
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

if os.path.exists(CHECKPOINT_FILE):
    print(f"Resuming from checkpoint: {CHECKPOINT_FILE}", flush=True)
    ckpt = np.load(CHECKPOINT_FILE)
    current_state = tf.constant(ckpt['current_state'], dtype=tf.float64)
    current_step_size = float(ckpt['step_size'])
    all_samples = [ckpt['samples']]
    all_log_probs = [ckpt['log_probs']]
    all_accepted = [ckpt['is_accepted']] if (not USE_NUTS and 'is_accepted' in ckpt) else []
    burnin_done = True
    n_completed = int(ckpt['samples'].shape[0])
    print(f"  {n_completed}/{NUM_RESULTS} samples already completed, "
          f"step_size={current_step_size:.6f}", flush=True)
else:
    print("Starting fresh run", flush=True)
    np.random.seed(42)
    initial_positions = np.random.uniform(0.05, 0.95, size=(N_CHAINS, n_params))
    initial_positions = np.clip(initial_positions, 0.0, 1.0)
    current_state = tf.constant(initial_positions, dtype=tf.float64)
    current_step_size = STEP_SIZE_INIT
    all_samples = []
    all_log_probs = []
    all_accepted = []
    burnin_done = False
    n_completed = 0

print(f"State shape: {current_state.shape}")
print(f"Running {N_CHAINS} chains, {NUM_RESULTS} total samples, CHUNK_SIZE={CHUNK_SIZE}")

# =============================================================================
# Run sampling in chunks, saving a checkpoint after each
# =============================================================================

chunk_idx = 0
while n_completed < NUM_RESULTS:
    chunk = min(CHUNK_SIZE, NUM_RESULTS - n_completed)

    if not burnin_done:
        run_kernel = make_kernel(current_step_size, int(NUM_BURNIN * 0.8))
        n_burnin = NUM_BURNIN
        burnin_done = True
        print(f"\nChunk {chunk_idx}: burnin ({NUM_BURNIN} steps) + {chunk} samples...", flush=True)
    else:
        # num_adaptation_steps=0 means DualAveraging acts as a pass-through (fixed step size)
        run_kernel = make_kernel(current_step_size, 0)
        n_burnin = 0
        print(f"\nChunk {chunk_idx}: {chunk} samples "
              f"(step_size={current_step_size:.6f})...", flush=True)

    chunk_samples, chunk_trace = tfp.mcmc.sample_chain(
        num_results=chunk,
        num_burnin_steps=n_burnin,
        current_state=current_state,
        kernel=run_kernel,
        trace_fn=trace_fn,
        seed=42 + chunk_idx * 1000
    )

    # Advance state for next chunk
    current_state = chunk_samples[-1]
    step_size_val = chunk_trace['step_size'][-1]
    current_step_size = float(step_size_val.numpy()
                              if hasattr(step_size_val, 'numpy') else step_size_val)

    all_samples.append(chunk_samples.numpy())
    all_log_probs.append(chunk_trace['target_log_prob'].numpy())
    if not USE_NUTS:
        all_accepted.append(chunk_trace['is_accepted'].numpy())

    n_completed += chunk
    chunk_idx += 1

    # Save checkpoint
    ckpt_data = dict(
        current_state=current_state.numpy(),
        step_size=np.array(current_step_size),
        samples=np.concatenate(all_samples, axis=0),
        log_probs=np.concatenate(all_log_probs, axis=0),
    )
    if not USE_NUTS and all_accepted:
        ckpt_data['is_accepted'] = np.concatenate(all_accepted, axis=0)
    np.savez(CHECKPOINT_FILE, **ckpt_data)
    print(f"  Checkpoint saved: {n_completed}/{NUM_RESULTS} samples", flush=True)

print(f"\nSampling complete!", flush=True)

# Reconstruct combined arrays for diagnostics/saving
samples_np = np.concatenate(all_samples, axis=0)   # [num_results, n_chains, n_params]
log_probs_np = np.concatenate(all_log_probs, axis=0)
samples = tf.constant(samples_np)

print(f"Samples shape: {samples.shape}")

if not USE_NUTS and all_accepted:
    accepted_np = np.concatenate(all_accepted, axis=0)
    accept_rate = tf.reduce_mean(tf.cast(accepted_np, tf.float32))
    print(f"Acceptance rate: {accept_rate.numpy():.2%}")

# =============================================================================
# Diagnostics
# =============================================================================

# Effective sample size (per parameter, across chains)
ess_byChain = tfp.mcmc.effective_sample_size(samples)
ess = tf.reduce_sum(ess_byChain, axis=0)
print(f"\nEffective sample size (min/median/max): "
      f"{tf.reduce_min(ess).numpy():.0f} / "
      f"{np.median(ess.numpy()):.0f} / "
      f"{tf.reduce_max(ess).numpy():.0f}")

# R-hat (potential scale reduction factor)
rhat = tfp.mcmc.potential_scale_reduction(samples)
print(f"R-hat (min/median/max): "
      f"{tf.reduce_min(rhat).numpy():.4f} / "
      f"{np.median(rhat.numpy()):.4f} / "
      f"{tf.reduce_max(rhat).numpy():.4f}")

# Mean log probability
mean_log_prob = tf.reduce_mean(log_probs_np).numpy()
print(f"Mean log probability: {mean_log_prob:.2f}")

# =============================================================================
# Save results
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save samples: [num_results, n_chains, n_params] -> [n_chains * num_results, n_params]
samples_flat = samples.numpy().reshape(-1, n_params)
samples_df = pd.DataFrame(samples_flat, columns=all_columns)
samples_df.to_csv(os.path.join(OUTPUT_DIR, 'posterior_samples.csv'), index=False)

# Save diagnostics
diagnostics = {
    'ess': ess.numpy(),
    'rhat': rhat.numpy(),
    'mean_log_prob': mean_log_prob,
    'n_chains': N_CHAINS,
    'num_results': NUM_RESULTS,
    'num_burnin': NUM_BURNIN,
}
np.savez(os.path.join(OUTPUT_DIR, 'diagnostics.npz'), **diagnostics)

print(f"\nResults saved to {OUTPUT_DIR}")
print(f"Total posterior samples: {samples_flat.shape[0]}")
