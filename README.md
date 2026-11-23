# CLM6-PPE

Community Land Model v6 – Perturbed Parameter Ensemble generation and calibration.

## 1. Generate a Latin hypercube ensemble

1. **Check out the codebase:**
   ```bash
   bash ./gen_ensembles/jobscripts/step0.startup.sh
   ```
2. **Set up base cases:**
   ```bash
   bash ./gen_ensembles/jobscripts/step1.setup_basecases.sh
   ```
3. **Generate paramfiles:** [`gen_ensembles/gen_paramfiles/generate_paramfiles_lhc.ipynb`](gen_ensembles/gen_paramfiles/generate_paramfiles_lhc.ipynb)
4. **Run the ensemble:**
   ```bash
   bash ./gen_ensembles/jobscripts/run_ens.sh ./gen_ensembles/jobscripts/lhc/lhc.config
   ```
5. **Post-process:** see [`postprocessing/README.txt`](postprocessing/README.txt)

## 2. History matching

1. **Train emulators:** [`clm-hm/1.train_emulators.ipynb`](clm-hm/1.train_emulators.ipynb)
2. **Sample parameter space, emulate, calculate implausibility, and save NROY parameter sets:** [`clm-hm/2.history_matching.ipynb`](clm-hm/2.history_matching.ipynb)
3. **Select parameter sets for the next wave:** [`clm-hm/3.select_paramsets.ipynb`](clm-hm/3.select_paramsets.ipynb)
4. **Run the next ensemble:**
   - Generate paramfiles and namelist mods: [`gen_ensembles/gen_paramfiles/generate_paramfiles_waves.ipynb`](gen_ensembles/gen_paramfiles/generate_paramfiles_waves.ipynb)
   - Run the ensemble (one config per wave, e.g. [`wave4/wave4.config`](gen_ensembles/jobscripts/wave4/wave4.config)):
     ```bash
     bash ./gen_ensembles/jobscripts/run_ens.sh ./gen_ensembles/jobscripts/waveN/waveN.config
     ```
5. **Post-process**
6. **Repeat!**

## 3. Calibration

1. **Train emulators:** [`clm-hmc/1.train_emulators.ipynb`](clm-hmc/1.train_emulators.ipynb)
2. **Perform Hamiltonian Monte Carlo sampling with emulators:** [`clm-hmc/2.calibrate_HMC_batched.py`](clm-hmc/2.calibrate_HMC_batched.py)
3. **Evaluate convergence and sample from the posterior:** [`clm-hmc/3.eval_select_write.ipynb`](clm-hmc/3.eval_select_write.ipynb)
