# CLM6-PPE
Community Land Model v6 - Perturbed Parameter Ensemble generation and calibration

Generate a latin hypercube ensemble
    1a. checkout codebase: bash ./gen_ensembles/jobscripts/step0.startup.sh
    1b. setup base cases: bash ./gen_ensembles/jobscripts/step1.setup_basecases.sh
    1c. generate paramfiles: ./gen_ensembles/gen_paramfiles/generate_paramfiles_lhc.ipynb
    1d. run ensemble: bash ./gen_ensembles/jobscripts/run_ens.sh ./gen_ensembles/jobscripts/lhc/lhc.config
    1e. post process (see ./postprocessing/README.txt)

History Matching
    2a. Train emulators: ./clm-hm/1.train_emulators.ipynb
    2b. Sample parameter space, emulate, calculate implausibility and save NROY parameter sets: ./clm-hm/2.history_matching.ipynb
    2c. Select parameter sets for the next wave: ./clm-hm/3.select_paramsets.ipynb
    2d. Run next ensemble:
        - generate paramfiles and namelist mods: ./gen_ensembles/gen_paramfiles/generate_paramfiles_waves.ipynb
        - run ensemble: bash ./gen_ensembles/jobscripts/run_ens.sh ./gen_ensembles/jobscripts/wave1/wave1.config
    2e. post process
    2f. Repeat!

Calibration
    3a. Train emulators: ./clm-hmc/1.train_emulators.ipynb
    3b. Perform Hamiltonian Monte Carlo sampling with emulators: ./clm-hmc/2.calibrate_HMC_batched.py
    3c. Evaluate convergence and sample from posterior: ./clm-hmc/3.eval_select_write.ipynb
    