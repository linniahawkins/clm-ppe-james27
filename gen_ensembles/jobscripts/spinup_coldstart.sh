#check out codebase
# run this on derecho only once: 
cd /glade/work/linnia/
git clone https://github.com/ESCOMP/CTSM.git CTSM
cd CTSM
./bin/git-fleximod update

git fetch --all --tags
git checkout ctsm5.3.065
./bin/git-fleximod update

chargenum='P93300041'

# Setup AD case:
cd /glade/work/linnia/CTSM/cime/scripts
./create_newcase --case ./cases/ctsm53065_cal000_cold_AD --compset I1850Clm60BgcCrujra --res f19_g17 --project P93300041 --run-unsupported

cd /glade/work/linnia/CTSM/cime/scripts/cases/ctsm53065_cal000_cold_AD/
./case.setup

# namelist mods:
echo "fsurdat = '/glade/derecho/scratch/wwieder/mksrf_5.4/surfdata_1.9x2.5_hist_1850_78pfts_c250429.nc'" >> user_nl_clm
echo "paramfile = '/glade/work/linnia/CLM6-PPE/ctsm6_cal/paramfiles/caltest0000.nc'" >> user_nl_clm
echo "snow_thermal_cond_glc_method = 'Jordan1991'" >> user_nl_clm
echo "snow_thermal_cond_lake_method = 'Sturm1997'" >> user_nl_clm

echo "hist_fincl1 = 'TOTECOSYSC', 'TOTECOSYSN', 'TOTSOMC', 'TOTSOMN', 'TOTVEGC', 'TOTVEGN', 'TLAI', 'GPP', 'CPOOL', 'NPP', 'TWS','H2OSNO','NFIX'" >> user_nl_clm
echo "do_grossunrep=.false." >> user_nl_clm
echo "reseed_dead_plants=.true." >> user_nl_clm

# xml changes
./xmlchange MOSART_MODE=NULL
./xmlchange STOP_N=40,STOP_OPTION=nyears
./xmlchange CLM_ACCELERATED_SPINUP=on,CLM_FORCE_COLDSTART=on
./xmlchange CONTINUE_RUN=FALSE,RESUBMIT=8
./xmlchange DOUT_S=FALSE

# Generate namelists
./preview_namelists
# Build case
./case.build
# Submit case
#./case.submit

#########################################
# Setup SASU case: 
./create_clone --case ./cases/ctsm53065_cal000_cold_SASU --clone ./cases/ctsm53065_cal000_cold_AD

cd /glade/work/linnia/CTSM/cime/scripts/cases/ctsm53065_cal000_cold_SASU/
./case.setup

./xmlchange MOSART_MODE=NULL
./xmlchange RUN_TYPE=hybrid
./xmlchange STOP_N=40,STOP_OPTION=nyears
./xmlchange CLM_ACCELERATED_SPINUP=sasu,CLM_FORCE_COLDSTART=off
./xmlchange CONTINUE_RUN=FALSE,RESUBMIT=7
./xmlchange GET_REFCASE=TRUE
./xmlchange RUN_REFCASE=ctsm53065_cal000_cold_AD
./xmlchange RUN_REFDATE=0361-01-01
./xmlchange DRV_RESTART_POINTER=rpointer.cpl.0361-01-01-00000
./xmlchange RUN_REFDIR=/glade/derecho/scratch/linnia/ctsm53065_cal000_cold_AD/run/
./xmlchange DOUT_S=FALSE

#./preview_namelists
#./case.build
#./case.submit

############################################################
# Setup pSASU case: 
./create_clone --case ./cases/ctsm53065_cal000_cold_pSASU --clone ./cases/ctsm53065_cal000_cold_SASU

cd /glade/work/linnia/CTSM/cime/scripts/cases/ctsm53065_cal000_cold_pSASU/
./case.setup

./xmlchange MOSART_MODE=NULL
./xmlchange STOP_N=40,STOP_OPTION=nyears
./xmlchange CLM_ACCELERATED_SPINUP=off,CLM_FORCE_COLDSTART=off
./xmlchange CONTINUE_RUN=FALSE,RESUBMIT=3
./xmlchange RUN_TYPE=hybrid
./xmlchange GET_REFCASE=TRUE
./xmlchange RUN_REFDATE=0321-01-01
./xmlchange DRV_RESTART_POINTER=rpointer.cpl.0321-01-01-00000
./xmlchange RUN_REFCASE=ctsm53065_cal000_cold_SASU
./xmlchange RUN_REFDIR=/glade/derecho/scratch/linnia/ctsm53065_cal000_cold_SASU/run/
./xmlchange DOUT_S=FALSE

#./preview_namelists
#./case.build
#./case.submit

############################################################
# Setup hist
cd /glade/work/linnia/CTSM/cime/scripts
./create_newcase --case ./cases/ctsm53065_cal094_cold_HIST --compset HIST_DATM%CRUv7_CLM60%BGC_SICE_SOCN_SROF_SGLC_SWAV_SESP --res f19_g17 --project P93300041 --run-unsupported

cd /glade/work/linnia/CTSM/cime/scripts/cases/ctsm53065_cal094_cold_HIST/
./case.setup

# xml changes 1850-1871 
./xmlchange RUN_TYPE=startup
./xmlchange RUN_STARTDATE=1850-01-01
./xmlchange STOP_OPTION=nyears
./xmlchange STOP_N=21
./xmlchange CONTINUE_RUN=FALSE
./xmlchange RESUBMIT=0
./xmlchange DATM_YR_ALIGN=1901
./xmlchange DATM_YR_START=1901
./xmlchange DATM_YR_END=1920
./xmlchange DOUT_S=FALSE

echo "paramfile = '/glade/derecho/scratch/wwieder/param_mods/cal0094.nc'" >> user_nl_clm
echo "finidat = '/glade/derecho/scratch/linnia/ctsm53065_cal094_cold_pSASU/ctsm53065_cal094_cold_pSASU.clm2.r.0161-01-01-00000.nc'">> user_nl_clm

echo "flanduse_timeseries = '/glade/derecho/scratch/wwieder/mksrf_5.4/landuse.timeseries_1.9x2.5_hist_1850-2023_78pfts_c250429.nc'" >> user_nl_clm
echo "fsurdat = '/glade/derecho/scratch/wwieder/mksrf_5.4/surfdata_1.9x2.5_hist_1850_78pfts_c250429.nc'" >> user_nl_clm

echo "snow_thermal_cond_glc_method = 'Jordan1991'" >> user_nl_clm
echo "snow_thermal_cond_lake_method = 'Sturm1997'" >> user_nl_clm
echo "use_c13 = .false." >> user_nl_clm
echo "use_c14 = .false." >> user_nl_clm

echo "do_grossunrep=.false." >> user_nl_clm
echo "reseed_dead_plants = .false." >> user_nl_clm
echo "use_init_interp = .true." >> user_nl_clm
echo "init_interp_fill_missing_urban_with_HD = .true." >> user_nl_clm

echo "ndep_taxmode = 'extend'" >> user_nl_clm
echo "calc_human_stress_indices = 'NONE'" >> user_nl_clm

#! "Survivability" output
#hist_dov2xy(1) = .true.
#hist_dov2xy(2) = .false.
#hist_fields_list_file = .false.
#hist_fincl2 = 'TLAI', 'TSA', 'TREFMNAV', 'TREFMXAV', 'BTRANMN', 'Vcmx25Z', 'FSH', 'VEGWP', 'FCTR', 'FCEV', 'FGEV',
#        'FIRE', 'FSR', 'FIRA', 'FSA', 'GSSUNLN', 'GSSHALN', 'TSKIN', 'GPP', 'NPP', 'AGNPP', 'TOTVEGC',
#        'NPP_NUPTAKE', 'AR', 'HR', 'HTOP'
#hist_mfilt(1) = 1
#hist_mfilt(2) = 1
#hist_nhtfrq(1) = 0
#hist_nhtfrq(2) = 0
#hist_type1d_pertape(1) = ' '
#hist_type1d_pertape(2) = ' '



