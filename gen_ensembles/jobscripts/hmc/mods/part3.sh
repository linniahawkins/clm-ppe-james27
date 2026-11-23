#!/bin/bash
prevcase=$1 
CASENAME=$(basename $2)
scratch='/glade/derecho/scratch/linnia/'

#########################################################################################
# PART 3
#########################################################################################
#
#    This script changes the datastreams, surfdata & land use timeseries to SSP3-70
#    and then runs the model for 2015-2023 as a branch run
#
#########################################################################################

# --- Ensure that the env_run.xml file has the correct content
./xmlchange RUN_TYPE=hybrid
#./xmlchange RUN_REFCASE=$CASENAME
./xmlchange RUN_STARTDATE=2015-01-01
./xmlchange STOP_OPTION=nyears
./xmlchange STOP_N=9
./xmlchange CONTINUE_RUN=FALSE
./xmlchange RESUBMIT=0

WDIR=$scratch$CASENAME'/run/'
DDIR=$WDIR'restart_dump/'

mv $WDIR$CASENAME.datm.rs1*.bin $DDIR
mv $WDIRrpointer.lnd.2015-01-01-00000
gzip $DDIR$CASENAME*.bin

# save original user_nl_clm
cp user_nl_clm user_nl_clm_transient_1985-2014

#comment out historical fsurdat and landuse.timeseries and finidat in user_nl_clm
:> user_nl_clm.tmp
while read line; do
    
    if [[ "$line" == *"fsurdat"* || "$line" == *"landuse.timeseries"* || "$line" == *"finidat"* ]]; then
      printf '!%s\n' "$line" >> user_nl_clm.tmp
    else
      printf '%s\n' "$line" >> user_nl_clm.tmp
    fi

    done<user_nl_clm
mv user_nl_clm.tmp user_nl_clm

# update the fsurdat and landuse.timeseries
echo -e "\n">> user_nl_clm
cat user_nl_clm_hist_ssplanduse_2015 >> user_nl_clm

restart=$(ls $scratch$CASENAME'/run/'$CASENAME".clm2.r."*".nc" | tail -n 1)
restart=$(echo $restart) #expands wildcard
echo -e 'finidat="'$restart'"'>>user_nl_clm

