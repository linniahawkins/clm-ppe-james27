import xarray as xr
import numpy as np
import glob
import sys
sys.path.append('/glade/u/home/linnia/CLM6-PPE/')

utils_path = '/glade/u/home/linnia/CLM6-PPE/utils/'
sys.path.append(utils_path)
from pyfunctions import *

ens=sys.argv[1] 
out_dir=sys.argv[2]
if out_dir[-1]!='/':
    out_dir+='/'

######################################################
# setup 
dir='/glade/campaign/cgd/tss/projects/PPE/ctsm53065_lhc/ssp370/'
exp = 'ctsm5.3.065_transient_SSP370_'
tape='h0a'

dvs=['GPP','AR','HR','NPP','NBP','NEP','ER',
     'EFLX_LH_TOT','FCTR','FCEV','FGEV','BTRANMN','FGR','FSH',
     'SOILWATER_10CM','TWS','QRUNOFF','SNOWDP','H2OSNO','FSNO', 'TLAI','FSR','FSDS','ALTMAX','TV','TG','NPP_NUPTAKE','LAND_USE_FLUX',
     'FAREA_BURNED','COL_FIRE_CLOSS',
     'TOTVEGC','TOTECOSYSC','TOTSOMC_1m',
     'TOTVEGN','TOTECOSYSN']

def pp(ds):
    return ds[dvs]

def bmean(da,la):
    x=(1/la.sum(dim='gridcell'))*(la*da).sum(dim='gridcell')
    return x.compute()

######################################################
# load and process data

f0=sorted(glob.glob(dir+exp+'lhc0000*.'+tape+'.*'))
f=sorted(glob.glob(dir+'*'+ens+'*.'+tape+'.*'))

if len(f)<len(f0):
    #hacky way to generate correctly shaped nan output for failed simulations
    # if the requested lhc is missing, we will reanalyze lhc0000 and multiply by np.nan
    bad=True
    f=f0
else:
    bad=False
    
ds=xr.open_mfdataset(f,combine='by_coords',preprocess=pp)

# calculate biome mean
la=xr.open_dataset(utils_path+'lasg_biomes_ctsm53065.nc').landarea_biome
out=xr.Dataset()

for v in dvs:
    if v=='TLAI':
        x=amax(ds[v])
        out[v+'_biome_amax'] = bmean(x,la)
        out[v+'_biome_amax'].attrs =ds[v].attrs
    else:
        x=amean(ds[v])
        out[v+'_biome_amean'] =bmean(x,la)
        out[v+'_biome_amean'].attrs =ds[v].attrs

# save 
fout=out_dir+f[0].split('/')[-1].split('clm2')[0]+'postp.nc'
# nan output if no files
if bad:
    out=np.nan*out
    fout=fout.replace('lhc0000',ens)
    
out.to_netcdf(fout)

