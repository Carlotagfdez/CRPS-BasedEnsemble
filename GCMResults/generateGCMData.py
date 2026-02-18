# Set paths
DATA_PATH = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/0.Datos'
FIGURES_PATH = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/CRPS-BasedEnsemble/Figuras'
MODELS_PATH = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/CRPS-BasedEnsemble/TrainedModels'
PREDS_PATH = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/CRPS-BasedEnsemble/Preds'
ASYM_PATH = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/CRPS-BasedEnsemble/asym_parameters'


import xarray as xr
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from importlib import reload
import os
import sys

from utilsGCMS import load_gcm, load_surface_gcm


# VARIABLES QUE HAY QUE SACAR DE LOS GCMS 
    # t500     (time, lat, lon) float32 84MB ...
    # t700     (time, lat, lon) float32 84MB ...
    # t850     (time, lat, lon) float32 84MB ...
    # q500     (time, lat, lon) float32 84MB ...
    # q700     (time, lat, lon) float32 84MB ...
    # q850     (time, lat, lon) float32 84MB ...
    # v500     (time, lat, lon) float32 84MB ...
    # v700     (time, lat, lon) float32 84MB ...
    # v850     (time, lat, lon) float32 84MB ...
    # u500     (time, lat, lon) float32 84MB ...
    # u700     (time, lat, lon) float32 84MB ...
    # u850     (time, lat, lon) float32 84MB ...
    # msl      (time, lat, lon) float32 84MB ...

# POSIBLES GCMS: 
    # EC-Earth3-VEg
    # CMCC-CM2-SR5 

# PERIODOS DE TIEMPO: 2015-2040 / 2041-2070 / 2071-2100


gcm = 'EC-Earth3_r1i1p1f1' #'CanESM5_r1i1p1f1'
scenario = 'historical'
gcm_path = '/gpfs/projects/meteo/DATA/CMIP6_NorthAtlanticRegion_1.5dg'
gcm_raw = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/CRPS-BasedEnsemble/GCMResults' 


gcm_hist_data = load_gcm(gcm=gcm, scenario=scenario, gcm_path=gcm_path) 
gcm_hist_data.to_netcdf(f'{gcm_raw}/{gcm}_{scenario}.nc')
gcm_SURFACE_data = load_surface_gcm(gcm=gcm, var='pr', scenario=scenario, gcm_path=gcm_path)
gcm_SURFACE_data.to_netcdf(f'{gcm_raw}/{gcm}_{scenario}_surface.nc')

scenario = 'ssp370'

gcm_future_data = load_gcm(gcm=gcm, scenario=scenario, gcm_path=gcm_path) 
gcm_future_data.to_netcdf(f'{gcm_raw}/{gcm}_{scenario}.nc')
gcm_SURFACE_data_proj = load_surface_gcm(gcm=gcm, var='pr', scenario=scenario, gcm_path=gcm_path)
gcm_SURFACE_data_proj.to_netcdf(f'{gcm_raw}/{gcm}_{scenario}_surface.nc')




# AHORA QUIERO GENERAR A PARTIR DE LOS GCMS LOS FICHEROS CON LA VARIABLE A NIVEL DE SUPERFICIE 


