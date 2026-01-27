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

from GCMResults.utilsGCMS import load_gcm


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


gcm = 'CanESM5_r1i1p1f1'
scenario = 'historical'
gcm_path = '/gpfs/projects/meteo/DATA/CMIP6_NorthAtlanticRegion_1.5dg'
gcm_raw = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/CRPS-BasedEnsemble/GCMResults' 


gcm_hist_data = load_gcm(gcm=gcm, scenario=scenario, gcm_path=gcm_path) 
gcm_hist_data.to_netcdf(f'{gcm_raw}/{gcm}_{scenario}.nc')

scenario = 'ssp370'

gcm_future_data = load_gcm(gcm=gcm, scenario=scenario, gcm_path=gcm_path) 
gcm_future_data.to_netcdf(f'{gcm_raw}/{gcm}_{scenario}.nc')

# Lo leemos de mi fichero
gcm_filename = f'{gcm_raw}/{gcm}_historical.nc'
gcm_historical = xr.open_dataset(f'{gcm_raw}/{gcm}_historical.nc').load()

gcm_filename = f'{gcm_raw}/{gcm}_historical.nc'
gcm_future = xr.open_dataset(f'{gcm_raw}/{gcm}_ssp370.nc').load()

# Recortamos los periodos 

future_periodA = ('2015', '2040')
future_periodB = ('2041', '2070')
future_periodC = ('2071', '2100')


gcm_futA = gcm_future.sel(time=slice(*future_periodA))
gcm_futB = gcm_future.sel(time=slice(*future_periodB))
gcm_futC = gcm_future.sel(time=slice(*future_periodC))



print("Inicio:", str(gcm_historical.time.min().values)[:10])
print("Fin:   ", str(gcm_historical.time.max().values)[:10])

print("Inicio:", str(gcm_futA.time.min().values)[:10])
print("Fin:   ", str(gcm_futA.time.max().values)[:10])

print("Inicio:", str(gcm_futB.time.min().values)[:10])
print("Fin:   ", str(gcm_futB.time.max().values)[:10])

print("Inicio:", str(gcm_futC.time.min().values)[:10])
print("Fin:   ", str(gcm_futC.time.max().values)[:10])