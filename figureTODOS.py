import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import deep.utils as deep_utils
from scipy import fft
from scipy.stats import linregress

import metrics_ccs


## BLOQUE DATOS GCM: 
# Datos:
gcm_raw = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/CRPS-BasedEnsemble/GCMResults' 

# Periodos de Comparación:
historical_period = ('1980','2014')
future_periodA = ('2015', '2040')
future_periodB = ('2041', '2070')
future_periodC = ('2071', '2100')

# GCM:
gcm_filename_GCM = f'{gcm_raw}/EC-Earth3_r1i1p1f1_historical_surface.nc'
gcm = xr.open_dataset(f'{gcm_filename_GCM}').load()
gcm_filename_GCM = f'{gcm_raw}/EC-Earth3_r1i1p1f1_ssp370_surface.nc'
gcmf = xr.open_dataset(f'{gcm_filename_GCM}').load()

gcm = gcm * 86400
gcmf = gcmf * 86400

gcm_hist = gcm.sel(time=slice(*historical_period))
gcm_futA = gcmf.sel(time=slice(*future_periodA))
gcm_futB = gcmf.sel(time=slice(*future_periodB))
gcm_futC = gcmf.sel(time=slice(*future_periodC))

gcm_cc_A = (gcm_futA.mean('time') - gcm_hist.mean('time')) / gcm_hist.mean('time')
gcm_cc_A = gcm_cc_A * 100
gcm_cc_A = gcm_cc_A.pr

gcm_cc_B = (gcm_futB.mean('time') - gcm_hist.mean('time')) / gcm_hist.mean('time')
gcm_cc_B = gcm_cc_B * 100
gcm_cc_B = gcm_cc_B.pr

gcm_cc_C = (gcm_futC.mean('time') - gcm_hist.mean('time')) / gcm_hist.mean('time')
gcm_cc_C = gcm_cc_C * 100
gcm_cc_C = gcm_cc_C.pr

## BLOQUE DEEP ESD ASYM: 
deepASYM_filename_GCM = f'{gcm_raw}/GCM_proj_historical_DeepESD_ASYM_EC-Earth3.nc'
deepASYM = xr.open_dataset(f'{deepASYM_filename_GCM}').load()
deepASYM_filename_GCM = f'{gcm_raw}/GCM_proj_future_DeepESD_ASYM_EC-Earth3.nc'
deepASYMf = xr.open_dataset(f'{deepASYM_filename_GCM}').load()

deepASYM_hist = deepASYM.sel(time=slice(*historical_period))
deepASYM_A = deepASYMf.sel(time=slice(*future_periodA))
deepASYM_B = deepASYMf.sel(time=slice(*future_periodB))
deepASYM_C = deepASYMf.sel(time=slice(*future_periodC))

deepASYM_A_cc = (deepASYM_A.mean('time') - deepASYM_hist.mean('time')) / deepASYM_hist.mean('time')
deepASYM_A_cc = deepASYM_A_cc * 100
deepASYM_A_cc = deepASYM_A_cc.pr

deepASYM_B_cc = (deepASYM_B.mean('time') - deepASYM_hist.mean('time')) / deepASYM_hist.mean('time')
deepASYM_B_cc = deepASYM_B_cc * 100
deepASYM_B_cc = deepASYM_B_cc.pr

deepASYM_C_cc = (deepASYM_C.mean('time') - deepASYM_hist.mean('time')) / deepASYM_hist.mean('time')
deepASYM_C_cc = deepASYM_C_cc * 100
deepASYM_C_cc = deepASYM_C_cc.pr

## BLOQUE DEEP ESD CRPS: 
deepCRPS_filename_GCM = f'{gcm_raw}/GCM_proj_historical_DeepESD_CRPS_EC-Earth3.nc'
deepCRPS = xr.open_dataset(f'{deepCRPS_filename_GCM}').load()
deepCRPS_filename_GCM = f'{gcm_raw}/GCM_proj_future_DeepESD_CRPS_EC-Earth3.nc'
deepCRPSf = xr.open_dataset(f'{deepCRPS_filename_GCM}').load()

deepCRPS_hist = deepCRPS.sel(time=slice(*historical_period))
deepCRPS_A = deepCRPSf.sel(time=slice(*future_periodA))
deepCRPS_B = deepCRPSf.sel(time=slice(*future_periodB))
deepCRPS_C = deepCRPSf.sel(time=slice(*future_periodC))

deepCRPS_A_cc = (deepCRPS_A.mean('time') - deepCRPS_hist.mean('time')) / deepCRPS_hist.mean('time')
deepCRPS_A_cc = deepCRPS_A_cc * 100
deepCRPS_A_cc = deepCRPS_A_cc.pr

deepCRPS_B_cc = (deepCRPS_B.mean('time') - deepCRPS_hist.mean('time')) / deepCRPS_hist.mean('time')
deepCRPS_B_cc = deepCRPS_B_cc * 100
deepCRPS_B_cc = deepCRPS_B_cc.pr

deepCRPS_C_cc = (deepCRPS_C.mean('time') - deepCRPS_hist.mean('time')) / deepCRPS_hist.mean('time')
deepCRPS_C_cc = deepCRPS_C_cc * 100
deepCRPS_C_cc = deepCRPS_C_cc.pr

## BLOQUE VIT ASYM: 
vitASYM_filename_GCM = f'{gcm_raw}/GCM_proj_historical_ViT_ASYM_EC-Earth3.nc'
vitASYM = xr.open_dataset(f'{vitASYM_filename_GCM}').load()
vitASYM_filename_GCM = f'{gcm_raw}/GCM_proj_future_ViT_ASYM_EC-Earth3.nc'
vitASYMf = xr.open_dataset(f'{vitASYM_filename_GCM}').load()

vitASYM_hist = vitASYM.sel(time=slice(*historical_period))
vitASYM_A = vitASYMf.sel(time=slice(*future_periodA))
vitASYM_B = vitASYMf.sel(time=slice(*future_periodB))
vitASYM_C = vitASYMf.sel(time=slice(*future_periodC))

vitASYM_A_cc = (vitASYM_A.mean('time') - vitASYM_hist.mean('time')) / vitASYM_hist.mean('time')
vitASYM_A_cc = vitASYM_A_cc * 100
vitASYM_A_cc = vitASYM_A_cc.pr

vitASYM_B_cc = (vitASYM_B.mean('time') - vitASYM_hist.mean('time')) / vitASYM_hist.mean('time')
vitASYM_B_cc = vitASYM_B_cc * 100
vitASYM_B_cc = vitASYM_B_cc.pr

vitASYM_C_cc = (vitASYM_C.mean('time') - vitASYM_hist.mean('time')) / vitASYM_hist.mean('time')
vitASYM_C_cc = vitASYM_C_cc * 100
vitASYM_C_cc = vitASYM_C_cc.pr


## BLOQUE VIT CRPS: 
vitCRPS_filename_GCM = f'{gcm_raw}/GCM_proj_historical_ViT_CRPS_EC-Earth3.nc'
vitCRPS = xr.open_dataset(f'{vitCRPS_filename_GCM}').load()
vitCRPS_filename_GCM = f'{gcm_raw}/GCM_proj_future_ViT_CRPS_EC-Earth3.nc'
vitCRPSf = xr.open_dataset(f'{vitCRPS_filename_GCM}').load()

vitCRPS_hist = vitCRPS.sel(time=slice(*historical_period))
vitCRPS_A = vitCRPSf.sel(time=slice(*future_periodA))
vitCRPS_B = vitCRPSf.sel(time=slice(*future_periodB))
vitCRPS_C = vitCRPSf.sel(time=slice(*future_periodC))

vitCRPS_A_cc = (vitCRPS_A.mean('time') - vitCRPS_hist.mean('time')) / vitCRPS_hist.mean('time')
vitCRPS_A_cc = vitCRPS_A_cc * 100
vitCRPS_A_cc = vitCRPS_A_cc.pr

vitCRPS_B_cc = (vitCRPS_B.mean('time') - vitCRPS_hist.mean('time')) / vitCRPS_hist.mean('time')
vitCRPS_B_cc = vitCRPS_B_cc * 100
vitCRPS_B_cc = vitCRPS_B_cc.pr

vitCRPS_C_cc = (vitCRPS_C.mean('time') - vitCRPS_hist.mean('time')) / vitCRPS_hist.mean('time')
vitCRPS_C_cc = vitCRPS_C_cc * 100
vitCRPS_C_cc = vitCRPS_C_cc.pr

## BLOQUE VIT CRPS_SPECTRAL: 
vitCRPSS_filename_GCM = f'{gcm_raw}/GCM_proj_historical_ViT_CRPSS_EC-Earth3.nc'
vitCRPSS = xr.open_dataset(f'{vitCRPSS_filename_GCM}').load()
vitCRPSS_filename_GCM = f'{gcm_raw}/GCM_proj_future_ViT_CRPSS_EC-Earth3.nc'
vitCRPSSf = xr.open_dataset(f'{vitCRPSS_filename_GCM}').load()

vitCRPSS_hist = vitCRPSS.sel(time=slice(*historical_period))
vitCRPSS_A = vitCRPSSf.sel(time=slice(*future_periodA))
vitCRPSS_B = vitCRPSSf.sel(time=slice(*future_periodB))
vitCRPSS_C = vitCRPSSf.sel(time=slice(*future_periodC))

vitCRPSS_A_cc = (vitCRPSS_A.mean('time') - vitCRPSS_hist.mean('time')) / vitCRPSS_hist.mean('time')
vitCRPSS_A_cc = vitCRPSS_A_cc * 100
vitCRPSS_A_cc = vitCRPSS_A_cc.pr

vitCRPSS_B_cc = (vitCRPSS_B.mean('time') - vitCRPSS_hist.mean('time')) / vitCRPSS_hist.mean('time')
vitCRPSS_B_cc = vitCRPSS_B_cc * 100
vitCRPSS_B_cc = vitCRPSS_B_cc.pr

vitCRPSS_C_cc = (vitCRPSS_C.mean('time') - vitCRPSS_hist.mean('time')) / vitCRPSS_hist.mean('time')
vitCRPSS_C_cc = vitCRPSS_C_cc * 100
vitCRPSS_C_cc = vitCRPSS_C_cc.pr


# === FIGURA: 3 filas x 2 columnas (GCM vs DeepESD ASYM) ===
fig, axes = plt.subplots(
    nrows=3, ncols=6,
    figsize=(14, 14),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

datasets = [
    (gcm_cc_A,  deepASYM_A_cc, deepCRPS_A_cc, vitASYM_A_cc, vitCRPS_A_cc, vitCRPSS_A_cc, "2015–2040"),
    (gcm_cc_B,  deepASYM_B_cc, deepCRPS_B_cc , vitASYM_B_cc, vitCRPS_B_cc, vitCRPSS_B_cc, "2041–2070"),
    (gcm_cc_C,  deepASYM_C_cc, deepCRPS_C_cc , vitASYM_C_cc, vitCRPS_C_cc, vitCRPSS_C_cc, "2071–2100")
]

for i, (gcm_data, deepA_data, deepC_data, vitA_data, vitC_data, vitCS_data, period_label) in enumerate(datasets):

    for j, (data, col_title) in enumerate([
        (gcm_data, "GCM"),
        (deepA_data, "DeepESD ASYM"),
        (deepC_data, "DeepESD CRPS"),
        (vitA_data, "ViT ASYM"),
         (vitC_data, "ViT CRPS"),
          (vitCS_data, "ViT CRPS Spectral")

    ]):
        ax = axes[i, j]

        ax.set_extent([-10, 5, 35, 45], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAND, facecolor="lightgray")


        im = ax.pcolormesh(
            data.lon, data.lat, data,
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap="BrBG",
            vmin=-40, vmax=40
        )

        if i == 0:
            ax.set_title(col_title, fontsize=12, fontweight="bold")

        ax.text(
            0.02, 0.95, period_label,
            transform=ax.transAxes,
            fontsize=10, va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )

# === Colorbar fuera de los mapas ===
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])   # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Mean Precipitaction % Change Signal", fontsize=12, fontweight="bold")

plt.subplots_adjust(left=0.05, right=0.9, top=0.93, bottom=0.05, hspace=0.05, wspace=0.05)
plt.savefig("Figuras/AllClimateSignalChange_Mean.png", dpi=300, bbox_inches="tight")
plt.close()