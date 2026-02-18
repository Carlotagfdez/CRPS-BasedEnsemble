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


# DeepESD+ASYM:
gcm_filename_h = f'{gcm_raw}/GCM_proj_historical_DeepESD_ASYM_EC-Earth3.nc'
gcm_historical = xr.open_dataset(f'{gcm_filename_h}').load()

gcm_filename_f = f'{gcm_raw}/GCM_proj_future_DeepESD_ASYM_EC-Earth3.nc'
gcm_future = xr.open_dataset(f'{gcm_filename_f}').load()

deep_hist = gcm_historical.sel(time=slice(*historical_period))
deep_futA = gcm_future.sel(time=slice(*future_periodA))
deep_futB = gcm_future.sel(time=slice(*future_periodB))
deep_futC = gcm_future.sel(time=slice(*future_periodC))

# DeepESD+CRPS:
gcm_filename_h = f'{gcm_raw}/GCM_proj_historical_DeepESD_ASYM_EC-Earth3.nc'   # HABRA QUE SELECCIONAR UN MIEMBRO ALEATORIAMENTE
gcm_historicalC = xr.open_dataset(f'{gcm_filename_h}').load()
# gcm_historicalC = gcm_historicalC.sel(member=0)

gcm_filename_f = f'{gcm_raw}/GCM_proj_future_DeepESD_ASYM_EC-Earth3.nc'
gcm_futureC = xr.open_dataset(f'{gcm_filename_f}').load()
# gcm_futureC = gcm_futureC.sel(member=0)

deepC_hist = gcm_historicalC.sel(time=slice(*historical_period))
deepC_futA = gcm_futureC.sel(time=slice(*future_periodA))
deepC_futB = gcm_futureC.sel(time=slice(*future_periodB))
deepC_futC = gcm_futureC.sel(time=slice(*future_periodC))

periods = [
    #("1980–2014", gcm_hist, deep_hist, deepC_hist),
    ("2015–2040", gcm_futA, deep_futA, deepC_futA),
    ("2041–2070", gcm_futB, deep_futB, deepC_futB),
    ("2071–2100", gcm_futC, deep_futC, deepC_futC),
]


# MEDIA DEL HISTORICO 

gcm_histmean = gcm_hist["pr"].mean(dim='time') 
deep_histmean = deep_hist["pr"].mean(dim='time')
deepC_histmean = deepC_hist["pr"].mean(dim='time')

# === FIGURA === MEAN PRECIPITATION ===
fig, axes = plt.subplots(
    nrows=3, ncols=3,
    figsize=(16, 16),
    subplot_kw={'projection': ccrs.PlateCarree()}
)


for i, (label, gcm_ds, deep_ds, crps_ds) in enumerate(periods):

    # Medias temporales
    gcm_mean  = (gcm_ds["pr"].mean(dim="time")-gcm_histmean/gcm_histmean)*100
    deep_mean = (deep_ds["pr"].mean(dim="time")-deep_histmean/deep_histmean)*100
    crps_mean = (crps_ds["pr"].mean(dim="time")-deepC_histmean/deepC_histmean)*100

    for j, (data, title) in enumerate([
        (gcm_mean,  "GCM"),
        (deep_mean, "DeepESD ASYM"),
        (crps_mean, "DeepESD CRPS")
    ]):
        ax = axes[i, j]
        ax.set_extent([-10, 5, 35, 45], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')

        im = ax.pcolormesh(
            data.lon, data.lat, data,
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap="turbo",
        )

        if i == 0:
            ax.set_title(title, fontsize=12, fontweight="bold", pad = 5)
        else:
            ax.set_title("", pad=0)

        ax.text(
            0.02, 0.95, label,
            transform=ax.transAxes,
            fontsize=10, va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )

# Colorbar FUERA de los mapas
cbar_ax = fig.add_axes([0.915, 0.15, 0.015, 0.7])  # más cerca de los mapas
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.yaxis.set_label_position('right')  # etiqueta a la derecha
cbar.ax.set_ylabel("%", rotation=90, labelpad=15, fontsize=16,fontweight="bold") 

fig.text(0.01, 0.5,"Change Signal of Mean Precipitation",fontsize=18, fontweight="bold",rotation=90,va='center',ha='center')

plt.subplots_adjust(left=0.02,right=0.9, top=0.93, bottom=0.05, hspace=0.0025, wspace=0.05)
plt.savefig("Figuras/EvolutionMap_Mean.png", dpi=300, bbox_inches="tight")
plt.savefig("Figuras/EvolutionMap_Mean.pdf", dpi=300, bbox_inches="tight")
plt.close()


##############################################################################################

# === FIGURA === RX1DAY ===
fig, axes = plt.subplots(
    nrows=3, ncols=3,
    figsize=(16, 16),
    subplot_kw={'projection': ccrs.PlateCarree()}
)


# rx1day of historical 


gcm_histmean = gcm_hist["pr"].groupby('time.year').max('time')
gcm_histmean = gcm_histmean.mean('year')
deep_histmean = deep_hist["pr"].groupby('time.year').max('time')
deep_histmean = deep_histmean.mean('year')
deepC_histmean = deepC_hist["pr"].groupby('time.year').max('time')
deepC_histmean = deepC_histmean.mean('year')

for i, (label, gcm_ds, deep_ds, crps_ds) in enumerate(periods):

    # RXday
    rxgcm = gcm_ds["pr"].groupby('time.year').max('time')
    rxgcm = rxgcm.mean('year')
    rxgcm = rxgcm - gcm_histmean/gcm_histmean

    rxgcmdeep = deep_ds["pr"].groupby('time.year').max('time')
    rxgcmdeep = rxgcmdeep.mean('year')
    rxgcmdeep = rxgcmdeep - deep_histmean/deep_histmean

    rxgcmcrps = crps_ds["pr"].groupby('time.year').max('time')
    rxgcmcrps = rxgcmcrps.mean('year')
    rxgcmcrps = rxgcmcrps - deepC_histmean/deep_histmean

    for j, (data, title) in enumerate([
        (rxgcm,  "GCM"),
        (rxgcmdeep, "DeepESD ASYM"),
        (rxgcmcrps, "DeepESD CRPS")
    ]):
        ax = axes[i, j]
        ax.set_extent([-10, 5, 35, 45], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')

        im = ax.pcolormesh(
            data.lon, data.lat, data,
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap="turbo",
        )

        if i == 0:
            ax.set_title(title, fontsize=12, fontweight="bold", pad = 5)
        else:
            ax.set_title("", pad=0)

        ax.text(
            0.02, 0.95, label,
            transform=ax.transAxes,
            fontsize=10, va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )

# Colorbar FUERA de los mapas
cbar_ax = fig.add_axes([0.915, 0.15, 0.015, 0.7])  # más cerca de los mapas
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.yaxis.set_label_position('right')  # etiqueta a la derecha
cbar.ax.set_ylabel("%", rotation=90, labelpad=15, fontsize=16,fontweight="bold") 

fig.text(0.01, 0.5,"Change Signal of Rx1day",fontsize=18, fontweight="bold",rotation=90,va='center',ha='center')

plt.subplots_adjust(left=0.02,right=0.9, top=0.93, bottom=0.05, hspace=0.0025, wspace=0.05)
plt.savefig("Figuras/EvolutionMap_Rx.png", dpi=300, bbox_inches="tight")
plt.savefig("Figuras/EvolutionMap_Rx.pdf", dpi=300, bbox_inches="tight")
plt.close()



##############################################################################################

# === FIGURA === R01 ===
fig, axes = plt.subplots(
    nrows=3, ncols=3,
    figsize=(16, 16),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

threshold = 1
obs_mask = gcm_hist["pr"].mean('time')
obs_mask = (obs_mask - obs_mask) + 1

obs_01 = ((gcm_hist["pr"] >= threshold) * 1) * obs_mask
obs_01_mean = obs_01.mean("time")

obsD_mask = deep_hist["pr"].mean('time')
obsD_mask = (obsD_mask - obsD_mask) + 1

obsD_01 = ((deep_hist["pr"] >= threshold) * 1) * obsD_mask
obsD_01_mean = obsD_01.mean("time")

obsC_mask = deepC_hist["pr"].mean('time')
obsC_mask = (obsC_mask - obsC_mask) + 1

obsC_01 = ((deepC_hist["pr"] >= threshold) * 1) * obsC_mask
obsC_01_mean = obsC_01.mean("time")


for i, (label, gcm_ds, deep_ds, crps_ds) in enumerate(periods):
    

    # R01
    rxgcm_mask = gcm_ds["pr"].mean('time')
    rxgcm_mask = (rxgcm_mask - rxgcm_mask) + 1
    rxgcm_01 = (gcm_ds["pr"] >= threshold)*rxgcm_mask
    rxgcm_01 = rxgcm_01.mean("time")
    r01_G = ((rxgcm_01 - obs_01_mean) / obs_01_mean) * 100

    rxgcmD_mask = deep_ds["pr"].mean('time')
    rxgcmD_mask = (rxgcmD_mask - rxgcmD_mask) + 1
    rxgcmD_01 = (deep_ds["pr"] >= threshold)*rxgcmD_mask
    rxgcmD_01 = rxgcmD_01.mean("time")
    r01_D = ((rxgcmD_01 - obsD_01_mean) / obsD_01_mean) * 100

    rxgcmC_mask = crps_ds["pr"].mean('time')
    rxgcmC_mask = (rxgcmC_mask - rxgcmC_mask) + 1
    rxgcmC_01 = (crps_ds["pr"] >= threshold)*rxgcmC_mask
    r01_D = rxgcmC_01.mean("time")
    r01_C = ((rxgcmC_01 - obsC_01_mean) / obsC_01_mean) * 100

    for j, (data, title) in enumerate([
        (r01_G,  "GCM"),
        (r01_D, "DeepESD ASYM"),
        (r01_C, "DeepESD CRPS")
    ]):
        ax = axes[i, j]
        ax.set_extent([-10, 5, 35, 45], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')

        im = ax.pcolormesh(
            data.lon, data.lat, data,
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap="turbo",
        )

        if i == 0:
            ax.set_title(title, fontsize=12, fontweight="bold", pad = 5)
        else:
            ax.set_title("", pad=0)

        ax.text(
            0.02, 0.95, label,
            transform=ax.transAxes,
            fontsize=10, va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )

# Colorbar FUERA de los mapas
cbar_ax = fig.add_axes([0.915, 0.15, 0.015, 0.7])  # más cerca de los mapas
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.yaxis.set_label_position('right')  # etiqueta a la derecha
cbar.ax.set_ylabel("%", rotation=90, labelpad=15, fontsize=16,fontweight="bold") 

fig.text(0.01, 0.5,"R01",fontsize=18, fontweight="bold",rotation=90,va='center',ha='center')

plt.subplots_adjust(left=0.02,right=0.9, top=0.93, bottom=0.05, hspace=0.0025, wspace=0.05)
plt.savefig("Figuras/EvolutionMap_R01.png", dpi=300, bbox_inches="tight")
plt.savefig("Figuras/EvolutionMap_R01.pdf", dpi=300, bbox_inches="tight")
plt.close()