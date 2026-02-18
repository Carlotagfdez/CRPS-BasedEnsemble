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


# === FIGURA: 3 filas x 2 columnas (GCM vs DeepESD ASYM) ===
fig, axes = plt.subplots(
    nrows=3, ncols=3,
    figsize=(14, 14),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

datasets = [
    (gcm_cc_A,  deepASYM_A_cc, vitASYM_A_cc, "2015–2040"),
    (gcm_cc_B,  deepASYM_B_cc, vitASYM_B_cc, "2041–2070"),
    (gcm_cc_C,  deepASYM_C_cc, vitASYM_C_cc,"2071–2100")
]

for i, (gcm_data, deep_data, vit_data, period_label) in enumerate(datasets):

    for j, (data, col_title) in enumerate([
        (gcm_data, "GCM"),
        (deep_data, "DeepESD ASYM"),
        (vit_data, "Vit_ASYM")
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
plt.savefig("Figuras/ClimateSignalChange_Mean.png", dpi=300, bbox_inches="tight")
plt.close()

gcm_rx1_A = metrics_ccs.compute_ccs(hist_data=gcm_hist, fut_data=gcm_futA, reduction_function=metrics_ccs.RX1day, relative=True)
gcm_rx1_B = metrics_ccs.compute_ccs(hist_data=gcm_hist, fut_data=gcm_futB, reduction_function=metrics_ccs.RX1day, relative=True)
gcm_rx1_C = metrics_ccs.compute_ccs(hist_data=gcm_hist, fut_data=gcm_futC, reduction_function=metrics_ccs.RX1day, relative=True)

deepA_rx1_A = metrics_ccs.compute_ccs(hist_data=deepASYM_hist, fut_data=deepASYM_A, reduction_function=metrics_ccs.RX1day, relative=True)
deepA_rx1_B = metrics_ccs.compute_ccs(hist_data=deepASYM_hist, fut_data=deepASYM_B, reduction_function=metrics_ccs.RX1day, relative=True)
deepA_rx1_C = metrics_ccs.compute_ccs(hist_data=deepASYM_hist, fut_data=deepASYM_C, reduction_function=metrics_ccs.RX1day, relative=True)

vitA_rx1_A = metrics_ccs.compute_ccs(hist_data=vitASYM_hist, fut_data=vitASYM_A, reduction_function=metrics_ccs.RX1day, relative=True)
vitA_rx1_B = metrics_ccs.compute_ccs(hist_data=vitASYM_hist, fut_data=vitASYM_B, reduction_function=metrics_ccs.RX1day, relative=True)
vitA_rx1_C = metrics_ccs.compute_ccs(hist_data=vitASYM_hist, fut_data=vitASYM_C, reduction_function=metrics_ccs.RX1day, relative=True)


# === FIGURA: 3 filas x 2 columnas (GCM vs DeepESD ASYM) ===
fig, axes = plt.subplots(
    nrows=3, ncols=3,
    figsize=(14, 14),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

datasets = [
    (gcm_rx1_A,  deepA_rx1_A, vitA_rx1_A, "2015–2040"),
    (gcm_rx1_B,  deepA_rx1_B, vitA_rx1_B, "2041–2070"),
    (gcm_rx1_C,  deepA_rx1_C, vitA_rx1_C, "2071–2100")
]

for i, (gcm_data, deep_data, vit_data, period_label) in enumerate(datasets):

    for j, (data, col_title) in enumerate([
        (gcm_data, "GCM"),
        (deep_data, "DeepESD ASYM"),
        (vit_data, "ViT ASYM")
    ]):
        ax = axes[i, j]

        ax.set_extent([-10, 5, 35, 45], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAND, facecolor="lightgray")

        data = data.pr

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
cbar.set_label("RX1day % Change Signal", fontsize=12, fontweight="bold")

plt.subplots_adjust(left=0.05, right=0.9, top=0.93, bottom=0.05, hspace=0.05, wspace=0.05)
plt.savefig("Figuras/ClimateSignalChange_RX1day.png", dpi=300, bbox_inches="tight")
plt.close()



gcm_rx1_A = metrics_ccs.compute_ccs(hist_data=gcm_hist, fut_data=gcm_futA, reduction_function=metrics_ccs.R01, relative=True)
gcm_rx1_B = metrics_ccs.compute_ccs(hist_data=gcm_hist, fut_data=gcm_futB, reduction_function=metrics_ccs.R01, relative=True)
gcm_rx1_C = metrics_ccs.compute_ccs(hist_data=gcm_hist, fut_data=gcm_futC, reduction_function=metrics_ccs.R01, relative=True)

deepA_rx1_A = metrics_ccs.compute_ccs(hist_data=deepASYM_hist, fut_data=deepASYM_A, reduction_function=metrics_ccs.R01, relative=True)
deepA_rx1_B = metrics_ccs.compute_ccs(hist_data=deepASYM_hist, fut_data=deepASYM_B, reduction_function=metrics_ccs.R01, relative=True)
deepA_rx1_C = metrics_ccs.compute_ccs(hist_data=deepASYM_hist, fut_data=deepASYM_C, reduction_function=metrics_ccs.R01, relative=True)

vitA_rx1_A = metrics_ccs.compute_ccs(hist_data=vitASYM_hist, fut_data=vitASYM_A, reduction_function=metrics_ccs.R01, relative=True)
vitA_rx1_B = metrics_ccs.compute_ccs(hist_data=vitASYM_hist, fut_data=vitASYM_B, reduction_function=metrics_ccs.R01, relative=True)
vitA_rx1_C = metrics_ccs.compute_ccs(hist_data=vitASYM_hist, fut_data=vitASYM_C, reduction_function=metrics_ccs.R01, relative=True)

# === FIGURA: 3 filas x 2 columnas (GCM vs DeepESD ASYM) ===
fig, axes = plt.subplots(
    nrows=3, ncols=3,
    figsize=(14, 14),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

datasets = [
    (gcm_rx1_A,  deepA_rx1_A, vitA_rx1_A , "2015–2040"),
    (gcm_rx1_B,  deepA_rx1_B, vitA_rx1_B, "2041–2070"),
    (gcm_rx1_C,  deepA_rx1_C, vitA_rx1_C, "2071–2100")
]

for i, (gcm_data, deep_data, vit_data, period_label) in enumerate(datasets):

    for j, (data, col_title) in enumerate([
        (gcm_data, "GCM"),
        (deep_data, "DeepESD ASYM"),
        (vit_data, "ViT ASYM")
    ]):
        ax = axes[i, j]

        ax.set_extent([-10, 5, 35, 45], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAND, facecolor="lightgray")

        data = data.pr

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
cbar.set_label("R01 % Change Signal", fontsize=12, fontweight="bold")

plt.subplots_adjust(left=0.05, right=0.9, top=0.93, bottom=0.05, hspace=0.05, wspace=0.05)
plt.savefig("Figuras/ClimateSignalChange_R01.png", dpi=300, bbox_inches="tight")
plt.close()

# # Figura
# fig = plt.figure(figsize=(8, 6))
# ax = plt.axes(projection=ccrs.PlateCarree())

# ax.set_extent([-25, 25, 20, 75], crs=ccrs.PlateCarree())

# ax.add_feature(cfeature.COASTLINE, linewidth=1)
# ax.add_feature(cfeature.BORDERS, linestyle=":")
# ax.add_feature(cfeature.LAND, facecolor="lightgray")

# # Pintar el campo
# im = ax.pcolormesh(
#     gcm_cc_A.lon, gcm_cc_A.lat, gcm_cc_A,
#     transform=ccrs.PlateCarree(),
#     shading="auto",
#     cmap="BrBG",
#     vmin=-40, vmax=40
# )

# cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02)

# plt.tight_layout()
# plt.savefig("Figuras/ClimateSignalChange_Mean.png", dpi=300, bbox_inches="tight")






