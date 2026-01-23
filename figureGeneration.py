import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import src.deep.utils as deep_utils
from scipy import fft
from scipy.stats import linregress
# Datos:

# Observado: 
groundTruth = xr.open_dataset("pr_AEMET.nc")
gt = groundTruth["pr"].sel(time=slice('01-01-2011', '12-31-2020'))
gt = gt.sel(lon=slice(-9.425, 3.375))
lat = gt.lat.values
dlat = np.diff(lat).mean()
n_extra = 256 - lat.size
new_lat = np.concatenate([lat, lat[-1] + dlat * np.arange(1, n_extra + 1)])
gt = gt.reindex(lat=new_lat)

# Modelos: 
vitRmse = xr.open_dataset("ViT_MSE_pr_pred_test.nc")
vr = vitRmse["pr"].sel(time=slice('01-01-2011', '12-31-2020'))

vitAsym = xr.open_dataset("modelos/vit_ASYM.nc") #xr.open_dataset("ViT_ASYM_pr_pred_test.nc")
va = vitAsym["pr"].sel(time=slice('01-01-2011', '12-31-2020'))

vitCRPS = xr.open_dataset("modelos/vit_CRPS.nc")
vc = vitCRPS["pr"].sel(time=slice('01-01-2011', '12-31-2020'))
vc1 = vc.sel(member=0)

vitBerGamma =  xr.open_dataset("modelos/vit_BerGamma.nc") 
vg = vitBerGamma["pr"].sel(time=slice('01-01-2011', '12-31-2020'))
vg1 = vg.sel(member=0)

vitCRPSSpect =  xr.open_dataset("modelos/vit_CRPS_spectral.nc") 
vs = vitCRPSSpect["pr"].sel(time=slice('01-01-2011', '12-31-2020'))
vs1 = vs.sel(member=0)

gt_flat = gt.values.flatten()
vr_flat = vr.values.flatten()
va_flat = va.values.flatten()
vc_flat = vc1.values.flatten()
vg_flat = vg1.values.flatten()
vs_flat = vs1.values.flatten()

gt_flat = gt_flat[~np.isnan(gt_flat)]
vr_flat = vr_flat[~np.isnan(vr_flat)]
va_flat = va_flat[~np.isnan(va_flat)]
vc_flat = vc_flat[~np.isnan(vc_flat)]
vg_flat = vg_flat[~np.isnan(vg_flat)]
vs_flat = vs_flat[~np.isnan(vs_flat)]

gt_flat = gt_flat[gt_flat >= 0]
vr_flat = vr_flat[vr_flat >= 0]
va_flat = va_flat[va_flat >= 0]
vc_flat = vc_flat[vc_flat >= 0]
vg_flat = vg_flat[vg_flat >= 0]
vs_flat = vs_flat[vs_flat >= 0]

bins = 100
bins_full = 1000
range_linear = (1, 50)
range_linear_full = (0, 100)

colors = {
    "Ground Truth": "black",
    "MSE": "blue",
    "ASYM": "red",
    "Bernoulli-Gamma": "purple",
    "CRPS": "green",
    "CRPS Spectral": "gray"
}
def compute_spatial_psd_2d(data):
    """
    Compute radially averaged 2D PSD for a spatial field.
    """
    data = np.nan_to_num(data, nan=0.0)
    ny, nx = data.shape[-2:]
    fft_data = fft.fft2(data, axes=(-2, -1))
    power = np.abs(fft_data) ** 2

    # Average over time if present
    if len(data.shape) > 2:
        power = power.mean(axis=0)

    kx = fft.fftfreq(nx)
    ky = fft.fftfreq(ny)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_grid = np.sqrt(kx_grid**2 + ky_grid**2)

    k_bins = np.linspace(0, k_grid.max(), min(nx, ny)//2)
    psd_radial = np.zeros(len(k_bins)-1)

    for i in range(len(k_bins)-1):
        mask = (k_grid >= k_bins[i]) & (k_grid < k_bins[i+1])
        if mask.any():
            psd_radial[i] = power[mask].mean()

    k_centers = (k_bins[:-1] + k_bins[1:]) / 2
    return k_centers[k_centers>0], psd_radial[k_centers>0]

# ---------- (c) Power Spectral Density ----------
# Seleccionamos primer miembro de cada modelo
psd_models = {"Ground Truth": gt, "MSE": vr,
              "ASYM": va, "Bernoulli-Gamma": vg1,
              "CRPS": vc1, "CRPS Spectral": vs1}


# ==========================================
# FIGURA COMBINADA: 2 FILAS
# ==========================================
fig = plt.figure(figsize=(26, 10))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.1])

# -------- Fila 1: Gráficos estadísticos --------
ax_psd   = fig.add_subplot(gs[0, 0:2])
ax_hist  = fig.add_subplot(gs[0, 2:4])
ax_skill = fig.add_subplot(gs[0, 4:6])

# ===== (a) PSD =====
for name, data in psd_models.items():
    k, psd = compute_spatial_psd_2d(data)
    ax_psd.loglog(k, psd, label=name, color=colors[name])

ax_psd.set_title('(a) Spatial Power Spectral Density', fontweight='bold', fontsize=16)
ax_psd.set_xlabel('Spatial wavenumber', fontweight='bold', fontsize=16)
ax_psd.set_ylabel('Power', fontweight='bold', fontsize=16)
ax_psd.legend(fontsize=15)
ax_psd.grid(True, which="both", alpha=0.3)


# ===== (b) Histograma =====
ax_hist.hist(gt_flat, bins=bins, histtype='step', color='black', label='GT')
ax_hist.hist(vr_flat, bins=bins, histtype='step', color='blue', label='MSE')
ax_hist.hist(va_flat, bins=bins, histtype='step', color='red', label='ASYM')
ax_hist.hist(vg_flat, bins=bins, histtype='step', color='purple', label='Bernoulli-Gamma')
ax_hist.hist(vc_flat, bins=bins, histtype='step', color='green', label='CRPS')
ax_hist.hist(vs_flat, bins=bins, histtype='step', color='gray', label='CRPS Spectral')

ax_hist.set_yscale('log')
ax_hist.set_title('(b) Precipitation distribution', fontweight='bold', fontsize=16)
ax_hist.set_xlabel('Precipitation (mm/day)', fontweight='bold', fontsize=16)
ax_hist.set_ylabel('Count', fontweight='bold', fontsize=16)
ax_hist.legend(fontsize=15)
ax_hist.grid(True, alpha=0.3)

# ===== (c) Spread–Skill =====

# ===== (c) Spread–Skill =====
models = {
    "Bernoulli-Gamma": vg,
    "CRPS": vc,
    "CRPS Spectral": vs
}

colors = {
    "Bernoulli-Gamma": "purple",
    "CRPS": "green",
    "CRPS Spectral": "gray"
}
for name, pred_all in models.items():
    mean_pred = pred_all.mean(dim="member")
    rmse_spatial = np.sqrt(((mean_pred - gt)**2).mean(dim="time"))
    spread_spatial = pred_all.std(dim="member").mean(dim="time")

    x = rmse_spatial.values.flatten()
    y = spread_spatial.values.flatten()
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]

    ax_skill.scatter(x, y, s=5, alpha=0.5, color=colors[name], label=name)
    slope, intercept, r, *_ = linregress(x, y)
    xx = np.linspace(x.min(), x.max(), 100)
    ax_skill.plot(xx, slope*xx+intercept, '--', color=colors[name])

ax_skill.plot([0, x.max()], [0, x.max()], '--', color='black')
ax_skill.set_title('(c) Spread–skill relationship', fontweight='bold', fontsize=16)
ax_skill.set_xlabel('RMSE', fontweight='bold', fontsize=16)
ax_skill.set_ylabel('Spread', fontweight='bold', fontsize=16)
ax_skill.legend(fontsize=15, markerscale=3)
ax_skill.grid(True, alpha=0.3)

# -------- Fila 2: Mapas --------

# -------- Fila 2: Mapas --------
# Lista de DataArrays y títulos
day ='18-10-2017'
gt_day = gt.sel(time=day)
vr_day = vr.sel(time=day)
va_day = va.sel(time=day)
vc_day = vc1.sel(time=day)  
vg_day = vg1.sel(time=day)   
vs_day = vs1.sel(time=day)  
# ve_day = ve1.sel(time=day) 
lon_min, lon_max = -10.0, 5.0   # longitud mínima y máxima
lat_min, lat_max = 35.0, 45.0   # latitud mínima y máxima
data_list = [gt_day, vr_day, va_day, vg_day, vc_day, vs_day]
titles = ["Ground Truth", "MSE", "ASYM", "Bernoulli-Gamma", "CRPS", "CRPS Spectral"]

# Límites de color comunes
vmin = min(float(gt_day.min()), float(gt_day.min()))
vmax = max(float(gt_day.max()), float(gt_day.max()))
axes_maps = []
for j in range(6):
    axm = fig.add_subplot(gs[1, j], projection=ccrs.PlateCarree())
    axes_maps.append(axm)

model_colors = ["black","blue","red","purple","green","gray"]

for j, da in enumerate(data_list):
    mappable = da.plot(
        ax=axes_maps[j],
        transform=ccrs.PlateCarree(),
        cmap="turbo",
        vmin=vmin, vmax=vmax,
        add_colorbar=False
    )

    # 👉 SIN títulos
    axes_maps[j].set_title("")

    # 👉 Borde con color del modelo
    for spine in axes_maps[j].spines.values():
        spine.set_edgecolor(model_colors[j])
        spine.set_linewidth(5)

    axes_maps[j].add_feature(cfeature.COASTLINE, linewidth=1)
    axes_maps[j].add_feature(cfeature.BORDERS, linestyle=':')
    axes_maps[j].set_extent([lon_min, lon_max, lat_min, lat_max])

# 👉 Colorbar más bajita
cbar_ax = fig.add_axes([0.90, 0.17, 0.010, 0.20])
cbar = fig.colorbar(mappable, cax=cbar_ax)
cbar.set_label("Pr (mm)", fontweight='bold')

plt.tight_layout(rect=[0, 0, 0.90, 1], h_pad=0.7)
plt.savefig("Figuras/Fig_Combined_2Lines.png", dpi=300, bbox_inches="tight")
plt.savefig("Figuras/Fig_Combined_2Lines.pdf", dpi=300, bbox_inches="tight")
plt.close()

