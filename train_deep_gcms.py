# Set paths
DATA_PATH = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/0.Datos'
FIGURES_PATH = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/CRPS-BasedEnsemble/Figuras'
MODELS_PATH = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/CRPS-BasedEnsemble/TrainedModels'
PREDS_PATH = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/CRPS-BasedEnsemble/Preds'
ASYM_PATH = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/CRPS-BasedEnsemble/asym_parameters'

# Import libraries
import xarray as xr
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from importlib import reload

import sys

import deep.viz
import deep.trans as trans
import deep.loss
import deep.utils
import deep.models
import deep.train 
import deep.pred
import deep.utils 

# Opción 1:
uncertainty_approach = 'ASYM' 

# Opción 2:
# uncertainty_approach = 'CRPS'

# Set device
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Load predictors
predictor_filename = f'{DATA_PATH}/ERA5_NorthAtlanticRegion_1-5dg_full.nc'
predictor = xr.open_dataset(predictor_filename)
predictor = predictor.load()

# Subset predictors 
predictor = predictor.sel(lon=slice(-24, 22.5))

# Extent lattiude to 32 grid points
lat = predictor.lat.values
dlat = np.diff(lat).mean()
n_extra = 32 - lat.size
new_lat = np.concatenate([lat, lat[-1] + dlat * np.arange(1, n_extra + 1)])
predictor = predictor.reindex(lat=new_lat, method='nearest')

# Load predictand
predictand_filename = f'{DATA_PATH}/pr_AEMET.nc'
predictand = xr.open_dataset(predictand_filename)
predictand = predictand.load()

# Subset predictand
predictand = predictand.sel(lon=slice(-9.425, 3.375))

# Extend the latitude to 264 grid points
lat = predictand.lat.values
dlat = np.diff(lat).mean()
n_extra = 256 - lat.size
new_lat = np.concatenate([lat, lat[-1] + dlat * np.arange(1, n_extra + 1)])

predictand = predictand.reindex(lat=new_lat)

# Remove days with nans in the predictor
predictor = trans.remove_days_with_nans(predictor)

# Align both datasets in time
predictor, predictand = trans.align_datasets(predictor, predictand, 'time')

# Split data into training and test sets
years_train = ('1980', '2010')
years_test = ('2011', '2020')

x_train = predictor.sel(time=slice(*years_train))
y_train = predictand.sel(time=slice(*years_train))

x_test = predictor.sel(time=slice(*years_test))
y_test = predictand.sel(time=slice(*years_test))

# Standardize the predictors
x_train_stand = trans.standardize(data_ref=x_train, data=x_train)

# Stack the predictand
y_train_stack = y_train.stack(gridpoint=('lat', 'lon'))

# Por ser el modelo DeepESD:
y_mask = trans.compute_valid_mask(y_train) 
y_mask_stack = y_mask.stack(gridpoint=('lat', 'lon'))
y_mask_stack_filt = y_mask_stack.where(y_mask_stack==1, drop=True)
y_train_stack_filt = y_train_stack.where(y_train_stack['gridpoint'] == y_mask_stack_filt['gridpoint'],
                                             drop=True)


# Fit the Gammas for the ASYM loss function
if uncertainty_approach == 'ASYM':
    loss_function = deep.loss.Asym(ignore_nans=True,
                                   asym_path=ASYM_PATH)
    if not loss_function.parameters_exist():
        loss_function.compute_parameters(data=y_train_stack,
                                     var_target='pr')
    loss_function.load_parameters()
    loss_function.prepare_parameters(device=device)
    loss_function.mask_parameters(mask=y_mask)
elif uncertainty_approach == 'MSE':
    loss_function = deep.loss.MseLoss(ignore_nans=True)
elif uncertainty_approach == 'CRPS':
    loss_function = deep.loss.CRPSLoss(ignore_nans=True)

# Convert the data to numpy arrays
x_train_stand_arr = trans.xarray_to_numpy(x_train_stand)
y_train_arr = trans.xarray_to_numpy(y_train_stack_filt)

# Create Dataset
train_dataset = deep.utils.StandardDataset(x=x_train_stand_arr,
                                                            y=y_train_arr)

# Split into training and validation sets
train_dataset, valid_dataset = random_split(train_dataset,
                                            [0.9, 0.1])

# Create DataLoaders
batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=True)

# Set model name
model_name = f'DeepESD_{uncertainty_approach}'

# Create model
if uncertainty_approach == 'ASYM' or uncertainty_approach == 'MSE':
    model = deep.models.DeepESDpr(x_shape=x_train_stand_arr.shape,
                                      y_shape=y_train_arr.shape,
                                      filters_last_conv=1,
                                      stochastic=False)
elif uncertainty_approach == 'CRPS':
    model = deep.models.NoisyDeepESD(x_shape=x_train_stand_arr.shape, y_shape=y_train_arr.shape,
                                    filters_last_conv=1, num_channels_noise=3,
                                    last_relu=False,
                                    members_for_training=2)



# Wrap the model for multi-GPU training if available
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs!")
#     model = torch.nn.DataParallel(model)
# model.to(device)

# Set hyperparameters
num_epochs = 10000
learning_rate = 0.0001
patience_early_stopping = 40

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
# Train the model
# train_loss, val_loss = deep.train.standard_training_loop(model=model, 
#                                                             model_name=model_name, 
#                                                             model_path=MODELS_PATH,
#                                                             device=device, 
#                                                             num_epochs=num_epochs,
#                                                             loss_function=loss_function, 
#                                                             optimizer=optimizer,
#                                                             train_data=train_dataloader,
#                                                             valid_data=valid_dataloader,
#                                                             patience_early_stopping=patience_early_stopping,
#                                                             mixed_precision=True)

# Load the model weights into the architecture
# model.load_state_dict(torch.load(f'{MODELS_PATH}/{model_name}.pt', weights_only=True))


state_dict = torch.load(f'{MODELS_PATH}/{model_name}.pt', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=True)
model.to('cpu')
device = 'cpu'
# Standardize the test data
x_test_stand = trans.standardize(data_ref=x_train, data=x_test)

# Compute mask
y_mask = trans.compute_valid_mask(y_test)

# Compute predictions
if uncertainty_approach == 'ASYM' or uncertainty_approach == 'MSE':
    pred_test = deep.pred.compute_preds_standard(x_data=x_test_stand, model=model, device=device,
                                                              var_target='pr', mask=y_mask, batch_size=16)
elif uncertainty_approach == 'CRPS':
    pred_test = deep.pred.compute_preds_standard(x_data=x_test_stand, model=model, device=device,
                                                              var_target='pr', mask=y_mask, batch_size=16,ensemble_size=10)
# Save the predictions
pred_test.to_netcdf(f'{PREDS_PATH}/{model_name}.nc')


## SEGUNDA PARTE APLICAR LOS RESULTADOS PARA LOS GCMS: 

gcm_raw = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/CRPS-BasedEnsemble/GCMResults' 
gcm_hist_filename = f'{gcm_raw}/EC-Earth3_r1i1p1f1_historical.nc'
gcm_hist = xr.open_dataset(f'{gcm_raw}/EC-Earth3_r1i1p1f1_historical.nc').load()

gcm_fut_filename = f'{gcm_raw}/EC-Earth3_r1i1p1f1_ssp370.nc'
gcm_fut = xr.open_dataset(f'{gcm_raw}/EC-Earth3_r1i1p1f1_ssp370.nc').load()

gcmname = 'EC-Earth3' # EC-Earth3_r1i1p1f1

years_train = ('1980', '2010')
x_train = predictor.sel(time=slice(*years_train))

gcm_hist = gcm_hist.sel(lon=slice(-24, 22.5))
gcm_fut = gcm_fut.sel(lon=slice(-24, 22.5))

# Extend the latitude to 32 grid points
lat = gcm_hist.lat.values
dlat = np.diff(lat).mean()
n_extra =  32 - lat.size
new_lat = np.concatenate([lat, lat[-1] + dlat * np.arange(1, n_extra + 1)])
gcm_hist = gcm_hist.reindex(lat=new_lat, method='nearest')

lat = gcm_fut.lat.values
dlat = np.diff(lat).mean()
n_extra =  32 - lat.size
new_lat = np.concatenate([lat, lat[-1] + dlat * np.arange(1, n_extra + 1)])
gcm_fut = gcm_fut.reindex(lat=new_lat, method='nearest')

"""
Before feeding the data to the model, and as explained in the manuscript, GCM predictors are first bias-corrected
and then standardized.
"""

gcm_hist_corrected = trans.scaling_delta_correction(data=gcm_hist,
                                                    gcm_hist=gcm_hist, obs_hist=x_train)
gcm_fut_corrected = trans.scaling_delta_correction(data=gcm_fut,
                                                   gcm_hist=gcm_hist, obs_hist=x_train)

gcm_hist_corrected_stand = trans.standardize(data_ref=x_train, data=gcm_hist_corrected)
gcm_fut_corrected_stand = trans.standardize(data_ref=x_train, data=gcm_fut_corrected)

# Antes de predecir
gcm_hist_corrected_stand = gcm_hist_corrected_stand.astype('float32')
gcm_fut_corrected_stand  = gcm_fut_corrected_stand.astype('float32')



"""
Compute the projections for the historical and future periods in a manner similar to the predictions for the test
set, and save them to the file_name_hist and file_name_fut paths.
"""

if uncertainty_approach == 'ASYM' or uncertainty_approach == 'MSE':
    proj_historical = deep.pred.compute_preds_standard(x_data=gcm_hist_corrected_stand, model=model,
                                                        device=device, var_target='pr',
                                                        mask=y_mask, batch_size=16)
    proj_future = deep.pred.compute_preds_standard(x_data=gcm_fut_corrected_stand, model=model,
                                                    device=device, var_target='pr',
                                                    mask=y_mask, batch_size=16)
elif uncertainty_approach == 'CRPS':
    proj_historical = deep.pred.compute_preds_standard(x_data=gcm_hist_corrected_stand, model=model,
                                                        device=device, var_target='pr',
                                                        mask=y_mask, batch_size=16, ensemble_size=2)
    proj_future = deep.pred.compute_preds_standard(x_data=gcm_fut_corrected_stand, model=model,
                                                    device=device, var_target='pr',
                                                    mask=y_mask, batch_size=16, ensemble_size=2)

file_name_hist = f'GCM_proj_historical_{model_name}_{gcmname}'
file_name_fut = f'GCM_proj_future_{model_name}_{gcmname}'


PROJ_PATH = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/CRPS-BasedEnsemble/GCMResults'

proj_historical.to_netcdf(f'{PROJ_PATH}/{file_name_hist}.nc')
proj_future.to_netcdf(f'{PROJ_PATH}/{file_name_fut}.nc')



