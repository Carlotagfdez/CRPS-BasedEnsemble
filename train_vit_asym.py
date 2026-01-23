# Set paths
DATA_PATH = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/0.Datos'
FIGURES_PATH = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/CRPS-BasedEnsemble/Figuras'
MODELS_PATH = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/CRPS-BasedEnsemble/deep/models'
PREDS_PATH = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/CRPS-BasedEnsemble/Preds'
ASYM_PATH = '/gpfs/projects/meteo/WORK/garciafdez/data_PNACC/CRPS-BasedEnsemble/asym-parameters'

# Import libraries
import xarray as xr
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from importlib import reload

import sys; 
sys.path.append()

import viz
import trans
import deep.loss
import deep.utils
import deep.models
import deep.train
import deep.pred
import metrics
import metrics_ccs
import utils 

# Uncertainty approach
uncertainty_approach = 'ASYM'

# Set device
device = 'cuda'

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

# Fit the Gammas for the ASYM loss function
loss_function = deep.loss.Asym(ignore_nans=True, asym_path=ASYM_PATH)
if not loss_function.parameters_exist():
    loss_function.compute_parameters(data=y_train_stack,
                                     var_target='pr')
loss_function.load_parameters()
loss_function.prepare_parameters(device=device)

# Convert the data to numpy arrays
x_train_stand_arr = trans.xarray_to_numpy(x_train_stand)
y_train_arr = trans.xarray_to_numpy(y_train_stack)

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
model_name = f'vit_{uncertainty_approach}'

# Create model
model = deep.models.ViT(x_shape=x_train_stand_arr.shape,
                                         y_shape=y_train_arr.shape,
                                         patch_size=2,
                                         dim=768,
                                         depth=12,
                                         num_heads=12,
                                         mlp_dim=3072,
                                         orog=None,
                                         last_relu=True,
                                         stochastic=False)

# Wrap the model for multi-GPU training if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
model.to(device)

# Set hyperparameters
num_epochs = 10000
learning_rate = 0.0001
patience_early_stopping = 40

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
# Train the model
train_loss, val_loss = deep.train.standard_training_loop(model=model, 
                                                            model_name=model_name, 
                                                            model_path=MODELS_PATH,
                                                            device=device, 
                                                            num_epochs=num_epochs,
                                                            loss_function=loss_function, 
                                                            optimizer=optimizer,
                                                            train_data=train_dataloader,
                                                            valid_data=valid_dataloader,
                                                            patience_early_stopping=patience_early_stopping,
                                                            mixed_precision=True)

# Load the model weights into the architecture
model.load_state_dict(torch.load(f'{MODELS_PATH}/{model_name}.pt', weights_only=True))

# Standardize the test data
x_test_stand = trans.standardize(data_ref=x_train, data=x_test)

# Compute mask
y_mask = trans.compute_valid_mask(y_test)

# Compute predictions
pred_test = deep.pred.compute_preds_standard(x_data=x_test_stand, model=model, device=device,
                                                              var_target='pr', mask=y_mask, batch_size=16)

# Save the predictions
pred_test.to_netcdf(f'{PREDS_PATH}/{model_name}.nc')