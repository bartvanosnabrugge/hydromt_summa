import os
from hydromt_summa.summa import SummaModel
from hydromt import workflows
import xarray as xr
from hydromt.log import setuplog
import numpy as np

#%%
data_catalogue_path = os.path.join(os.path.expanduser("~"),'projects','rpp-kshook','CompHydCore','datacatalog.yml')

root = os.path.join('./dummy_summa')
mod = SummaModel(root=root, mode="w",
                  data_libs=[data_catalogue_path])

bbox = [-116.55,50.95,-115.52,51.74]
_region = {'subbasin': [[-115.53],[50.964]], 'uparea': 200, 'bounds': bbox}

# base response_unit geometry
r = mod.setup_region(_region)

mod.setup_response_unit(
    hydrography_fn="merit_hydro",
    split_regions = True,        
    split_method = "streamorder",
    min_sto = 8,
    mask = None
)

# add downstream links
dl = mod.setup_downstream_links()
# plot response_unit geometries
workflows.ru_geometry_to_gpd(
    mod.response_units).plot(column='value',edgecolor='black',categorical=True)

mod.setup_forcing('era5_landsurfacemodel_params',all_touched=True)
mod.setup_soilclass('usda_soilclass')
mod.setup_landclass('modis_landclass_mode')
mod.setup_elevation()

mod.write_response_units()

#%% write model specific files
mod.write_filemanager()
mod.copy_base_files()
mod.write_forcing()
mod.write_states()
mod.write_trial_params()
mod.write_attributes()
