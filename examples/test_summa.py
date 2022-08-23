import os
from hydromt_summa.summa import SummaModel
from hydromt import workflows
import xarray as xr
from hydromt.log import setuplog
import numpy as np

#%%
hydromt_test_catalogue = os.path.expanduser("~")+r'\.hydromt_data\data\v0.0.6\data_catalog.yml'

root = os.path.join('./dummy_summa')
mod = SummaModel(root=root, mode="w",
                  data_libs=[hydromt_test_catalogue,
                             './catalog/datacatalog_summa.yml'])

_region = {'subbasin': [[12.6,12.6,12.5], [45.8,46.2,45.95]], 'strord': 7, 'bounds': [12.1, 45.5, 12.9, 46.5]}

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
