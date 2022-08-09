import os
from hydromt_summa.summa import SummaModel

import hydromt
import xarray as xr
from hydromt.log import setuplog
import numpy as np
import os
from scipy import stats
import geopandas as gpd

import HRU as hru
#import meshclasses as mesh

#%%
root = os.path.join('./dummy_mcmesh')
mod = SummaModel(root=root, mode="w",
                  data_libs=[r'c:\Users\osnabrug\.hydromt_data\data\v0.0.6\data_catalog.yml',
                             './catalog/datacatalog_summa.yml'])

#_region = {"subbasin":[-115.5722,51.01174]}
_region = {'subbasin': [[12.6,12.6,12.5], [45.8,46.2,45.95]], 'strord': 7, 'bounds': [12.1, 45.5, 12.9, 46.5]}

# base response_unit geometry
r = mod.setup_response_unit_geom(
    region=_region,
    hydrography_fn="merit_hydro",
    basin_index_fn="merit_hydro_index",
    split_regions = True,        
    split_method = "streamorder",
    min_sto = 8,
    mask = None
)

# add downstream links
dl = mod.setup_downstream_links(mod.geoms['response_unit']['outlet_geometry'])
# plot response_unit geometries
mod.geoms['response_unit'].plot(column='value',edgecolor='black',categorical=True)

response_units = mod.geoms['response_unit']
#%% remap era5 forcing data
ds_era5 = mod.data_catalog.get_rasterdataset('era5_landsurfacemodel_params',geom=mod.region)
zstats_era5 = ds_era5.raster.zonal_stats(response_units,'mean',all_touched=True)

#%% remap soil class
# open usda soilclass
ds_soil = mod.data_catalog.get_rasterdataset('usda_soilclass',geom=mod.region)

#gpd_basins_soilclass = count_per_class(ds_soil,gpd_basins,range(13),class_abbrev='USGS_')
ds_soil_zstats = ds_soil.raster.zonal_stats_per_class(response_units,np.unique(ds_soil),
                 stat='count', class_dim_name='soil'                                     )

#%% count modis landclass
ds_modis = mod.data_catalog.get_rasterdataset('modis_landclass_mode',geom=mod.region)
classes = np.unique(ds_modis)
modis_zstats = ds_modis.raster.zonal_stats_per_class( response_units, classes, stat='count' ,class_dim_name='landcover')


#%% remap merit dem
# open merit dem
ds_hyd = mod.data_catalog.get_rasterdataset('merit_hydro',geom=mod.region)
mdem = ds_hyd['elevtn']
zstats = mdem.raster.zonal_stats(response_units,'mean')

#%% create static_properties_db
ds = xr.Dataset(
    data_vars=dict(
        basin_id=(["index"],response_units['value'].astype('int')),
        basin_geometry_str = (["index"], [str(g) for g in response_units['geometry']]),
        outlet_geometry_str = (["index"], [str(g) for g in response_units['outlet_geometry']])
    ),
    #coords=dict(
    #    basin_id=(["subbasin"],response_units['value'].astype('int'))
    #)
)
ds.to_netcdf('test.nc')

#%%
ds2 = xr.Dataset(
    data_vars=dict(
        basin_id=(["index"],response_units['value']),
    ),
    coords=dict(
        basin_geometry = (["index"], response_units['geometry']),
        outlet_geometry = (["index"], response_units['outlet_geometry'])
    ),
)
ds2['index']=(["index"],response_units['value'])
#%% add zstats variables (single dimension)
for var in list(zstats.keys()):
    ds2[var] = (["index"],zstats[var].data)
    
#%% setup GRUs
#!!!! TODO: make sure IDs match! some take the index other now the 'value'!!!
ds3 = xr.merge([ds2,ds_soil_zstats,modis_zstats])#,zstats_era5])
ds4 = xr.merge([ds3,zstats_era5])

ds3['land_mode']= (["index"]  , 
                 [ds3.landcover.values[i] for i in ds3['land_classes_count'].argmax(dim='landcover').data])


ds3['land_fractions']= ds3['land_classes_count'] / ds3['land_classes_count'].sum(dim='landcover') 