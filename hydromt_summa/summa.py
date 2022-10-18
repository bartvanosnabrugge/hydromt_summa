"""Implement plugin model class"""
# FIXME implement model class following model API

import glob
import os
from os.path import join, basename
import logging
import pyproj
import geopandas as gpd
import xarray as xr
import numpy as np
from scipy import stats

import pandas as pd
import shutil
import hydromt
from hydromt.models.model_lumped import LumpedModel
from hydromt import gis_utils, io
from . import workflows, DATADIR

import pysumma as ps

logger = logging.getLogger(__name__)

# helper functions
# TODO: save these helper functions somewhere else

import itertools


def split_by_chunks(dataset):
    '''Split dataset in multiple datasets by chunk
    
    Notes
    ------
    from https://ncar.github.io/esds/posts/2020/writing-multiple-netcdf-files-in-parallel-with-xarray-and-dask/
    '''
    chunk_slices = {}
    for dim, chunks in dataset.chunks.items():
        slices = []
        start = 0
        for chunk in chunks:
            if start >= dataset.sizes[dim]:
                break
            stop = start + chunk
            slices.append(slice(start, stop))
            start = stop
        chunk_slices[dim] = slices
    for slices in itertools.product(*chunk_slices.values()):
        selection = dict(zip(chunk_slices.keys(), slices))
        yield dataset[selection]

def create_filepath(ds, prefix='filename', root_path="."):
    """
    Generate a filepath when given an xarray dataset
    """
    start = pd.to_datetime(ds.time.data[0]).strftime("%Y-%m-%d-%H-%M-%S")
    end = pd.to_datetime(ds.time.data[-1]).strftime("%Y-%m-%d-%H-%M-%S")
    filepath = f'{root_path}/{prefix}_{start}_{end}.nc'
    return filepath

def get_timestep(ds):
    dift = ds['time'][1].values-ds['time'][0].values
    secs = int(dift/np.timedelta64(1,'s'))
    return secs

class SummaModel(LumpedModel):
    """This is the class for the SUMMA hydrological model in HydroMT"""

    # FIXME
    _NAME = "summa"
    _CONF = "summa_to_run_themodel.ini"
    _DATADIR = DATADIR
    _GEOMS = {}
    _MAPS = {}
    _FORCING = {}
    _FOLDERS = ["response_units","output"]
    _MODELFILES = []


    def __init__(
        self,
        root=None,
        mode="w",
        config_fn=None,
        data_libs=None,
        logger=logger,
    ):
        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )
        # initialize specific options
        

    ## components
    def setup_forcing(
        self,
        forcing_fn,
        get_rasterdataset_kwargs = {},
        **zonal_stats_kwargs    
    ):
        response_units = hydromt.workflows.ru_geometry_to_gpd(self.response_units)
        ds_forcing = self.data_catalog.get_rasterdataset(forcing_fn,geom=self.region,buffer=2,**get_rasterdataset_kwargs)
        ds_zstats = ds_forcing.raster.zonal_stats(response_units,'mean',**zonal_stats_kwargs)
        ds_zstats['index'] = (["index"], response_units['value'])
        self.set_forcing(ds_zstats, name="forcing")
        
    def setup_soilclass(
        self,
        soilclass_fn,
        **zonal_stats_kwargs
    ):
        rus =  hydromt.workflows.ru_geometry_to_gpd(self.response_units)
        ds_soil = self.data_catalog.get_rasterdataset(soilclass_fn,geom=self.region,buffer=2)
        ds_zstats = ds_soil.raster.zonal_stats_per_class(rus,np.unique(ds_soil),
                        stat='count', class_dim_name='soil')
        ds_zstats['index'] = (["index"], rus['value'])
        # and calculate fractions
        fracs = hydromt.workflows.fracs(ds_zstats,'tax_usda_count','soil')
        soil_mode = hydromt.workflows.ds_class_mode(ds_zstats,'tax_usda_count','soil')
        
        self.set_response_units(ds_zstats,name='tax_usda_count')
        self.set_response_units(fracs, name='tax_usda_fraction')
        self.set_response_units(soil_mode, name='tax_usda_mode')
        
    def setup_landclass(
        self,
        landclass_fn,
        **zonal_stats_kwargs
    ):
        rus =  hydromt.workflows.ru_geometry_to_gpd(self.response_units)
        ds_land = self.data_catalog.get_rasterdataset(landclass_fn,geom=rus,buffer=2)
        ds_land_stack = ds_land.to_stacked_array(new_dim='years',sample_dims=['x','y'])
        # calculate mode across years dimension, stored in np.array
        modestat = stats.mode(ds_land_stack,axis=2,nan_policy='omit')[0].squeeze()
        # create data array from np.array
        ds_landclass_mode = xr.DataArray(modestat,{'y':ds_land.y,'x': ds_land.x})
        ds_landclass_mode.name = 'landclass'
        # generate class list from classes in array
        class_list = np.unique(ds_landclass_mode)
        
        ds_zstats = ds_landclass_mode.raster.zonal_stats_per_class(rus,class_list,
                        stat='count', class_dim_name='landclass')
        ds_zstats['index'] = (["index"], rus['value'])
        # and calculate fractions
        fracs = hydromt.workflows.fracs(ds_zstats,'landclass_count','landclass')
        soil_mode = hydromt.workflows.ds_class_mode(ds_zstats,'landclass_count','landclass')
        
        self.set_response_units(fracs, name='landclass_fraction')
        self.set_response_units(soil_mode, name='landclass_mode')
    
    def setup_elevation(
        self,
        hydrography_fn='merit_hydro',
        **zonal_stats_kwargs
    ):
        rus =  hydromt.workflows.ru_geometry_to_gpd(self.response_units)
        ds_hyd = self.data_catalog.get_rasterdataset(hydrography_fn,geom=self.region,buffer=2)
        mdem = ds_hyd['elevtn']
        zstats = mdem.raster.zonal_stats(rus,'mean')
        zstats['index'] = (["index"], rus['value'])
        self.set_response_units(zstats, name='elevtn')
    
    def setup_states():
        pass
    
    def setup_config():
        pass
    
    def copy_base_files(self,base_settings_path=None):
        if not base_settings_path:
            base_settings_path = os.path.join(self._DATADIR,'base_settings')
        for f in os.listdir(os.path.join(base_settings_path)):
            shutil.copyfile(os.path.join(base_settings_path,f),
                            os.path.join(self.root,f))
    
    def write_filemanager(self):
        ds_f1 = self.forcing[list(self.forcing.keys())[0]]
        ts = pd.to_datetime(ds_f1.time[0].values)
        sim_start = ts.strftime('%Y-%m-%d %H:%M') # TODO: make also configurable
        
        ts = pd.to_datetime(ds_f1.time[-1].values)
        sim_end = ts.strftime('%Y-%m-%d %H:%M') # TODO: make also configurable
        
        experiment_id = 'test_hydromt_summa' # TODO: make configurable
        path_to_settings = os.path.join(self.root)
        path_to_forcing = os.path.join(self.root,'forcing') # TODO: make configurable
        path_to_output = os.path.join(self.root,'output') # TODO: make configurable
        
        initial_conditions_nc = 'coldState.nc'
        attributes_nc = 'attributes.nc'
        trial_parameters_nc = 'trialParams.nc'
        forcing_file_list_txt = 'forcingFileList.txt'
        
        with open(os.path.join(self.root,'fileManager.txt'), 'w') as fm:    
            # Header
            fm.write("controlVersion       'SUMMA_FILE_MANAGER_V3.0.0' !  file manager version \n")
            
            # Simulation times
            fm.write("simStartTime         '{}' ! \n".format(sim_start))
            fm.write("simEndTime           '{}' ! \n".format(sim_end))
            fm.write("tmZoneInfo           'utcTime' ! \n")
            
            # Prefix for SUMMA outputs
            fm.write("outFilePrefix        '{}' ! \n".format(experiment_id))
            
            # Paths
            fm.write("settingsPath         '{}/' ! \n".format(path_to_settings))
            fm.write("forcingPath          '{}/' ! \n".format(path_to_forcing))
            fm.write("outputPath           '{}/' ! \n".format(path_to_output))
            
            # Input file names
            fm.write("initConditionFile    '{}' ! Relative to settingsPath \n".format(initial_conditions_nc))
            fm.write("attributeFile        '{}' ! Relative to settingsPath \n".format(attributes_nc))
            fm.write("trialParamFile       '{}' ! Relative to settingsPath \n".format(trial_parameters_nc))
            fm.write("forcingListFile      '{}' ! Relative to settingsPath \n".format(forcing_file_list_txt))
            
            # Base files (not domain-dependent)
            fm.write("decisionsFile        'modelDecisions.txt' !  Relative to settingsPath \n")
            fm.write("outputControlFile    'outputControl.txt' !  Relative to settingsPath \n")
            fm.write("globalHruParamFile   'localParamInfo.txt' !  Relative to settingsPath \n")
            fm.write("globalGruParamFile   'basinParamInfo.txt' !  Relative to settingsPatho \n")
            fm.write("vegTableFile         'TBL_VEGPARM.TBL' ! Relative to settingsPath \n")
            fm.write("soilTableFile        'TBL_SOILPARM.TBL' ! Relative to settingsPath \n")
            fm.write("generalTableFile     'TBL_GENPARM.TBL' ! Relative to settingsPath \n")
            fm.write("noahmpTableFile      'TBL_MPTABLE.TBL' ! Relative to settingsPath \n")        

    ## I/O 

    def read(self,summa_exe='summa.exe',config_dir=''):
        """Method to read the complete model schematization and configuration from file."""
        self.logger.info(f"Reading model data from {self.root}")
        #self.read_config()
        #self.read_staticmaps()
        #self.read_staticgeoms()
        # use pysumma simulation class to read all text files
        self.Simulation = ps.Simulation(summa_exe,os.path.join(self.root,'fileManager.txt'),config_dir=config_dir)

    def write(self):
        """Method to write the complete model schematization and configuration to file."""
        self.logger.info(f"Writing model data to {self.root}")
        # if in r, r+ mode, only write updated components
        if not self._write:
            self.logger.warning("Cannot write in read-only mode")
            return
        if self._response_units:
            self.write_response_units()
        if self._forcing:
            self.write_forcing()
        if hasattr(self,'Simulation'):
            self.Simulation._write_configuration()
        else:
            self.copy_base_files()
            self.write_filemanager()

    def read_staticgeoms(self):
        """Read staticgeoms at <root/?/> and parse to dict of geopandas"""
        if not self._write:
            # start fresh in read-only mode
            self._staticgeoms = dict()
        for fn in glob.glob(join(self.root, "*.xy")):
            name = basename(fn).replace(".xy", "")
            geom = hydromt.open_vector(fn, driver="xy", crs=self.crs)
            self.set_staticgeoms(geom, name)

    def write_staticgeoms(self):
        """Write staticmaps at <root/?/> in model ready format"""
        # to write use self.staticgeoms[var].to_file()
        if not self._write:
            raise IOError("Model opened in read-only mode")
        for name, geom in self.staticgeoms.items():
            fn_out = join(self.root, f"{name}.xy")
            io.write_xy(fn_out, self.staticgeoms[name])

    def read_forcing(self):
        """Read forcing at <root/?/> and parse to dict of xr.DataArray"""
        return self._forcing
        # raise NotImplementedError()

    def write_forcing(self):
        """write forcing at <root/?/> in model ready format"""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        elif not self._forcing:
            self.logger.warning("No model forcing to write - Exiting")
            return
        else:
            self.logger.info("Write model forcing files")

        fn = os.path.join(self.root, "forcing")
        if not os.path.isdir(fn):
            os.makedirs(fn)
        # prepare dataset for writing    
        ds = xr.Dataset(self.forcing)

        ds = ds.rename_dims({'index':'hru'})
        ds = ds.rename_vars({'index':'hru'})
        # drop the hru coordinate
        # ds = ds.drop_vars('hru')
        
        # transpose to adhere to (time,hru) order
        ds = ds.transpose()
        
        # rename variables to SUMMA standard
        # TODO: rename from hydromt standards
        new_names = []
        for n in list(self.forcing):
            if '_' in n:
                n = n.split('_')[0] # strip of any zonal stats addendums to name
            new_names.append(n)
        ds = ds.rename_vars(dict(zip(self.forcing,new_names)))
        
        # TODO: implement sorting based on GRUs and HRUs - solution sort shapes first
        
        # add data_step variable
        ds['data_step'] = get_timestep(ds)
        
        # add hruid variable
        ds['hruId'] = ds['hru'].astype(np.float64)
        
        
        # write forcing, over multiple files depending on chunking
        # keep only chunks in time
        ds = ds.chunk({'hru':len(ds.hru)})
        
        # then write file for each chunk
        datasets = list(split_by_chunks(ds))        
        paths = [create_filepath(ds,'forcing',fn) for ds in datasets]
        xr.save_mfdataset(datasets=datasets, paths=paths,engine="netcdf4") # on GRAHAM currently only works with hdf5 engine
        
        # then also create forcing_file_list.txt
        with open(os.path.join(self.root,'forcingFileList.txt'), 'w') as ffl:
            for ff in paths:
                ffl.write(os.path.basename(ff)+'\n')

    def read_states(self):
        """Read states at <root/?/> and parse to dict of xr.DataArray"""
        return self._states
        # raise NotImplementedError()

    def write_states(self):
        """write states at <root/?/> in model ready format"""
        # --- Define the dimensions and fill values
        # from CWARHM: Knoben et al. 2022.
        # Specify the dimensions
        nSoil   = 8         # number of soil layers
        nSnow   = 0         # assume no snow layers currently exist
        midSoil = 8         # midpoint of soil layer
        midToto = 8         # total number of midpoints for snow+soil layers
        ifcToto = midToto+1 # total number of layer boundaries
        scalarv = 1         # auxiliary dimension variable

        # Layer variables
        mLayerDepth  = np.asarray([0.025, 0.075, 0.15, 0.25, 0.5, 0.5, 1, 1.5])
        iLayerHeight = np.asarray([0, 0.025, 0.1, 0.25, 0.5, 1, 1.5, 2.5, 4])

        # States
        scalarCanopyIce      = 0      # Current ice storage in the canopy
        scalarCanopyLiq      = 0      # Current liquid water storage in the canopy
        scalarSnowDepth      = 0      # Current snow depth
        scalarSWE            = 0      # Current snow water equivalent
        scalarSfcMeltPond    = 0      # Current ponded melt water
        scalarAquiferStorage = 1.0    # Current aquifer storage
        scalarSnowAlbedo     = 0      # Snow albedo
        scalarCanairTemp     = 283.16 # Current temperature in the canopy airspace
        scalarCanopyTemp     = 283.16 # Current temperature of the canopy 
        mLayerTemp           = 283.16 # Current temperature of each layer; assumed that all layers are identical
        mLayerVolFracIce     = 0      # Current ice storage in each layer; assumed that all layers are identical
        mLayerVolFracLiq     = 0.2    # Current liquid water storage in each layer; assumed that all layers are identical
        mLayerMatricHead     = -1.0   # Current matric head in each layer; assumed that all layers are identical
        
        # get timestep from forcing
        sample_forcing = self.forcing[list(self.forcing)[0]]
        dt_init = get_timestep(sample_forcing)
        # get hruIds (indexes)
        hruIds = self.response_units.index.values.astype(int)
        num_hru = len(hruIds)
        
        dsinit = xr.Dataset(
            data_vars=dict(
                hruId=(["hru"],hruIds),
                dt_init=(["scalarv","hru"],np.full((scalarv,num_hru),dt_init).astype('float64')),
                nSoil=(["scalarv","hru"],np.full((scalarv,num_hru),nSoil)),
                nSnow=(["scalarv","hru"],np.full((scalarv,num_hru),nSnow)),
                scalarCanopyIce=(["scalarv","hru"],np.full((scalarv,num_hru),scalarCanopyIce).astype('float64')),
                scalarCanopyLiq=(["scalarv","hru"],np.full((scalarv,num_hru),scalarCanopyLiq).astype('float64')),
                scalarSnowDepth=(["scalarv","hru"],np.full((scalarv,num_hru),scalarSnowDepth).astype('float64')),
                scalarSWE=(["scalarv","hru"],np.full((scalarv,num_hru),scalarSWE).astype('float64')),
                scalarSfcMeltPond=(["scalarv","hru"],np.full((scalarv,num_hru),scalarSfcMeltPond).astype('float64')),
                scalarAquiferStorage=(["scalarv","hru"],np.full((scalarv,num_hru),scalarAquiferStorage).astype('float64')),
                scalarSnowAlbedo=(["scalarv","hru"],np.full((scalarv,num_hru),scalarSnowAlbedo).astype('float64')),
                scalarCanairTemp=(["scalarv","hru"],np.full((scalarv,num_hru),scalarCanairTemp).astype('float64')),
                scalarCanopyTemp=(["scalarv","hru"],np.full((scalarv,num_hru),scalarCanopyTemp).astype('float64')),
                
                mLayerTemp=(["midToto","hru"],np.full((midToto,num_hru),mLayerTemp).astype('float64')),
                mLayerVolFracIce=(["midToto","hru"],np.full((midToto,num_hru),mLayerVolFracIce).astype('float64')),
                mLayerVolFracLiq=(["midToto","hru"],np.full((midToto,num_hru),mLayerVolFracLiq).astype('float64')),
                mLayerDepth=(["midToto","hru"],np.stack([mLayerDepth for i in range(num_hru)],axis=1).astype('float64')),
                
                mLayerMatricHead=(["midSoil","hru"],np.full((midSoil,num_hru),mLayerMatricHead).astype('float64')),
                
                iLayerHeight=(["ifcToto","hru"],np.stack([iLayerHeight for i in range(num_hru)],axis=1).astype('float64')),
            )
        )
        
        dsinit.to_netcdf(os.path.join(self.root,'coldState.nc'))

    def write_trial_params(self):
        hruIds = self.response_units.index.values.astype(int)
        num_hru = len(hruIds)
        dstrial = xr.Dataset(
            data_vars=dict(
                hruId = (["hru"],self.response_units.index.values.astype(int)),
                maxstep = (["hru"],np.full(num_hru,900).astype('float64'))
            )
        )
        dstrial.to_netcdf(os.path.join(self.root,'trialParams.nc'))
    
    def write_attributes(self):
        tan_slope = 0.1 # [-] TODO: make configurable
        contourLength = 30 # [m] TODO; make configurable
        slopeTypeIndex = 1 # [-] TODO: make configurable
        mHeight = 3 # [m] TODO: make configurable
        
        hruIds = self.response_units.index.values.astype(int)
        num_hru = len(hruIds)
        
        # extract geometries to gdf
        gdf = hydromt.workflows.ru_geometry_to_gpd(self.response_units)
        
        dsattr = xr.Dataset(
            data_vars=dict(
                hruId=(["hru"],hruIds),
                gruId=(["gru"],hruIds), # TODO: work out example with different GRUS and HRUS (Bow at Banff example)
                hru2gruId=(["hru"],hruIds),
                downHRUindex=(["hru"],self.response_units.down_id.values.astype(int)),
                longitude=(["hru"],gdf.centroid.x),
                latitude=(["hru"],gdf.centroid.y),
                elevation=(["hru"],self.response_units.elevtn_mean.values),
                HRUarea=(["hru"],gdf['geometry'].to_crs({'proj':'cea'}).area),
                tan_slope=(["hru"],np.full(num_hru,tan_slope).astype('float64')),
                contourLength=(["hru"],np.full(num_hru,contourLength).astype('float64')),
                slopeTypeIndex=(["hru"],np.full(num_hru,slopeTypeIndex).astype('int')),
                soilTypeIndex=(["hru"],self.response_units.tax_usda_count_mode.values.astype('int')),
                vegTypeIndex=(["hru"],self.response_units.landclass_count_mode.values.astype('int')),
                mHeight=(["hru"],np.full(num_hru,mHeight).astype('float64')),
            )
        )
        dsattr.to_netcdf(os.path.join(self.root,'attributes.nc'))
        
    def read_results(self):
        """Read results at <root/?/> and parse to dict of xr.DataArray"""
        return self._results
        # raise NotImplementedError()

    def write_results(self):
        """write results at <root/?/> in model ready format"""
        pass
        # raise NotImplementedError()
    
    def run(self, summa_bin):
        summa_bin = summa_bin
        fileman = os.path.join(self.root,'fileManager.txt')
        ex_string = "{} -m {}".format(summa_bin, fileman)
        print(" executable string is {}".format(ex_string))
        os.system(ex_string)

       
        
    @property
    def crs(self):
        return pyproj.CRS.from_epsg(self.get_config("global.epsg", fallback=4326))
