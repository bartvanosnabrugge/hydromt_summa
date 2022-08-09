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

logger = logging.getLogger(__name__)


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
        **zonal_stats_kwargs    
    ):
        response_units = hydromt.workflows.ru_geometry_to_gpd(self.response_units)
        ds_forcing = self.data_catalog.get_rasterdataset(forcing_fn,geom=response_units)
        ds_zstats = ds_forcing.raster.zonal_stats(response_units,'mean',**zonal_stats_kwargs)
        ds_zstats['index'] = (["index"], response_units['value'])
        self.set_forcing(ds_zstats, name="forcing")
        
    def setup_soilclass(
        self,
        soilclass_fn,
        **zonal_stats_kwargs
    ):
        rus =  hydromt.workflows.ru_geometry_to_gpd(self.response_units)
        ds_soil = self.data_catalog.get_rasterdataset(soilclass_fn,geom=rus)
        ds_zstats = ds_soil.raster.zonal_stats_per_class(rus,np.unique(ds_soil),
                        stat='count', class_dim_name='soil')
        ds_zstats['index'] = (["index"], rus['value'])
        # and calculate fractions
        fracs = hydromt.workflows.fracs(ds_zstats,'soil_classes_count','soil')
        soil_mode = hydromt.workflows.ds_class_mode(ds_zstats,'soil_classes_count','soil')
        
        self.set_response_units(fracs, name='soil_fraction')
        self.set_response_units(soil_mode, name='soil_mode')
        
    def setup_landclass(
        self,
        landclass_fn,
        **zonal_stats_kwargs
    ):
        rus =  hydromt.workflows.ru_geometry_to_gpd(self.response_units)
        ds_class = self.data_catalog.get_rasterdataset(landclass_fn,geom=rus)
        ds_zstats = ds_class.raster.zonal_stats_per_class(rus,np.unique(ds_class),
                        stat='count', class_dim_name='landclass')
        ds_zstats['index'] = (["index"], rus['value'])
        # and calculate fractions
        fracs = hydromt.workflows.fracs(ds_zstats,'land_classes_count','landclass')
        soil_mode = hydromt.workflows.ds_class_mode(ds_zstats,'land_classes_count','landclass')
        
        self.set_response_units(fracs, name='landclass_fraction')
        self.set_response_units(soil_mode, name='landclass_mode')
    
    def setup_elevation(
        self,
        hydrography_fn='merit_hydro',
        **zonal_stats_kwargs
    ):
        rus =  hydromt.workflows.ru_geometry_to_gpd(self.response_units)
        ds_hyd = self.data_catalog.get_rasterdataset(hydrography_fn,geom=self.region)
        mdem = ds_hyd['elevtn']
        zstats = mdem.raster.zonal_stats(rus,'mean')
        zstats['index'] = (["index"], rus['value'])
        self.set_response_units(zstats, name='elevtn')
    
    def setup_states():
        pass
    
    def setup_config():
        pass
    
    def copy_base_files(self,base_settings_path):
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
        path_to_forcing = os.path.join(self.root) # TODO: make configurable
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

    def read(self):
        """Method to read the complete model schematization and configuration from file."""
        self.logger.info(f"Reading model data from {self.root}")
        self.read_config()
        self.read_staticmaps()
        self.read_staticgeoms()

    def write(self):
        """Method to write the complete model schematization and configuration to file."""
        self.logger.info(f"Writing model data to {self.root}")
        # if in r, r+ mode, only write updated components
        if not self._write:
            self.logger.warning("Cannot write in read-only mode")
            return
        if self.config:  # try to read default if not yet set
            self.write_config()
        if self._response_units:
            self.write_response_units()
        if self._staticgeoms:
            self.write_staticgeoms()
        if self._forcing:
            self.write_forcing()

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
        super().write_forcing
        # then also create forcing_file_list.txt
        # raise NotImplementedError()

    def read_states(self):
        """Read states at <root/?/> and parse to dict of xr.DataArray"""
        return self._states
        # raise NotImplementedError()

    def write_states(self):
        """write states at <root/?/> in model ready format"""
        pass
        # raise NotImplementedError()

    def read_results(self):
        """Read results at <root/?/> and parse to dict of xr.DataArray"""
        return self._results
        # raise NotImplementedError()

    def write_results(self):
        """write results at <root/?/> in model ready format"""
        pass
        # raise NotImplementedError()

    @property
    def crs(self):
        return pyproj.CRS.from_epsg(self.get_config("global.epsg", fallback=4326))
