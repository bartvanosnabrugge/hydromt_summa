"""Implement plugin model class"""
# FIXME implement model class following model API

import glob
from os.path import join, basename
import logging
import pyproj
import geopandas as gpd
import xarray as xr

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
    _FOLDERS = ["output"]
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

    def setup_basemaps(self, region, res=1000, crs="utm", basemaps_fn="merit_hydro"):
        """BOILERPLATE FUNCTION Define model region and geometries.

        Adds model layers:

        * **dem** map: elevation [m+ref]
        * **basins** geom: basins or HRU shape vector

        Parameters
        ----------
        region : dict
            Dictionary describing region of interest, e.g. {'bbox': [xmin, ymin, xmax, ymax]}
            See :py:meth:`~hydromt.cli.parse_region()` for all options
        res : float
            Model resolution [m], by default 100 m.
        crs : str, int
            Model Coordinate Reference System as epsg code, by default 'utm' in which
            case the region centroid UTM zone is used.
        basemaps_fn : str
            Path or data source name for hydrography raster data, by default 'hydro_merit'.

            * Required variables: ['elevtn'].
        """
        # read data (lazy!) and return dataset
        ds_org = self.data_catalog.get_rasterdataset(
            basemaps_fn, single_var_as_array=False, variables=["elevtn"]
        )
        geom = region.get("geom", None)
        bbox = region.get("bbox", None)
        if geom is None and bbox is None:
            self.logger.error("Plugin model requires a 'bbox' or 'geom' region.")
        # parse dst_crs. if 'utm' the best utm zone is calculated based on bbox_epsg4326
        bbox_epsg4326 = bbox if bbox is not None else geom.to_crs(4326).total_bounds
        dst_crs = gis_utils.parse_crs(crs, bbox_epsg4326)
        self.set_config("global.epsg", dst_crs.to_epsg())
        # transfrom bbox/geom to geom with destination CRS to deal with nonlinear
        # transformations along domain edges when clipping

        if geom is not None:
            # to epsg required to be understood when writing GEOJSON
            dst_geom = geom.to_crs(dst_crs.to_epsg())
        else:
            dst_bbox = transform_bounds(pyproj.CRS.from_epsg(4326), dst_crs, *bbox)
            dst_geom = gpd.GeoDataFrame(geometry=[box(*dst_bbox)], crs=dst_crs)
        # reproject to destination CRS and clip to actual extent
        da_elv_org = ds_org["elevtn"].raster.clip_geom(geom=dst_geom, buffer=5)
        da_elv_proj = da_elv_org.raster.reproject(
            dst_res=res, dst_crs=dst_crs, align=True, method="average"
        )
        da_elv = da_elv_proj.raster.clip_geom(dst_geom, align=res)
        # set elevation map
        self.set_staticmaps(da_elv, self._MAPS["elevtn"])
    
    def setup_basins(self, basins_fn, **kwargs):
        """Setup model basin geometries

        Adds model layers:

        * **basins** geom: basin geometries

        Parameters
        ----------
        basins_fn: str
            Path to basin geometry file.
            See :py:meth:`~hydromt.open_vector`, for accepted files.
        """
        name = self._GEOMS["geom"]     
        gdf = self.data_catalog.get_geodataframe(
            basins_fn, geom=self.region, **kwargs
            ).to_crs(self.crs)
        self.set_staticgeoms(gdf,name)
        self.logger.info(f"{name} set based on {basins_fn}")

    def setup_drainage_db():
        # create SUMMA topology
        mcl.generate_mesh_topology(control_options['river_network_shp_path'], 
            control_options['river_basin_shp_path'],
            drain_db_path,
            control_options['settings_make_outlet'])
        ranks, drain_db = mcl.reindex_topology_file(drain_db_path)


    def setup_gauges(self, gauges_fn=None, **kwargs):
        """BOILERPLATE FUNCTION Setup model observation point locations.

        Adds model layers:

        * **obs** geom: observation point locations

        Parameters
        ---------
        gauges_fn: str, optional
            Path to observation points geometry file.
            See :py:meth:`~hydromt.open_vector`, for accepted files.
        """
        if gauges_fn is not None:
            name = self._GEOMS["gauges"]
            kwargs.update(assert_gtype="Point")
            gdf = self.data_catalog.get_geodataframe(
                gauges_fn, geom=self.region, **kwargs
            ).to_crs(self.crs)
            self.set_staticgeoms(gdf, name)
            self.set_config(f"{name}.{name}", f"{name}.xy")
            self.logger.info(f"{name} set based on {gauges_fn}")

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
        if self._staticmaps:
            self.write_staticmaps()
        if self._staticgeoms:
            self.write_staticgeoms()
        if self._forcing:
            self.write_forcing()

    def read_staticmaps(self):
        """Read staticmaps at <root/?/> and parse to xarray Dataset"""
        # to read gdal raster files use: hydromt.open_mfraster()
        # to read netcdf use: xarray.open_dataset()
        if not self._write:
            # start fresh in read-only mode
            self._staticmaps = xr.Dataset()
        self.set_staticmaps(hydromt.open_mfraster(join(self.root, "*.tif")))

    def write_staticmaps(self):
        """Write staticmaps at <root/?/> in model ready format"""
        # to write to gdal raster files use: self.staticmaps.raster.to_mapstack()
        # to write to netcdf use: self.staticmaps.to_netcdf()
        if not self._write:
            raise IOError("Model opened in read-only mode")
        self.staticmaps.raster.to_mapstack(self.root)

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
        pass
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
