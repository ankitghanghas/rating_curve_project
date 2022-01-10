# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 12:08:48 2020


@author: aghangha
"""

import arcpy
import os
import pandas as pd


arcpy.env.workspace = r"D:\ratingcurve\DEM"
arcpy.env.overwriteOutput = True
pd.options.mode.chained_assignment=None

def projectraster(inputfile):
    outfolder = r"D:\ratingcurve\DEM\prj"
    a=r"C:\Users\aghangha\Documents\ratingcurve\gages_shapefile\navd88\gages_us.shp"
    outcs=arcpy.Describe(a).spatialReference
    outfile = os.path.join(outfolder,'mid_dem_10m.tif')
    arcpy.ProjectRaster_management(in_raster=inputfile,
                                   out_raster=outfile,
                                   out_coor_system = outcs,
                                   resampling_type="Nearest",
                                   cell_size=10
                                   )








#################################################################

inputfile= r"D:\ratingcurve\DEM\mos\mid_10m_dem.tif"
projectraster(inputfile)




