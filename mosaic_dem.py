# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:59:33 2020
Name: Mosaic DEM




@author: aghangha
"""

import arcpy
import os
import pandas as pd


arcpy.env.workspace = r"D:\ratingcurve\DEM"
arcpy.env.overwriteOutput = True
pd.options.mode.chained_assignment=None


def main():
    work_folder = r"D:\ratingcurve\DEM"
    tiles_path = os.path.join(work_folder,'tiles','mid') 
    subfolders = [ f.name for f in os.scandir(tiles_path) if f.is_dir()] 
    names = [f if len(f)<8 else f[12:19] for f in subfolders]
    path=[tiles_path + "\\" + x + "\\grd" + y +"_13" for x,y in zip(subfolders,names)]
    outcs=outcs=arcpy.SpatialReference(4269,5703)
    for item in path:
        arcpy.DefineProjection_management(item,outcs) # first define projeciton for all the dem including their vertical coordinate system.
        # here we define projection as fcs nad 83 and navd 88 vcs
    output_path=os.path.join(work_folder,'mos')
    #outcs=arcpy.Describe(path[1]).spatialReference
    arcpy.MosaicToNewRaster_management(input_rasters=path,output_location=output_path,
                                       raster_dataset_name_with_extension='mid_10m_dem.tif',
                                       coordinate_system_for_the_raster=outcs,
                                       pixel_type='32_BIT_FLOAT',cellsize='9.25925929999997E-05',
                                       number_of_bands='1', mosaic_method="LAST",
                                       mosaic_colormap_mode="FIRST"
                                       )

if __name__ == '__main__' :
    main()              
    