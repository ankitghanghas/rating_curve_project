# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:28:36 2020



#incomplete and buggy not working.

# cant add addjoin management something wrong with naming. I am not using proper naming convention.

@author: aghangha
"""


import arcpy
import os
import pandas
import numpy

arcpy.env.workspace = r"C:\Users\aghangha\Documents\ratingcurve\indiana"
arcpy.env.overwriteOutput = True
pandas.options.mode.chained_assignment = None


Gauge_shp_file=r"C:\Users\aghangha\Documents\ratingcurve\indiana\gage_points\huc_051200.shp"
NHD_shp_file=r"C:\Users\aghangha\Documents\ratingcurve\indiana\nhd_clip\flowlines.shp"

arcpy.GenerateNearTable_analysis(in_features=Gauge_shp_file, near_features=NHD_shp_file, out_table = "near_gaugefeature", search_radius="", location="LOCATION", angle="NO_ANGLE", method="PLANAR", closest = 'ALL', closest_count = 1)
arcpy.AddField_management("near_gaugefeature","STAID","TEXT")
arcpy.AddJoin_management("near_gaugefeature","IN_FID",Gauge_shp_file,"FID")
#arcpy.CalculateField_management("near_gaugefeature","near_gaugefeature.STAID",expression="!huc_051200.COMID!", expression_type="Python3", code_block="")
#arcpy.RemoveJoin_management("near_gaugefeature")
arcpy.TableToTable_conversion("near_gaugefeature",'C:\\Users\\aghangha\\Documents\\ratingcurve\\indiana\\near_table',"Near.csv")

