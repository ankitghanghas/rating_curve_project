# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:50:51 2020

@author: Ankit Ghanghas
project nhdplus medium resolution stream to nad albers 1983

"""
import arcpy
import os
import pandas as pd
arcpy.env.workspace = r"D:\Masters\RC\Data\shapefiles"
arcpy.env.overwriteOutput = True
outnhd=r"D:\Masters\RC\Data\nhdprj\projectednhd.gdb"
outgage=r"D:\Masters\RC\data\clip_gages.gdb"
gages=r"D:\Masters\RC\Data\gages_shapefile\gagesII_9322_sept30_2011.shp"
outcs=arcpy.Describe(gages).spatialReference

df=pd.read_excel('D:\Masters\RC\Data\huc4id.xlsx', converters={0:str})

for i in range(len(df)):
    huc_id=df['HUC4ID'][i]
    WBDHU8=os.path.join(r"D:\Masters\RC\data\nhd",huc_id[0:2], r"NHDPLUS_H_" + huc_id + r"_HU4_GDB.gdb\WBD\WBDHU8")
    flowline=os.path.join(r"D:\Masters\RC\data\nhd",huc_id[0:2], r"NHDPLUS_H_" + huc_id + r"_HU4_GDB.gdb\Hydrography\NHDFlowline")
    dsc=arcpy.Describe(WBDHU8)
    if dsc.spatialReference.Name == "Unknown":
        print('skipped this fc due to undefined coordinate system:' + huc_id)
    else :
        outwbd=os.path.join(outnhd,r"WBDHU8")
        arcpy.Project_management(WBDHU8, outwbd, outcs)
        outflow=os.path.join(outnhd,r"flowline")
        
    arcpy.MakeFeatureLayer_management(flowline,"Flow_lyr")
    arcpy.SelectLayerByAttribute_management("Flow_lyr", 'NEW_SELECTION','"FType" = 334 Or "FType" = 336 Or "FType" = 460 Or "FType" = 558')
    arcpy.CopyFeatures_management("Flow_lyr",outflow)
    outstream=os.path.join(r"D:\Masters\RC\data\streams.gdb",r"stream"+huc_id)
    arcpy.Project_management(outflow,outstream,outcs)    
    xy_tolerance=""
    clip_gage=os.path.join(outgage,r"gage"+huc_id)
    arcpy.Clip_analysis(gages,outwbd,clip_gage,xy_tolerance)
    mxd=arcpy.mp.ArcGISProject("CURRENT")
    for mapp in mxd.listMaps():
        for lyr in mapp.listLayers():
            mapp.removeLayer(lyr)
