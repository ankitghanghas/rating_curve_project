#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 13:20:55 2019
Name: Near Tool on us gauges and NHDPlusHR dataset
#Purpose: Ues near tool in arcgis and gives a table with two nearest feature id's and gives a table(confuse) with points which
#         have (nearest feature distance - second nearest feature distance <10m). 
#       
#Author: Ankit Ghanghas
#--------------------------------------------------------------------------------------------------------------
"""


import arcpy
import os
import pandas
import numpy

arcpy.env.workspace = r"D:\Masters\RC\data\shapefiles"
arcpy.env.overwriteOutput = True

pandas.options.mode.chained_assignment = None

def main():
    NHD_shp_path = r"D:\Masters\RC\data\streams.gdb"
    gage_path = r"D:\Masters\RC\data\clip_gages.gdb"
    abcdf=pandas.read_excel('D:\Masters\RC\Data\huc4id.xlsx', converters={0:str})
    for i in range(len(abcdf)):
        huc_id=abcdf['HUC4ID'][i]
        #get relevant files
        NHD_shp_file = os.path.join(NHD_shp_path, r"stream" + huc_id)
        Gauge_shp_file = os.path.join(gage_path, r"gage" + huc_id)
        #Run near analysis to get near FID and location
    
        arcpy.GenerateNearTable_analysis(in_features=Gauge_shp_file, near_features=NHD_shp_file, out_table = "near_gaugefeature", search_radius="", location="LOCATION", angle="NO_ANGLE", method="PLANAR", closest = 'ALL', closest_count = 2)
        
        arcpy.AddField_management("near_gaugefeature","NHDPLUSID","DOUBLE")
        
        #get NHDPlusID for each extracted near object and corresponding totaldrainage area from the VAA Table
        arcpy.AddJoin_management("near_gaugefeature","Near_FID",NHD_shp_file,"OBJECTID")
        arcpy.CalculateField_management("near_gaugefeature","near_gaugefeature:NHDPLUSID",expression="!stream"+huc_id+".NHDPlusID!", expression_type="Python3", code_block="")
        arcpy.RemoveJoin_management("near_gaugefeature")
        flowVAA=os.path.join(r"D:\Masters\RC\data\nhd", huc_id[0:2],r"NHDPLUS_H_" + huc_id + r"_HU4_GDB.gdb\NHDPlusFlowlineVAA")
        arcpy.AddField_management("near_gaugefeature","TOTDASQKM","DOUBLE")
        arcpy.AddJoin_management("near_gaugefeature","NHDPLUSID",flowVAA,"NHDPlusID")
        arcpy.CalculateField_management("near_gaugefeature","near_gaugefeature:TOTDASQKM",expression="!NHDPlusFlowlineVAA.TotDASqKm!", expression_type="Python3", code_block="")
        arcpy.RemoveJoin_management("near_gaugefeature")
        #saves the near 
        arcpy.TableToTable_conversion("near_gaugefeature",'D:\\Masters\\RC\\data\\near_tables',"Near" + huc_id +".csv")
        #compare the nearest and chose the one with more drainage area. more confusing if i look at the one with closest area to the one mentioned in gage data.
        df=pandas.read_csv(os.path.join(r"D:\Masters\RC\data\near_tables","Near"+huc_id+".csv"), converters={10: str});
        nearest=df[df.NEAR_RANK==1];
        s_nearest=df[df.NEAR_RANK==2];
        nearest.set_index('IN_FID', inplace = True);
        s_nearest.set_index('IN_FID', inplace = True);
        confuse=nearest[s_nearest.NEAR_DIST-nearest.NEAR_DIST<10];
        bol=s_nearest.TOTDASQKM[confuse.index]>nearest.TOTDASQKM[confuse.index];
        idx=bol[bol].index;
        if len(idx)>0: 
            new_val=s_nearest.loc[idx,"NHDPLUSID"];
            nearest.loc[idx,"NHDPLUSID"]=new_val
            new_val=s_nearest.loc[idx,"TOTDASQKM"];
            nearest.loc[idx,"TOTDASQKM"]=new_val
        
        #Extract the gauge shape file feature to a pandas dataframe and then join this dataframe based ton FID with the nearest dataframe containing NHDPlusID and then export it as a csv
        tda=arcpy.da.FeatureClassToNumPyArray(Gauge_shp_file,('OBJECTID','STAID', 'STANAME','HUC02','LAT_GAGE','LNG_GAGE','STATE','DRAIN_SQKM'));
        gaugedf=pandas.DataFrame({'STAID':tda['STAID'],'DRAIN_SQKM':tda['DRAIN_SQKM'],'STANAME':tda['STANAME'],'HUC02':tda['HUC02'],'LAT_GAGE':tda['LAT_GAGE'],'LNG_GAGE':tda['LNG_GAGE'],'STATE':tda['STATE']},index=tda['OBJECTID']);
        gaugedf=gaugedf.merge(nearest[{"NHDPLUSID","TOTDASQKM"}], left_index=True,right_index=True, how ='inner')
        gaugedf.to_csv(os.path.join(r"D:\Masters\RC\data\tables\near","XYGauge_location"+huc_id+".csv"), index = None, header = True)
        mxd=arcpy.mp.ArcGISProject("CURRENT")
        for mapp in mxd.listMaps():
            for lyr in mapp.listLayers():
                mapp.removeLayer(lyr)
if __name__ == '__main__':
    main()
