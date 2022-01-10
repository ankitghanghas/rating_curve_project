# -*- coding: utf-8 -*-
"""
Created on Mon May 24 13:32:54 2021

@author: aghangha
"""

import pandas as pd
import os
import numpy as np
import arcpy
import matplotlib.pyplot as plt

comp_curve_path="C:/Users/aghangha/Documents/ratingcurve/comparison_curves/"
#station_list=os.listdir(comp_curve_path)
df=pd.DataFrame(columns = ["Station_id","Rec_RMSE", "Rec_Discharge_correction", "Rec_Depth_correction", "Tri_RMSE", "Tri_Discharge_correction", "Tri_Depth_correction" ])
gage_datum = pd.read_csv('C:/Users/aghangha/Documents/ratingcurve/datum_usgs.csv',converters={1:str})
gage_datum=gage_datum.set_index('site_no')
#src_datum = pd.read_csv('C:/Users/aghangha/Documents/ratingcurve/indiana/near_table/dem_values.csv',converters={3:str})
src_datum = pd.read_csv('D:/ratingcurve/states/nmw/dem_values.csv',converters={3:str})
src_datum = src_datum.set_index('Station_ID')
# comid_df=pd.read_csv(r"C:\Users\aghangha\Documents\ratingcurve\gage_comid_nhdtools.csv", converters={1:str,5:str})
# comid_df_1=pd.read_excel(r"C:\Users\aghangha\Documents\ratingcurve\no_comid_stations.xlsx", converters={1:str,2:str, 3:str})
# comid_df_1=comid_df_1.dropna()
# USGSID=pd.concat([comid_df_1['Station_id'],comid_df[('USGSID')]], ignore_index=True)
# COMID=pd.concat([comid_df_1['Comid'],comid_df['COMID']],ignore_index=True)
usgs_rc_path=r"C:\Users\aghangha\Documents\ratingcurve\observed_rating_jan20"

src_path="C:/Users/aghangha/Documents/ratingcurve/Outputs/"

station_list=src_datum['RASTERVALU'].index.values + '.csv' # change this appropirately I did this here to extract just indiana stations list.
df=pd.DataFrame(columns = ["Station_id", "Discharge_correction(cfs)", "Area_correction(m2)", "RMSE_sur_el", "NRMSE", 'COMID','StreamOrder','Length_km_nhd','Slope_nhd','Bias','NBias'])
one_reading=[]
large_var=[]
no_SRC=[]

# extracting flowline characteristics from NHD dataset.
flowline=r"D:\ratingcurve\states\nmw\shapefile\flowlines_nad83.shp"
comid_values, length_km, stream_ord,slope, =[str(int(row[0])) for row in arcpy.da.SearchCursor(flowline,['COMID'])],[row[0] for row in arcpy.da.SearchCursor(flowline,['LENGTHKM'])],[row[0] for row in arcpy.da.SearchCursor(flowline,['StreamOrde'])],[row[0] for row in arcpy.da.SearchCursor(flowline,['SLOPE'])]
df_stream=pd.DataFrame()
df_stream['COMID']=comid_values; df_stream['LENGTH']=length_km;df_stream['StreamOrder']=stream_ord; df_stream['Slope']=slope

# gages=r"C:\Users\aghangha\Documents\ratingcurve\gages_shapefile\gagesII_9322_sept30_2011.shp"
# STAID,Drain_sq_km=[str(int(row[0])) for row in arcpy.da.SearchCursor(gages,['STAID'])],[row[0] for row in arcpy.da.SearchCursor(gages,['DRAIN_SQKM'])]
# df_gage=pd.DataFrame()
# df_gage['STAID']=STAID;df_gage['Area']=Drain_sq_km


# this dataset is one which i created by first selecting ngvd29 files then creating a separate shapefile out of it
# i made 3d points of this. defined the projection as  ngvd29 and then projected all to navd88. This is done to keep consistency with DEM which are in NAVD 88
# here i am importing this shapefile in and then converting the z values (given in meters) to ft.
gage_29=r"C:\Users\aghangha\Documents\ratingcurve\gages_shapefile\ngvd29\usgs_final.shp"
STAID,z=[str(row[0]) for row in arcpy.da.SearchCursor(gage_29,['STAID'])],[row[0] for row in arcpy.da.SearchCursor(gage_29,['z'])]
df_gage=pd.DataFrame();
df_gage['STAID']=STAID;df_gage['Datum']=z
df_gage=df_gage.set_index('STAID')
df_gage=df_gage*3.28084 # converting everything into feet.

swot_acp=pd.read_csv(r'E:\ratingcurve\bathymety_select\swot\SWOT_ADCP_Dataset.txt', sep="\t", converters = {1:str,2:str,3:str})

# changing all the ngvd values to the projected navd 88 values
gage_datum['altitude'][df_gage.index]=df_gage.Datum

for station in station_list:
    Station_id=station[:-4]
    try:        
        curves=pd.read_csv(comp_curve_path + station)
        if curves.empty:
            pass
        else: 
            curves=curves.dropna()
            curves=curves.reset_index(drop=True)
            syn=curves.Depth_ft
            discharge=curves.Discharge_cfs
            gage=curves.Gauge_Depth
            
            if len(curves.Gauge_Depth)>1:
                gage_height = gage_datum['altitude'][Station_id]
                dem_elev = src_datum['RASTERVALU'][Station_id]*3.28084
                syn1=syn+dem_elev
                gage1=gage+gage_height
                if gage1[gage1.gt(syn1[0])].empty:
                    print('Very Large Variation in GAGE and SRC Datum:' + Station_id)
                    large_var.append(Station_id)
                else: 
                    idx=gage1[gage1.gt(syn1[0])].idxmin()
                    if idx==0:
                        q_cor=0.0
                    else :
                        q_cor=((discharge[idx]-discharge[idx-1])/(gage1[idx]-gage1[idx-1]))*(syn1[0]-gage1[idx-1])+discharge[idx-1]
                    #area_file=COMID[USGSID[USGSID==Station_id].index].values[0]+'.csv'
                    area_file=str(src_datum.COMID[Station_id]) +'.csv'
                    area_df=pd.read_csv(src_path+area_file)
                    dis_src=area_df['Discharge (m3s-1)']
                    area_src = area_df['WetArea (m2)']
                    area_src=area_src**(5/3)
                    idx1=dis_src[dis_src.gt(q_cor/35.3147)].idxmin()
                    if idx1<=1:
                        area_cor =0.0
                    else:
                        area_cor=(((area_src[idx1]-area_src[idx1-1])/(dis_src[idx1]-dis_src[idx1-1]))*(q_cor/35.3147-dis_src[idx1-1])+area_src[idx1-1])**(3/5)
                        area_cor=round(area_cor,4)
                    
                        q_cor=round(q_cor,4)
                    # this gives me area for correction. Next I will add this area to area corresponding to syn stage and recalculate discharge for syn from Q relation with a^5/3  linear interpolation
                    dis1=[]
                    
                    filtered_area_df=area_df[(round(area_df.Stage*3.28084,3)>=syn.min()) &  (round(area_df.Stage*3.28084,3) <=syn.max())]
#                    new_area=filtered_area_df['WetArea (m2)']+ area_cor
#                    height=swot_acp[swot_acp.site_no==Station_id]['stage_va']+gage_height
#                    
#                    plt.figure()
#                    plt.scatter(swot_acp[swot_acp.site_no==Station_id]['xsec_area_va'], height)
#                    plt.scatter(new_area,syn1)
#                    plt.scatter(filtered_area_df['WetArea (m2)'],syn1)
#                    
                    new_area=filtered_area_df['WetArea (m2)']+ area_cor
                    height=swot_acp[swot_acp.site_no==Station_id]['stage_va']-swot_acp[swot_acp.site_no==Station_id]['stage_va'].min()
                    
#                    plt.figure()
#                    plt.scatter(swot_acp[swot_acp.site_no==Station_id]['xsec_area_va'], height)
#                    plt.scatter(new_area,(syn1-syn1.min()))
#                    plt.scatter(filtered_area_df['WetArea (m2)'],(syn1-syn1.min()))
                    
                    swot_pair=pd.DataFrame({'area_swot':swot_acp[swot_acp.site_no==Station_id]['xsec_area_va'],'height_swot':height})
                    swot_pair=swot_pair.sort_values(by=['height_swot'])
                    swot_pair.reset_index(drop=True, inplace=True)
                    src_pair=pd.DataFrame({'area_corrected':new_area,'src_area':filtered_area_df['WetArea (m2)']})
                    print(swot_pair)
                    print(src_pair)
                    print((syn1-syn1.min()))
                    
                    
                    
                    
            else:
                one_reading.append(Station_id)  
    except OSError:
        no_SRC.append(Station_id)
        print("Station_curve_not_found:" + Station_id)                        
