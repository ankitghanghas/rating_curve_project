# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 13:26:37 2021

@author: aghangha
"""

df=pd.read_csv('D:/ratingcurve/states/merge/nrmse_param_table_2yrflow.csv', converters={1:str,2:str} )
cal=pd.read_csv('D:/ratingcurve/states/ca/bathymetric_area_ca.csv',converters={0:str,5:str})


for station in cal.Station_id:
    df.loc[df.Station_ID==station,'RMSE_sur_e']=cal[cal.Station_id==station].RMSE_sur_el.values
    df.loc[df.Station_ID==station,'Discharge_']=cal[cal.Station_id==station]['Discharge_correction(cfs)'].values
    df.loc[df.Station_ID==station,'Area_corre']=cal[cal.Station_id==station]['Area_correction(m2)'].values
    df.loc[df.Station_ID==station,'StreamOrde']=cal[cal.Station_id==station]['StreamOrder'].values
    df.loc[df.Station_ID==station,'Length_km_']=cal[cal.Station_id==station]['Length_km_nhd'].values
    
    
    