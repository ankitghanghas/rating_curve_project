# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:58:31 2021

@author: aghangha
"""

#####make figures,

#### median and mean huc 2 unit rmse nrmse and all
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


df=pd.read_csv('D:/ratingcurve/states/merge/nrmse_param_table_2yrflow.csv', converters={1:str,2:str} )
df['rmse_sign']=df['RMSE_sur_e']*df['nbias_sign']

gage_info=pd.read_csv('C:/Users/aghangha/Documents/ratingcurve/gage_info_table.csv', converters={1:str})
del gage_info['OID_']   ####    drop OID as its not useful here.
gage_info.loc[((gage_info.HUC02=='10U') | (gage_info.HUC02=='10L')),'HUC02']='10'

gage_info=gage_info.set_index(['STAID']) 
df=df.set_index(['Station_ID'])
merge=pd.merge(df,gage_info, how='inner',left_index=True, right_index=True)

huc_id = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18']


mean_nrmse =[]
median_nrmse = []
median_rmse = []
mean_rmse =[]
median_rmse_sign =[]
mean_rmse_sign = []
median_nrmse_sign = []
RMSE_25=[]
NRMSE_25=[]
RMSE_75=[]
NRMSE_75=[]

plt.figure()

for i, huc in enumerate (huc_id):
    mean_nrmse.append(merge[merge.HUC02==huc]['NRMSE'].mean())
    median_nrmse.append(merge[merge.HUC02==huc]['NRMSE'].median())
    
    median_rmse.append(merge[merge.HUC02==huc]['RMSE_sur_e'].median())
    mean_rmse.append(merge[merge.HUC02==huc]['RMSE_sur_e'].mean())
    
    median_rmse_sign.append(merge[merge.HUC02==huc]['rmse_sign'].median())
    mean_rmse_sign.append(merge[merge.HUC02==huc]['rmse_sign'].mean())
    
    median_nrmse_sign.append(merge[merge.HUC02==huc]['nrmse_sign'].median())
    
    NRMSE_25.append(merge[merge.HUC02==huc]['nrmse_sign'].quantile(0.25))
    NRMSE_75.append(merge[merge.HUC02==huc]['nrmse_sign'].quantile(0.75))
    RMSE_25.append(merge[merge.HUC02==huc]['rmse_sign'].quantile(0.25))
    RMSE_75.append(merge[merge.HUC02==huc]['rmse_sign'].quantile(0.75))

    
    plt.boxplot(merge[merge.HUC02==huc]['rmse_sign']*0.3048, positions =[i+1])
    
    
huc_metrics=pd.DataFrame({'HUC_ID': huc_id,'NRMSE_mean':mean_nrmse,'NRMSE_median':median_nrmse,'Median_nrmse_sign': median_nrmse_sign,'mean_rmse':mean_rmse,'median_rmse':median_rmse,'median_rmse_sign':median_rmse_sign,'mean_rmse_sign':mean_rmse_sign,'NRMSE_25':NRMSE_25,'NRMSE_75':NRMSE_75,'RMSE_25':RMSE_25,'RMSE_75':RMSE_75})