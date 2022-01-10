# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 18:14:34 2020

@author: aghangha
"""

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

lparams= ['Elevation','Stream Order','Lattitude','Longitude','Two Year Flow','Water Surface Slope','Percent Impervious']
list_param=['RASTERVALU','StreamOrde','LAT','LON','2yrFlow', 'Slope_nhd','per_imp']

df=pd.read_csv('D:/ratingcurve/states/merge/nrmse_param_table_2yrflow.csv', converters={1:str,2:str} )
df2=df[['nrmse_sign','RASTERVALU','StreamOrde','LAT','LON','2yrFlow', 'Slope_nhd','MEAN','per_imp']]
df3=df2.copy()

# To calculate kendall's tau to give correlation value.

tau=[]
p_val=[]

for i in range(df3.shape[1]-1):
   t,p=st.kendalltau(df3.iloc[:,0],df3.iloc[:,i+1])
   tau.append(t)
   p_val.append(p)
    
   
df_correlation=pd.DataFrame({'Parameter': lparams,'KendallTau':tau,'Pvalue':p_val})
df_correlation.to_csv('D:/ratingcurve/states/merge/kendall_correlation_firstseven_param.csv', index=False)

for i in range(len(list_param)): 
    plt.figure()
    plt.scatter(df3[list_param[i]],df3.nrmse_sign)
    plt.xlabel(lparams[i])
    plt.ylabel('NRMSE(with sign for bias)')
    plt.title(lparams[i])
    plt.savefig('D:/ratingcurve/states/merge/correlation_figs/'+lparams[i]+'.png')