# -*- coding: utf-8 -*-
"""
Created on Tue August 15 15:37:27 2020

@author: aghangha
"""

import arcpy
import pandas as pd
import os
from statistics import mean

work_database=r'D:\ratingcurve\huc\huc6.gdb'
arcpy.env.workspace = work_database

featureClass= arcpy.ListFeatureClasses()
df = pd.DataFrame(columns = ['HUC_ID','mean_nrmse'])		
for fc in featureClass :
	feature=os.path.join(work_database,fc)
	nrmse=[str(row[0]) for row in arcpy.da.SearchCursor(feature,['NRMSE'])]
	nrmse_avg= mean(float(n) for n in nrmse) # i get nrmse as a list of strings so need to convert it to float before

	df=df.append([{'HUC_ID': fc[1:],'mean_nrmse': nrmse_avg}], ignore_index= True)
df.to_csv('D:/ratingcurve/huc/huc_6.csv', index=False)
