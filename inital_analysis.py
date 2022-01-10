# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 10:18:56 2020
Initial Analysis
@author: Ankit Ghanghas

This code takes the comparison curves(having both SRC and gage depth) as input and return a file containing station id and corresponding rmse error in depth
for two case: the unchanged one and one with area correction. 
Here by area correction- i mean that i changed depth of each value of src depth by a constant value such that i get minimum rmse for that particular pair.

The thrid column in the output file "depth correciton" stores the value of this constant change for each station.

"gage_with_one_reading.txt" file store the USGS gage id for stations which have only one value in their USGS rating curve so cannot be used for comparison.

"""
import pandas as pd
# import matplotlib.pyplot as plt
import os
import numpy as np

comp_curve_path="C:/Users/aghangha/Documents/ratingcurve/comparison_curves/"
station_list=os.listdir(comp_curve_path)
df=pd.DataFrame(columns = ["Station_id","RMSE_unchanged", "RMSE_corrected_area", "depth_correction"])
one_reading=[]
for station in station_list:
    curves=pd.read_csv(comp_curve_path + station)
    curves=curves.dropna()
    curves=curves.reset_index(drop=True)
    syn=curves.Depth_ft
    discharge=curves.Discharge_cfs
    gage=curves.Gauge_Depth
    Station_id=station[:-4]
    
    if len(curves.Gauge_Depth)>1:
        error_unchanged=np.sqrt(((gage-syn)**2).mean())
        area_correction=sum(gage -syn)/len(gage)
        corrected_syn=syn + area_correction
        if corrected_syn[0]<0:  # allow only positive depth values even in corrected synthetic curve
            offset=corrected_syn[0]
            corrected_syn=corrected_syn + offset
            area_correction= area_correction + offset
        corrected_error=np.sqrt(((gage-corrected_syn)**2).mean())
       
        df=df.append([{'Station_id': Station_id,'RMSE_unchanged': error_unchanged, 'RMSE_corrected_area': corrected_error, "depth_correction" : area_correction}], ignore_index= True)
        
        # plt.figure()
        # plt.plot(discharge, syn, 'r-',discharge, gage, 'k-', discharge, corrected_syn, 'b--')
        # plt.legend (("HAND", "USGS", "Corrected Hand"), loc="lower right")
        
        # plt.show()
    else:
        one_reading.append(Station_id)

file=open('gage_with_one_reading.txt','w')
file.write('Station_id')
file.write('\n')
file.writelines("%s\n" % station for station in one_reading)
file.close()

df.to_csv("C:/Users/aghangha/Documents/ratingcurve/intial_rmse_analysis.csv")