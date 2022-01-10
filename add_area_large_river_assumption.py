# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:13:36 2020

@author: aghangha
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.linear_model import LinearRegression


def main():
    comp_curve_path="C:/Users/aghangha/Documents/ratingcurve/comparison_curves/"
    station_list=os.listdir(comp_curve_path)
    df=pd.DataFrame(columns = ["Station_id","Rec_RMSE", "Rec_Discharge_correction", "Rec_Depth_correction", "Tri_RMSE", "Tri_Discharge_correction", "Tri_Depth_correction" ])
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
            
            #Rectangular Crossection and assuming large base width
            model1=LinearRegression()
            syn1=(syn**(5/3)).values.reshape(-1,1)
            model1.fit(syn1,discharge)
            y_pred=model1.predict((gage**(5/3)).values.reshape(-1,1))
            q_cor_rec=sum(discharge-y_pred)/len(discharge)
            h_base_rec=abs((q_cor_rec/model1.coef_))**(3/5)
            corrected_dis= discharge+q_cor_rec
            RC2=pd.DataFrame()
            RC2['Depth']=syn.loc[corrected_dis>0]
            RC2['Discharge_cfs']=corrected_dis.loc[corrected_dis>0]
            RC2=RC2.reset_index(drop=True)
            RC=InterpolateRC(station,RC2)
            corrected_error_rec=np.sqrt(((RC.Gauge_Depth-RC.Depth)**2).mean())
            
            # Assuming Triangular crossection.
            model2=LinearRegression()
            syn2=(syn**(8/3)).values.reshape(-1,1)
            model2.fit(syn2,discharge)
            y_pred2=model2.predict((gage**(8/3)).values.reshape(-1,1))
            q_cor_tri=sum(discharge-y_pred2)/len(discharge)
            h_base_tri=abs((q_cor_tri/model2.coef_))**(3/8)
            corrected_dis2= discharge+q_cor_tri
            RC4=pd.DataFrame()
            RC4['Depth']=syn.loc[corrected_dis2>0]
            RC4['Discharge_cfs']=corrected_dis2.loc[corrected_dis2>0]
            RC4=RC4.reset_index(drop=True)
            RC3=InterpolateRC(station,RC4)
            corrected_error_tri=np.sqrt(((RC3.Gauge_Depth-RC3.Depth)**2).mean())
            df=df.append([{'Station_id': Station_id,'Rec_RMSE': corrected_error_rec, 'Rec_Discharge_correction': q_cor_rec, "Rec_Depth_correction" : h_base_rec, 'Tri_RMSE': corrected_error_tri, 'Tri_Discharge_correction': q_cor_tri, "Tri_Depth_correction" : h_base_tri}], ignore_index= True)
        
        else:
            one_reading.append(Station_id)      
    df.to_csv("C:/Users/aghangha/Documents/ratingcurve/rec_river_assump_rmse_analysis.csv")

def InterpolateRC(file, RC2):
    #File is USGS Station ID, RC2 is synthetic
    #This function caculated the depth as per RC1 for flows from RC2
    usgs_rc_path=r"C:\Users\aghangha\Documents\ratingcurve\observed_rating_jan20"
    path=os.path.join(usgs_rc_path,file)
    usgs_rc=pd.read_csv(path)
    if usgs_rc.DEP[0] == 0:
        Offset_val = usgs_rc.INDEP[0]
    else:
        Offset_val = usgs_rc.SHIFT[0]
    RC1=pd.DataFrame()
    RC1['Depth']=usgs_rc["INDEP"] - Offset_val
    RC1['Flow']=usgs_rc["DEP"]
    RC2=GetSRCinUSGSRange(RC1,RC2)
    RC_comparison = RC2
    RC_comparison["Gauge_Depth"] = 0
    for flow in RC2.Discharge_cfs:
        row_index = RC1.loc[RC1["Flow"]>=flow,"Flow"].idxmin()
        Q2 = RC1.Flow[row_index]
        Q1 = RC1.Flow[row_index - 1]
        y2 = RC1.Depth[row_index]
        y1 = RC1.Depth[row_index-1]

        if Q1 == 0 or y1 == 0:
            #Do linear interpolation
            depth_interp = (y2-y1)/(Q2-Q1)*(flow-Q1) + y1
            #print flow
        else:
            b1 = (np.log10(y2) - np.log10(y1))/(np.log10(Q2) - np.log10(Q1))
            b0 = y2/(Q2**b1)
            depth_interp = b0 * (flow ** b1)

        RC_comparison.loc[RC_comparison["Discharge_cfs"]==flow,"Gauge_Depth"] = depth_interp

    return RC_comparison

def GetSRCinUSGSRange (USGS_RC, Synth_RC):
    min_val = USGS_RC.Flow.min()
    max_val = USGS_RC.Flow.max()
    filtered_rc = Synth_RC[ (Synth_RC["Discharge_cfs"] >= min_val) & (Synth_RC["Discharge_cfs"] <= max_val) & (Synth_RC["Discharge_cfs"] > 0) ]
    filtered_rc = filtered_rc.reset_index(drop=True)
    return filtered_rc

if __name__ == '__main__':
    main()
