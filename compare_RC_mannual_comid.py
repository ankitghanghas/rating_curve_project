# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 18:16:30 2020

Modified Ankit Ghanghas
Original Sayan Dey
"""
import os
import pandas as pd
import numpy as np
import time
start_time =time.time()

def main():
    usgs_rc_path=r"C:\Users\aghangha\Documents\ratingcurve\observed_rating_jan20"
    Synthetic_path=r"C:\Users\aghangha\Documents\ratingcurve\Outputs"
    comid_df=pd.read_excel(r"C:\Users\aghangha\Documents\ratingcurve\no_comid_stations.xlsx", converters={1:str,2:str, 3:str})
    comid_df=comid_df.rename(columns={"Station_id": "USGSID","Comid": "COMID"})
    comid_df=comid_df.dropna()
    comid_df.index=comid_df.USGSID
    station_list=os.listdir(usgs_rc_path)
    no_comid=[]
    for file in station_list:
        path=os.path.join(usgs_rc_path,file)
        usgs_rc=pd.read_csv(path)
        Gauge_rc=pd.DataFrame()
        Gauge_rc['Depth']=usgs_rc["INDEP"]
        Gauge_rc['Flow']=usgs_rc["DEP"]
        Station_id=(file[0:-4])
        try:
            NHD_comID=comid_df.COMID[Station_id]
            SRC_cur = pd.DataFrame()
            SRC_cur = GetSRC_cfs(NHD_comID, Synthetic_path, Station_id)
            if SRC_cur.empty:
                continue
            else:
                SRC_cur_filt = GetSRCinUSGSRange(Gauge_rc,SRC_cur)
                RC_comp_tmp = InterpolateRC(Gauge_rc,SRC_cur_filt)
                RC_comp_tmp["GageID"]= Station_id
                RC_comp_tmp.to_csv("C:\\Users\\aghangha\\Documents\\ratingcurve\\comparison_curves\\" + file)
        except :
            no_comid.append(Station_id)
            next
    no_comid_df=pd.DataFrame(data={"Station_id": no_comid})
    no_comid_df.to_csv("C:\\Users\\aghangha\\Documents\\ratingcurve\\" + "no_comid_stations.csv")
    print("--- %s seconds ---" % (time.time() - start_time))

def GetSRC_cfs(NHD_comID, Synthetic_path, Station_id):
    file_path=os.path.join(Synthetic_path,NHD_comID + ".csv")
    try:
        raw=pd.read_csv(file_path, converters={3:str})
        if raw.empty:
            pass
        else:
            Synth_rc=pd.DataFrame()
            Synth_rc["Depth_ft"] = raw["Stage"]*3.28084
            Synth_rc["Depth_ft"] = Synth_rc.Depth_ft.round(2)
            Synth_rc["Discharge_cfs"] = raw["Discharge (m3s-1)"]*35.3147
            Synth_rc["Discharge_cfs"] = Synth_rc.Discharge_cfs.round(2)
        
    except OSError:
        print("SRC_not_found:" + Station_id)
    return Synth_rc

def GetSRCinUSGSRange (USGS_RC, Synth_RC):
    min_val = USGS_RC.Flow.min()
    max_val = USGS_RC.Flow.max()
    filtered_rc = Synth_RC[ (Synth_RC["Discharge_cfs"] >= min_val) & (Synth_RC["Discharge_cfs"] <= max_val) & (Synth_RC["Discharge_cfs"] > 0) ]
    filtered_rc = filtered_rc.reset_index(drop=True)
    return filtered_rc

def InterpolateRC(RC1, RC2):
    #RC1 is USGS, RC2 is synthetic
    #This function caculated the depth as per RC1 for flows from RC2
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

if __name__ == '__main__':
    main()
