"""
Created on Sat Feb 22 15:03:20 2020
Name Get SRC
Purpose: The R tool NHDPlus tool did not provide with comid for several points, so this code reads manually associated
list of GAGEid and comid and the gets the SRC for these GAGES and stores it in output folder
Author: Ankit Ghanghas
"""

import os
import pandas as pd


pd.options.mode.chained_assignment=None
work_folder = r"C:\Users\aghangha\Documents\ratingcurve"
HUC_id_list = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18"]
def main():
    SRC_folder = "C:\\Users\\aghangha\\Documents\\ratingcurve\\SyntheticRC\\" #specifies the SyntheticRC downloaded all locaiton
    huc6_src_list=os.listdir(SRC_folder) # creates a list of all the files in the folder
    huc02id=[]
    for file in huc6_src_list: # creates a associating each file path in SRC_folder to its corresponding HUC02 id
        huc02id.append(file[20:22])
    huc6_src_list=pd.DataFrame(data={'path':huc6_src_list, 'huc02':huc02id})
    
    comid_df=pd.read_excel(r"C:\Users\aghangha\Documents\ratingcurve\no_comid_stations.xlsx", converters={1:str,2:str, 3:str})
    comid_df=comid_df.dropna()
    for huc_id in HUC_id_list :
        allsrc_path=huc6_src_list.loc[huc6_src_list["huc02"]==huc_id,["path"]]
        df=pd.DataFrame()
        for i in range(len(allsrc_path)): #this for loop creates a df having the src of all the in HUC02 unit (combines the table of all huc06 units)
            src_path=SRC_folder + allsrc_path.iloc[i]
            df_sub=pd.read_csv(src_path.path)
            df=df.append(df_sub, ignore_index=True)
        station_list=comid_df.loc[comid_df["HUC02Id"]==huc_id,["Station_id","Comid"]]
        comid=station_list.Comid
        comid_list=comid.values
        try:
            for cur_comid in comid_list:
#                SRC_raw = df.loc[df['CatchId']==int(cur_comid),["Stage","Discharge (m3s-1)"]]
                SRC_raw = df.loc[df['CatchId']==int(cur_comid),["Stage","Discharge (m3s-1)", "WetArea (m2)"]]
                SRC_raw = SRC_raw.reset_index(drop=True)
                SRC_raw["COMID"]=cur_comid
                SRC_raw.to_csv(work_folder+"//Outputs//" + str(cur_comid) + ".csv")
        except OSError :
            print(huc_id)
            next

if __name__ == '__main__' :
    main()              