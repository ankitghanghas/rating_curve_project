"""
Created on Mon Feb 17 2020
Name Get SRC
Purpose: from the split(as per huc6 units) gage station data extract the SRC using COMID
Author: Ankit Ghanghas
"""

import arcpy
import os
import pandas as pd

arcpy.env.workspace = r"C:\Users\aghangha\Documents\ratingcurve\hu6gage"
arcpy.env.overwriteOutput = True
pd.options.mode.chained_assignment=None
work_folder = r"C:\Users\aghangha\Documents\ratingcurve"
def main():
    gage_path= r"C:\Users\aghangha\Documents\ratingcurve\hu6gage"
    comid_df=pd.read_csv(r"C:\Users\aghangha\Documents\ratingcurve\gage_comid_nhdtools.csv", converters={1:str,5:str})
    comid_df.index=comid_df.USGSID
    shp_name_list = arcpy.ListFeatureClasses()
    for shp_name in shp_name_list:
           huc6_id= shp_name[0:6]
           gage = os.path.join(gage_path, shp_name)
           arr = arcpy.da.TableToNumPyArray(gage, ('STAID'))
           df=pd.DataFrame(arr)
           comid=comid_df.COMID[df.STAID].dropna()
           try:
               SRC_path = "C:\\Users\\aghangha\\Documents\\ratingcurve\\SyntheticRC\\"
               SRC_all =pd.read_csv(SRC_path + "hydroprop-fulltable-"+huc6_id+".csv")
               comid_list=comid.values
               for cur_comid in comid_list:
                   SRC_raw = SRC_all.loc[SRC_all['CatchId']==int(cur_comid),["Stage","Discharge (m3s-1)", "WetArea (m2)"]]
                   SRC_raw = SRC_raw.reset_index(drop=True)
                   SRC_raw["COMID"]=cur_comid
                   SRC_raw.to_csv(work_folder+"//Outputs//" + str(cur_comid) + ".csv")
           except OSError:
              print(shp_name)
              next
if __name__ == '__main__' :
    main()              