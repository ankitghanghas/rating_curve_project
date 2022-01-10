# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 23:33:32 2020

@author: Ankit Ghanghas
"""
import pandas
import os

huc_id="0101"

df=pandas.read_csv(os.path.join(r"D:\Masters\RC\data\near_tables","Near"+huc_id+".csv"), converters={10: str});
nearest=df[df.NEAR_RANK==1];
s_nearest=df[df.NEAR_RANK==2];
nearest.set_index('IN_FID', inplace = True);
s_nearest.set_index('IN_FID', inplace = True);

confuse=nearest[s_nearest.NEAR_DIST-nearest.NEAR_DIST<10];
bol=s_nearest.TOTDASQKM[confuse.index]>nearest.TOTDASQKM[confuse.index];
idx=bol[bol].index;
if len(idx)>0: 
    new_val=s_nearest.loc[idx,"NHDPLUSID"];
    nearest.loc[idx,"NHDPLUSID"]=new_val
    new_val=s_nearest.loc[idx,"TOTDASQKM"];
    nearest.loc[idx,"TOTDASQKM"]=new_val