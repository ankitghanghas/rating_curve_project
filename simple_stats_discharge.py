# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 17:59:50 2020

@author: aghangha
"""

from climata.usgs import DailyValueIO
import pandas as pd
import numpy as np


nyears=50
ndays=365*nyears


#df=pd.read_csv('D:/ratingcurve/states/merge/nrmse_param_table.csv', converters={1:str,2:str} )
station_list = df['Station_ID'].values
i=0
two_year_flow=pd.DataFrame(0,index=range(0,len(station_list)), columns=['Station_ID','2yrFlow','gage_ten_perc'])


def Calc7Q(Qvalues):
    """This function computes the seven day low flow of an array of 
       values, typically a 1 year time series of streamflow, after 
       filtering out the NoData values. The index is calculated by 
       computing a 7-day moving average for the annual dataset, and 
       picking the lowest average flow in any 7-day period during
       that year.  The routine returns the 7Q (7-day low flow) value
       for the given data array."""
    Qvalues=Qvalues.dropna()
    val7Q=(Qvalues.rolling(7).mean()).min() 
    return ( val7Q )

def GetAnnualStatistics(DataDF):    
    col_name = ['PeakFlow','MeanFlow','MedianFlow','7Q']
    DataDF = DataDF.set_index('Date')
    data_1=DataDF.resample('AS-OCT').mean()
    WYDataDF = pd.DataFrame(0, index = data_1.index, columns=col_name)
    df=DataDF.resample('AS-OCT')
    WYDataDF['MeanFlow'] = df['Discharge'].mean()
    WYDataDF['PeakFlow'] = df['Discharge'].max()
    WYDataDF['MedianFlow'] = df['Discharge'].median()
    WYDataDF['7Q'] = df.apply({'Discharge': lambda x: Calc7Q(x)})

    return ( WYDataDF )

def percent_exced_flow(DataDF, percent):
    flow=DataDF['Discharge']
    flow=flow.sort_values(ascending=False)
    flow=flow.reset_index(drop=True)
    rank=np.arange(1,len(flow)+1)

    P=100*(rank/(len(rank)+1))
    id1=np.argmax(P>percent)# index of first value greater than percent
    if id1>0:
        val = (flow[id1]-flow[id1-1])*(percent-P[id1-1])/(P[id1]-P[id1-1]) + flow[id1-1]
    else:
        val = 0
    return val

for i in range(len(station_list)):
    station_id=station_list[i]
    
    param_id = "00060"
    percent_excedance=90
    
    datelist=pd.date_range(end=pd.datetime.today(), periods=ndays).tolist()
    
    data=DailyValueIO(start_date=datelist[0], end_date=datelist[-1], station =station_id, parameter=param_id)
    for series in data:
        discharge = [r[1] for r in series.data]
        dates = [r[0] for r in series.data]
    
    data_pair=pd.DataFrame(({"Date": dates, 'Discharge':discharge}))
    if len(data_pair)>0:        
        ten_per_flow= percent_exced_flow(data_pair,percent_excedance)
        WYdataDf = GetAnnualStatistics(data_pair)
        WYdataDf_peak=WYdataDf.copy()
        WYdataDf_peak=WYdataDf_peak['PeakFlow']
        WYdataDf_peak=WYdataDf_peak.rename('Discharge')
        WYdataDf_peak=WYdataDf_peak.reset_index()
        two_yr_flow_val=percent_exced_flow(WYdataDf_peak, 50)
        
        two_year_flow.iloc[i,0] = station_id
        two_year_flow.iloc[i,1] = two_yr_flow_val
        two_year_flow.iloc[i,2] = ten_per_flow
    else :
        pass

two_year_flow.to_csv('D:/ratingcurve/states/merge/Large_rivers_2_yr_flow.csv', index=False)
