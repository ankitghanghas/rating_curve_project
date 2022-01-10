# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:48:27 2020

@author: aghangha
"""

import pandas as pd
import os
import numpy as np
import arcpy


def main():
    comp_curve_path="C:/Users/aghangha/Documents/ratingcurve/comparison_curves/"
    #station_list=os.listdir(comp_curve_path)
    df=pd.DataFrame(columns = ["Station_id","Rec_RMSE", "Rec_Discharge_correction", "Rec_Depth_correction", "Tri_RMSE", "Tri_Discharge_correction", "Tri_Depth_correction" ])
    gage_datum = pd.read_csv('C:/Users/aghangha/Documents/ratingcurve/datum_usgs.csv',converters={1:str})
    gage_datum=gage_datum.set_index('site_no')
    #src_datum = pd.read_csv('C:/Users/aghangha/Documents/ratingcurve/indiana/near_table/dem_values.csv',converters={3:str})
    src_datum = pd.read_csv('D:/ratingcurve/states/wa/dem_values.csv',converters={3:str})
    src_datum = src_datum.set_index('Station_ID')
    # comid_df=pd.read_csv(r"C:\Users\aghangha\Documents\ratingcurve\gage_comid_nhdtools.csv", converters={1:str,5:str})
    # comid_df_1=pd.read_excel(r"C:\Users\aghangha\Documents\ratingcurve\no_comid_stations.xlsx", converters={1:str,2:str, 3:str})
    # comid_df_1=comid_df_1.dropna()
    # USGSID=pd.concat([comid_df_1['Station_id'],comid_df[('USGSID')]], ignore_index=True)
    # COMID=pd.concat([comid_df_1['Comid'],comid_df['COMID']],ignore_index=True)
    usgs_rc_path=r"C:\Users\aghangha\Documents\ratingcurve\observed_rating_jan20"

    src_path="C:/Users/aghangha/Documents/ratingcurve/Outputs/"
    
    station_list=src_datum['RASTERVALU'].index.values + '.csv' # change this appropirately I did this here to extract just indiana stations list.
    df=pd.DataFrame(columns = ["Station_id", "Discharge_correction(cfs)", "Area_correction(m2)", "RMSE_sur_el", "NRMSE", 'COMID','StreamOrder','Length_km_nhd','Slope_nhd','Bias','NBias'])
    one_reading=[]
    large_var=[]
    no_SRC=[]
    
    # extracting flowline characteristics from NHD dataset.
    flowline=r"D:\ratingcurve\states\wa\shapefile\flowlines_nad83.shp"
    comid_values, length_km, stream_ord,slope, =[str(int(row[0])) for row in arcpy.da.SearchCursor(flowline,['COMID'])],[row[0] for row in arcpy.da.SearchCursor(flowline,['LENGTHKM'])],[row[0] for row in arcpy.da.SearchCursor(flowline,['StreamOrde'])],[row[0] for row in arcpy.da.SearchCursor(flowline,['SLOPE'])]
    df_stream=pd.DataFrame()
    df_stream['COMID']=comid_values; df_stream['LENGTH']=length_km;df_stream['StreamOrder']=stream_ord; df_stream['Slope']=slope
    
    # gages=r"C:\Users\aghangha\Documents\ratingcurve\gages_shapefile\gagesII_9322_sept30_2011.shp"
    # STAID,Drain_sq_km=[str(int(row[0])) for row in arcpy.da.SearchCursor(gages,['STAID'])],[row[0] for row in arcpy.da.SearchCursor(gages,['DRAIN_SQKM'])]
    # df_gage=pd.DataFrame()
    # df_gage['STAID']=STAID;df_gage['Area']=Drain_sq_km
    
    
    # this dataset is one which i created by first selecting ngvd29 files then creating a separate shapefile out of it
    # i made 3d points of this. defined the projection as  ngvd29 and then projected all to navd88. This is done to keep consistency with DEM which are in NAVD 88
    # here i am importing this shapefile in and then converting the z values (given in meters) to ft.
    gage_29=r"C:\Users\aghangha\Documents\ratingcurve\gages_shapefile\ngvd29\usgs_final.shp"
    STAID,z=[str(row[0]) for row in arcpy.da.SearchCursor(gage_29,['STAID'])],[row[0] for row in arcpy.da.SearchCursor(gage_29,['z'])]
    df_gage=pd.DataFrame();
    df_gage['STAID']=STAID;df_gage['Datum']=z
    df_gage=df_gage.set_index('STAID')
    df_gage=df_gage*3.28084 # converting everything into feet.
    
    # changing all the ngvd values to the projected navd 88 values
    gage_datum['altitude'][df_gage.index]=df_gage.Datum

    for station in station_list:
        Station_id=station[:-4]
        try:        
            curves=pd.read_csv(comp_curve_path + station)
            if curves.empty:
                pass
            else: 
                curves=curves.dropna()
                curves=curves.reset_index(drop=True)
                syn=curves.Depth_ft
                discharge=curves.Discharge_cfs
                gage=curves.Gauge_Depth
                
                if len(curves.Gauge_Depth)>1:
                    gage_height = gage_datum['altitude'][Station_id]
                    dem_elev = src_datum['RASTERVALU'][Station_id]*3.28084
                    syn1=syn+dem_elev
                    gage1=gage+gage_height
                    if gage1[gage1.gt(syn1[0])].empty:
                        print('Very Large Variation in GAGE and SRC Datum:' + Station_id)
                        large_var.append(Station_id)
                    else: 
                        idx=gage1[gage1.gt(syn1[0])].idxmin()
                        if idx==0:
                            q_cor=0.0
                        else :
                            q_cor=((discharge[idx]-discharge[idx-1])/(gage1[idx]-gage1[idx-1]))*(syn1[0]-gage1[idx-1])+discharge[idx-1]
                        #area_file=COMID[USGSID[USGSID==Station_id].index].values[0]+'.csv'
                        area_file=str(src_datum.COMID[Station_id]) +'.csv'
                        area_df=pd.read_csv(src_path+area_file)
                        dis_src=area_df['Discharge (m3s-1)']
                        area_src = area_df['WetArea (m2)']
                        area_src=area_src**(5/3)
                        idx1=dis_src[dis_src.gt(q_cor/35.3147)].idxmin()
                        if idx1<=1:
                            area_cor =0.0
                        else:
                            area_cor=(((area_src[idx1]-area_src[idx1-1])/(dis_src[idx1]-dis_src[idx1-1]))*(q_cor/35.3147-dis_src[idx1-1])+area_src[idx1-1])**(3/5)
                            area_cor=round(area_cor,4)
                        
                            q_cor=round(q_cor,4)
                        # this gives me area for correction. Next I will add this area to area corresponding to syn stage and recalculate discharge for syn from Q relation with a^5/3  linear interpolation
                        dis1=[]
                        for i in range(len(syn)):
                            new_area=area_df['WetArea (m2)'][(area_df.Stage*3.28084).round(2)==syn[i]].values[0]+ area_cor
                            if len(area_df['WetArea (m2)'][area_df['WetArea (m2)'].gt(new_area)])>0 : #this if statement curbs calculates new discharge for stages only when the new_area exists in the range of area for given SRC
                                id1=area_df['WetArea (m2)'][area_df['WetArea (m2)'].gt(new_area)].idxmin()
                                new_dis= (((dis_src[id1]-dis_src[id1-1])/(area_src[id1]-area_src[id1-1]))*(new_area**(5/3)-area_src[id1-1])+dis_src[id1-1])
                                new_dis=(new_dis*35.3147).round(2)
                                dis1.append(new_dis)
                        
                        #importing original GageRC and then i will extract stage corresponding to new SRC discharge created and then save.
                        
                        #matplotlib.rc('xtick', labelsize=16)
                        #matplotlib.rc('ytick', labelsize=16)
                        #plt.figure();
                        #plt.plot(discharge,gage1,'r-', linewidth=3)
                        #plt.plot(discharge,syn1,'b-',linewidth=3)
                        #plt.plot(discharge+q_cor,syn1,'bo', linewidth=3)
                        #plt.plot(dis1,syn1,'b--', linewidth=3)
                        #dis1
                        #discharge+q_cor
                        #plt.xlabel('Discharge (cfs)', fontsize=20)
                        #plt.legend(('Gage Rating Curve','Synthetic Rating Curve (SRC)','SRC with just discharge shift','Area Corrected SRC'),fontsize=20)
                        #plt.ylabel('Surface Elevation of Water (ft)',fontsize=20)
                        #plt.savefig('a0_rating.png', dpi=300)
                        
                        
                        
                        
                        
                        path=os.path.join(usgs_rc_path,station)
                        usgs_rc=pd.read_csv(path)
                        Gauge_rc=pd.DataFrame()
                        Gauge_rc['Depth']=usgs_rc["INDEP"]
                        Gauge_rc['Flow']=usgs_rc["DEP"]
                        Synth_rc=pd.DataFrame()
                        Synth_rc["Depth_ft"] = syn[:len(dis1)] #this step also curbs the length of syn to match the length of new discharge we get. (in case the last few values of new are exceed the max area range in SRC data)
                        Synth_rc["Discharge_cfs"]=dis1
                        RC_cur_filt = GetSRCinUSGSRange(Gauge_rc,Synth_rc)
                        RC_comp_tmp = InterpolateRC(Gauge_rc,RC_cur_filt)
                        RC_comp_tmp["Gauge_sur_el"]=RC_comp_tmp.Gauge_Depth+gage_height
                        RC_comp_tmp['SRC_sur_el']=syn1
                        RC_comp_tmp["GageID"]= Station_id
                        RC_comp_tmp.to_csv("D:\\ratingcurve\\states\\wa\\area_corrected\\" + station)
                        
                        cid=area_file[:-4]
                        ind1=df_stream.StreamOrder[df_stream.COMID==area_file[:-4]].index.values[0]
                        length_km=df_stream.LENGTH[ind1]
                        stream_order=df_stream.StreamOrder[ind1]
                        slp=df_stream.Slope[ind1]
                        #drain_area=df_gage.Area[df_gage.STAID==station[:-4]].values[0]
                    
                        error_surf=np.sqrt(((RC_comp_tmp.Gauge_sur_el-RC_comp_tmp.SRC_sur_el)**2).mean())
                        nrmse=error_surf/(RC_comp_tmp.Gauge_sur_el.max()-RC_comp_tmp.Gauge_sur_el.min())
                        bias=sum(RC_comp_tmp.Gauge_sur_el-RC_comp_tmp.SRC_sur_el)
                        nbias=bias/(RC_comp_tmp.Gauge_sur_el.max()-RC_comp_tmp.Gauge_sur_el.min())
                        #if nrmse
                        
                        df=df.append([{'Station_id': Station_id,'Discharge_correction(cfs)': q_cor, "Area_correction(m2)" : area_cor, "RMSE_sur_el": error_surf,"NRMSE": nrmse,'COMID':cid,'StreamOrder':stream_order,'Length_km_nhd':length_km, 'Slope_nhd':slp, 'Bias':bias, 'NBias':nbias}], ignore_index= True)
                else:
                    one_reading.append(Station_id)  
        except OSError:
            no_SRC.append(Station_id)
            print("Station_curve_not_found:" + Station_id)
    
    df.to_csv("D:/ratingcurve/states/wa/bathymetric_area_wa.csv", index=False)
    df1=pd.DataFrame(large_var, columns=['Station_id'])
    df1.to_csv("D:/ratingcurve/states/wa/Large_Var_wa.csv", index=False)
    df1=pd.DataFrame(no_SRC, columns=['Station_id'])
    df1.to_csv("D:/ratingcurve/states/wa/NO_SRC_stations_wa.csv", index=False)
    df1=pd.DataFrame(one_reading, columns=['Station_id'])
    df1.to_csv("D:/ratingcurve/states/wa/one_reading_wa.csv", index=False)

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
       
        
        