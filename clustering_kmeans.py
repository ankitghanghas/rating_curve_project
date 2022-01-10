# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 14:14:44 2020

@author: aghangha
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
import random
import scipy.stats as st
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
import copy
import hdbscan


random.seed(0)

n_cat=6

lparams= ['Elevation','Drain_sqkm','Lattitude','Longitude','Two Year Flow','Water Surface Slope','Percent Impervious','Bathymetric Area']
list_param=['RASTERVALU','Drain_sqkm','LAT','LON','2yrFlow', 'Slope_nhd','per_imp','Area_corre']


df=pd.read_csv('D:/ratingcurve/states/merge/nrmse_param_table_2yrflow.csv', converters={1:str,2:str} )
#df2=df[['nrmse_sign','RASTERVALU','Drain_sqkm','LAT','LON','2yrFlow', 'Slope_nhd','per_imp','Area_corre']]
df2=df[['RMSE_sur_e','RASTERVALU','Drain_sqkm','LAT','LON','2yrFlow', 'Slope_nhd','per_imp','Area_corre']]

#df2=df[['nrmse_sign','RASTERVALU','Drain_sqkm','2yrFlow', 'Slope_nhd','per_imp','Area_corre']]
df3=df2.copy()

df4=df3.copy()
#Normalise the df3 column values of RasterValu, 2yrFlow, Area_corre

for i in df3.columns[1:]:
    df3[i]=(df3[i]-df3[i].mean())/df3[i].std()

def scale_vars(df,mapper):
    warnings.filterwarnings('ignore',category=sklearn.exceptions.DataConversionWarning)
    if mapper is None:
        map_f = [([n],StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]
        mapper =DataFrameMapper(map_f).fit(df)
        
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper
    


def custom_preproc(df, y_fld ,cont_vars, loc_vars=None,loc_min_max=None, do_scale=False,na_dict=None,log_vars=None,mapper=None):
    ignore_flds=[]
    skip_flds=[]
    
    dep_var=y_fld
    df=df[cont_vars + [dep_var]].copy()
    df[dep_var]=df[dep_var].astype(float)
    df=df.copy()
    ignored_flds=df.loc[:,ignore_flds]
    y=df[y_fld].values
    
    #deal with skip fields
    skip_flds += [y_fld]
    df.drop(skip_flds,axis=1,inplace=True)
    
    
    if na_dict is None : na_dict ={}
    else : na_dict =na_dict.copy()
    na_dict_initial =na_dict.copy()
    #fill missing
    for name, col in df.items():
        if is_numeric_dtype(col):
            if pd.isnull(col).sum():
                df[name +'_na']=pd.isnull(col)
                filler =col.median()
                df[name] - col.fillna(filler)
                na_dict[name] =filler
    # keep track of which entries are missing and possibly use them in the model
    if len(na_dict_initial.keys()) > 0:
        df.drop([a+'_na' for a in list(set(na_dict.keys())-set(na_dict_initial.keys()))], axis=1, inplace=True)

    #transformation
    
    df_log=df[log_vars].copy()
    df_log['RASTERVALU'][df_log['RASTERVALU']<0.0]=0.001
    df_log['Area_corre'][df_log['Area_corre']==0.0]=np.nan
    df_log['Slope_nhd'][df_log['Slope_nhd']==0.00001]=np.nan
    df_log[df_log ==0.0]=0.001 # done so that we dont have any issue with log transformation.
    df_log=np.log(df_log)
    [sl_std,sl_mean]=[df_log.Slope_nhd.std(),df_log.Slope_nhd.mean()]
    [area_std,area_mean]=[df_log.Area_corre.std(),df_log.Area_corre.mean()]
    # log transforming log variables.
    if do_scale : mapper=scale_vars(df_log,mapper)
    df_log.Area_corre[df_log.Area_corre.isna()]=(np.log(0.001)-area_mean)/area_std
    df_log.Slope_nhd[df_log.Slope_nhd.isna()]=(np.log(0.00001)-sl_mean)/sl_std
                                                 
    df[log_vars]=df_log
    
    
    
    i=0
    for x in loc_vars:
        df[x]=(df[x]-loc_min_max[i][0])/(loc_min_max[i][1]-loc_min_max[i][0])
        i+=1
    df['per_imp']=df['per_imp']/100
    
    res=[df,y,na_dict]
    # keep track of how things were normalized
    if do_scale : res = res + [mapper]
    return res

# tau=[]
# p_val=[]

# for i in range(df3.shape[1]-1):
#    t,p=st.kendalltau(df3.iloc[:,0],df3.iloc[:,i+1])
#    tau.append(t)
#    p_val.append(p)
    
   
# df_correlation=pd.DataFrame({'Parameter': lparams,'KendallTau':tau,'Pvalue':p_val})
# df_correlation.to_csv('D:/ratingcurve/states/merge/kendall_correlation_firstseven_param.csv', index=False)

# centroid, label = kmeans2(df3, n_cat, minit='points') # apply k means to cluster the nrmse with sign (input matrix df3 of dimensions m*n where m is the number of samples and n is the number of dimensions to classify)
# df3['label']=label

dep_var='RMSE_sur_e'
df3, y, nas, mapper = custom_preproc(df3, dep_var, cont_vars,loc_vars=loc_vars, log_vars=log_vars, loc_min_max=loc_min_max ,do_scale=True)




neigh=NearestNeighbors(n_neighbors=2)
nbrs=neigh.fit(df3[:].values)
distances, indices =nbrs.kneighbors(df3[:].values)
distances =np.sort(distances, axis=0)
distances=distances[:,1];plt.plot(distances, c='red');plt.xlabel('sorted ascending, data index');plt.ylabel('intersample distance in feature space');plt.title('Nearest Neighbour')



db_clusters=DBSCAN(eps=2.5, min_samples=20).fit(df3)# for DBSCAN clustering I need more information. maybe try adaptive dbscan # i need more specific info on eps and min_samples. not that big a problem in k-means algorithm here.
df4['label']=db_clusters.labels_ +1

gdf=gpd.GeoDataFrame(df4,geometry=gpd.points_from_xy(df4.LON, df4.LAT))

gdf.plot(column='label', legend=True)




# db_clusters=DBSCAN(eps=3, min_samples=10).fit(df2)    # for DBSCAN clustering I need more information. maybe try adaptive dbscan # i need more specific info on eps and min_samples. not that big a problem in k-means algorithm here.
# gdf['db_clus']=db_clusters.labels_

cluster_metrics=pd.DataFrame(0,index=range(0,3), columns=['Mean','Median','SD'])
plt.figure()


for i in range(3):
    cluster_metrics.iloc[i,0]=df3[df3.label==i]['nrmse_sign'].mean()
    cluster_metrics.iloc[i,1]=df3[df3.label==i]['nrmse_sign'].median()
    cluster_metrics.iloc[i,2]=df3[df3.label==i]['nrmse_sign'].std()
    plt.boxplot(df3[df3.label==i]['nrmse_sign'], positions =[i])
    

plt.show()

groups=df4.groupby("label")

   
# # df_correlation.to_csv('D:/ratingcurve/states/merge/kendall_correlation_firstseven_param.csv', index=False)
# index=1
# plt.figure()

# marker_list=['X','o','v']
# color_list=['b','g','r']
# label_list=['0-1 ft', '1-3 ft', ' > 3 ft']
# inv_int=[2,1,0]
# rmselabel_list=[0.0,1.0,2.0]
# for i in range(len(list_param)): 
#     plt.subplot(3,3,index)
#     plt.ylim((-10,25))
#     for id1 in range(3):
#         id1=inv_int[id1]
#         plt.scatter(df[df['rmse_label']==rmselabel_list[id1]][list_param[i]],df[df['rmse_label']==rmselabel_list[id1]]["rmse_sign"], marker = marker_list[id1], label=label_list[id1],color=color_list[id1],facecolors='none')
#     plt.xlabel(lparams[i],fontsize=20)
#     plt.ylabel('RMSE(with sign for bias)',fontsize=20)
#     # plt.title(lparams[i])
#     plt.legend()
#     index += 1
#     # plt.savefig('D:/ratingcurve/states/merge/correlation_figs/'+lparams[i]+'.png')
# plt.subplots_adjust(wspace=0.2, hspace=0.2)


index=1
loc=[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
fig, axes =plt.subplots(3,3,sharey=True, figsize=(18,10))

marker_list=['X','o','v']
color_list=['b','g','r']
label_list=['0-2 ft', '2-3 ft', ' > 3 ft']
inv_int=[2,1,0]
rmselabel_list=[0.0,1.0,2.0]
for i in range(len(list_param)): 
    loc_id=loc[i]
    ax=axes[loc_id[0],loc_id[1]]
    ax.set_ylim((-10,25))
    for id1 in range(3):
        id1=inv_int[id1]
        ax.scatter(df[df['rmse_label']==rmselabel_list[id1]][list_param[i]],df[df['rmse_label']==rmselabel_list[id1]]["rmse_sign"], marker = marker_list[id1], label=label_list[id1],color=color_list[id1],facecolors='none')
    ax.set_xlabel(lparams[i],fontsize=20)
    #plt.ylabel('RMSE(with sign for bias)',fontsize=20)
    # plt.title(lparams[i])
    # ax.legend()
    index += 1

    # plt.savefig('D:/ratingcurve/states/merge/correlation_figs/'+lparams[i]+'.png')
handles,labels =ax.get_legend_handles_labels()

axes[2,2].axis('off')
fig.tight_layout()
fig.legend(handles,labels,loc='lower right', fontsize=20)
fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.ylabel('RMSE (with sign for bias)', fontsize=20)

fig.subplots_adjust(left=0.055, wspace=0.09, hspace=0.350)

plt.savefig('rmse_params_variation.png', dpi=300)



####spectral clustering


spectral = SpectralClustering(n_clusters=6,assign_labels="kmeans",affinity = 'rbf',
                              gamma = 100,n_neighbors = 200,random_state=230).fit(df3.values)




######plot 

matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)
plt.figure();
plt.scatter(df.LON[df.rmse_label==0], df.LAT[df.rmse_label==0], marker='+', color='b')
plt.scatter(df.LON[df.rmse_label==1], df.LAT[df.rmse_label==1], marker='o', color='g', facecolors='none')
plt.scatter(df.LON[df.rmse_label==2], df.LAT[df.rmse_label==2], marker='v', color='r', facecolors='none')
plt.xlabel('LONGITUDE', fontsize=20)
plt.ylabel('LATTITUDE', fontsize=20)
plt.legend((' 0-1 ft', '1-3 ft', ' > 3 ft'), fontsize=20)
plt.title('RMSE(ft) Surface Water Elevation of Rating Curves', fontsize=20)






#############################
plt.figure(figsize=(6,6))
plt.hist(df.rmse_sign,bins=range(-25,25,1))
plt.xlabel('RMSE', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.title('Histogram of RMSE', fontsize=20)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15)
plt.savefig('hist_rmse.png', dpi=300)




#########################

index=1
fig, axes =plt.subplots(1,3, figsize=(18,5))
marker_list=['P','o','^']
color_list=['b','g','r']
label_list=['0-2 ft', '2-3 ft', ' > 3 ft']
inv_int=[2,0,1]
rmselabel_list=[0.0,1.0,2.0]
pairs=[['RASTERVALU','Slope_nhd'],['Drain_sqkm','per_imp'],['Area_corre','2yrFlow']]
xlabel_list=['Elevation (m)','Drainage Area ($km^2$)','Bathymetry Area ($m^2$)']
ylabel_list=['Slope of NHD','Percent Impervious(%)', 'Two Year Flow (cms)']
for i in range(3): 
    ax=axes[i]
    for id1 in range(3):
        id1=inv_int[id1]
        ax.scatter(df[df['rmse_label']==rmselabel_list[id1]][pairs[i][0]],df[df['rmse_label']==rmselabel_list[id1]][pairs[i][1]], marker = marker_list[id1], label=label_list[id1],color=color_list[id1],facecolors='none')
    ax.set_xlabel(xlabel_list[i],fontsize=20)
    ax.set_ylabel(ylabel_list[i],fontsize=20)
    index += 1

    # plt.savefig('D:/ratingcurve/states/merge/correlation_figs/'+lparams[i]+'.png')
handles,labels =ax.get_legend_handles_labels()
fig.tight_layout()
#fig.legend(handles,labels,loc='lower center', fontsize=20)
fig.legend(handles,labels,loc='lower center', fontsize=20, bbox_to_anchor=(0.5,0), ncol=3)
fig.subplots_adjust(left=0.057,  bottom=0.275, wspace=0.296)