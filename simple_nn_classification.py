# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:18:21 2020

@author: aghangha
"""


import random
random.seed(0)

import pandas as pd
import numpy as np
import re
from pandas.api.types import is_string_dtype, is_numeric_dtype
import warnings
from pdb import set_trace
import torch
from torch import nn, optim, as_tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.init import *
import sklearn
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, StandardScaler

import matplotlib.pyplot as plt




#file1=open("output.txt","w")

lparams= ['Elevation','Drain_sqkm','Lattitude','Longitude','Two Year Flow','Water Surface Slope','Percent Impervious','Bathymetric Area']
list_param=['RASTERVALU','Drain_sqkm','LAT','LON','2yrFlow', 'Slope_nhd','per_imp','Area_corre']

#list_param=['RASTERVALU','Drain_sqkm','LAT','LON','2yrFlow', 'Slope_nhd','per_imp']
df=pd.read_csv('D:/ratingcurve/states/merge/nrmse_param_table_2yrflow.csv', converters={1:str,2:str} )
df['rmse_sign']=df['RMSE_sur_e']*df['nbias_sign']



label=[]
# for i in df.rmse_sign:
#     if 1.0 > abs(i) >= 0.0:
#         label.append(0.0)
#     elif 3.0 > abs(i) >= 1.0:
#         label.append(1.0)
#     elif 5.0 > abs(i) >= 3.0:
#         label.append(2.0)
#     elif 7.0 > abs(i) >= 5.0:
#         label.append(3.0)
#     else:
#         label.append(4.0)

for i in df.rmse_sign:
    if 2.0 > abs(i) >= 0.0:
        label.append(0.0)
    elif 4.0 > abs(i) >= 2.0:
        label.append(1.0)
    else:
        label.append(2.0)
 
df['rmse_label']=label



n_label=[]
for i in df.nrmse_sign:
    if -0.75 > i :
        n_label.append(0.0)
    elif -0.50> i >= -0.75:
        n_label.append(1.0)
    elif -0.25> i >= -0.50:
        n_label.append(2.0)
    elif -0.125 > i >= -0.50:
        n_label.append(3.0)
    elif 0.125 > i >= -0.125:
        n_label.append(4.0)
    elif 0.25 > i  >= 0.125:
        n_label.append(3.0)
    elif 0.375 > i >= 0.25:
        n_label.append(2.0)
    elif 0.50  >  i  >=0.375:
        n_label.append(2.0)
    elif 0.75 > i >=0.5:
        n_label.append(1.0)
    else :
        n_label.append(0.0)

df['nrmse_label']=n_label

#df1=df[df.Npoints>=10].copy()
##working on only npoints greater than 10.

df2=df[['rmse_label','RASTERVALU','Drain_sqkm','LAT','LON','2yrFlow', 'Slope_nhd','per_imp','Area_corre']]
       
#df2=df[['rmse_label','RASTERVALU','Drain_sqkm','LAT','LON','2yrFlow', 'Slope_nhd','per_imp']]
    










#split into train validate and test  with 70% train, 15% validate and 15% test
train, validate, test = np.split(df2.sample(frac=1),[int(0.7*len(df2)),int(0.85*len(df2))])


dep_var='rmse_label'

def scale_vars(df,mapper):
    warnings.filterwarnings('ignore',category=sklearn.exceptions.DataConversionWarning)
    if mapper is None:
        map_f = [([n],StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]
        mapper =DataFrameMapper(map_f).fit(df)
        
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper
    




def proc_df(df,cat_vars, cont_vars,y_fld=None,do_scale=False,mapper=None,na_dict=None):

    ignore_flds=[]
    skip_flds=[]
    
    dep_var=y_fld
    df=df[cat_vars + cont_vars + [dep_var]].copy()
    df[dep_var]=df[dep_var].astype(float)
    df=df.copy()
    ignored_flds=df.loc[:,ignore_flds]
    y=df[y_fld].values
    
    #deal with skip fields
    skip_flds += [y_fld]
    df.drop(skip_flds,axis=1,inplace=True)
    # initialise the na dictionary
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
    #normalize
    if do_scale : mapper=scale_vars(df,mapper)
    res=[df,y,na_dict]
    # keep track of how things were normalized
    if do_scale : res = res + [mapper]
    return res



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
    df_log[df_log ==0.0]=0.001 # done so that we dont have any issue with log transformation.
    df_log=np.log(df_log) # log transforming log variables.
    if do_scale : mapper=scale_vars(df_log,mapper)
    i=0
    for x in loc_vars:
        df[x]=(df[x]-loc_min_max[i][0])/(loc_min_max[i][1]-loc_min_max[i][0])
        i+=1
    df['per_imp']=df['per_imp']/100
    df[log_vars]=df_log
    
    res=[df,y,na_dict]
    # keep track of how things were normalized
    if do_scale : res = res + [mapper]
    return res


# def custom_preproc(df, y_fld ,cont_vars, loc_vars=None,loc_min_max=None, do_scale=False,na_dict=None,log_vars=None,mapper=None):
#     ignore_flds=[]
#     skip_flds=[]
    
#     dep_var=y_fld
#     df=df[cont_vars + [dep_var]].copy()
#     df[dep_var]=df[dep_var].astype(float)
#     df=df.copy()
#     ignored_flds=df.loc[:,ignore_flds]
#     y=df[y_fld].values
    
#     #deal with skip fields
#     skip_flds += [y_fld]
#     df.drop(skip_flds,axis=1,inplace=True)
    
    
#     if na_dict is None : na_dict ={}
#     else : na_dict =na_dict.copy()
#     na_dict_initial =na_dict.copy()
#     #fill missing
#     for name, col in df.items():
#         if is_numeric_dtype(col):
#             if pd.isnull(col).sum():
#                 df[name +'_na']=pd.isnull(col)
#                 filler =col.median()
#                 df[name] - col.fillna(filler)
#                 na_dict[name] =filler
#     # keep track of which entries are missing and possibly use them in the model
#     if len(na_dict_initial.keys()) > 0:
#         df.drop([a+'_na' for a in list(set(na_dict.keys())-set(na_dict_initial.keys()))], axis=1, inplace=True)

#     #transformation
    
#     df_log=df[log_vars].copy()
#     df_log['RASTERVALU'][df_log['RASTERVALU']<0.0]=0.001
#     df_log['Area_corre'][df_log['Area_corre']==0.0]=np.nan
#     df_log['Slope_nhd'][df_log['Slope_nhd']==0.00001]=np.nan
#     df_log[df_log ==0.0]=0.001 # done so that we dont have any issue with log transformation.
#     df_log=np.log(df_log)
#     [sl_std,sl_mean]=[df_log.Slope_nhd.std(),df_log.Slope_nhd.mean()]
#     [area_std,area_mean]=[df_log.Area_corre.std(),df_log.Area_corre.mean()]
#     # log transforming log variables.
#     if do_scale : mapper=scale_vars(df_log,mapper)
#     df_log.Area_corre[df_log.Area_corre.isna()]=(np.log(0.001)-area_mean)/area_std
#     df_log.Slope_nhd[df_log.Slope_nhd.isna()]=(np.log(0.00001)-sl_mean)/sl_std
                                                 
#     df[log_vars]=df_log
    
    
    
#     i=0
#     for x in loc_vars:
#         df[x]=(df[x]-loc_min_max[i][0])/(loc_min_max[i][1]-loc_min_max[i][0])
#         i+=1
#     df['per_imp']=df['per_imp']/100
    
#     res=[df,y,na_dict]
#     # keep track of how things were normalized
#     if do_scale : res = res + [mapper]
#     return res


class ColumnarDataset(Dataset):
    """Dataset class for column dataset.
    Args:
       cats (list of str): List of the name of columns contain
                           categorical variables.
       conts (list of str): List of the name of columns which 
                           contain continuous variables.
       y (Tensor, optional): Target variables.
       is_reg (bool): If the task is regression, set ``True``, 
                      otherwise (classification) ``False``.
       is_multi (bool): If the task is multi-label classification, 
                        set ``True``.
    """
    def __init__(self, df, cat_flds, y, is_reg, is_multi):
        df_cat = df[cat_flds]
        df_cont = df.drop(cat_flds, axis=1)
        
        cats = [c.values for n,c in df_cat.items()]
        conts = [c.values for n,c in df_cont.items()]
        
        n = len(cats[0]) if cats else len(conts[0])
        self.cats = np.stack(cats, 1).astype(np.int64) if cats else np.zeros((n,1))
        self.conts = np.stack(conts, 1).astype(np.float32) if conts else np.zeros((n,1))
        self.y = np.zeros((n,1)) if y is None else y
        if is_reg: self.y =  self.y[:,None]
        self.is_reg = is_reg
        self.is_multi = is_multi
    def __len__(self): 
        return len(self.y)
    def __getitem__(self, idx):
        return [self.cats[idx], self.conts[idx], self.y[idx]]



class MixedInputModel(nn.Module):
    """Model able to handle inputs consisting of both categorical and continuous variables.
    Args:
       emb_size (list of int): List of embedding size
       n_cont (int): Number of continuous variables in inputs
       emb_drop (float): Dropout applied to the output of embedding
       out_sz (int): Size of model's output.
       szs (list of int): List of hidden variables sizes
       drops (list of float): List of dropout applied to hidden 
                              variables
       y_range (list of float): Min and max of `y`. 
                                y_range[0] = min, y_range[1] = max.
       use_bn (bool): If use BatchNorm, set ``True``
       is_reg (bool): If regression, set ``True``
       is_multi (bool): If multi-label classification, set ``True``
    """
    def __init__(self, emb_size, n_cont, emb_drop, out_sz, szs, 
                 drops, y_range=None, use_bn=False, is_reg=True, 
                 is_multi=False):
        super().__init__()
        for i,(c,s) in enumerate(emb_size): assert c > 1, f"cardinality must be >=2, got emb_size[{i}]: ({c},{s})"
        if is_reg==False and is_multi==False: assert out_sz >= 2, "For classification with out_sz=1, use is_multi=True"
        self.embs = nn.ModuleList([nn.Embedding(c, s) 
                                      for c,s in emb_size])
        for emb in self.embs: emb_init(emb)
        n_emb = sum(e.embedding_dim for e in self.embs)
        self.n_emb, self.n_cont=n_emb, n_cont
        
        szs = [n_emb+n_cont] + szs
        self.lins = nn.ModuleList([
            nn.Linear(szs[i], szs[i+1]) for i in range(len(szs)-1)])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(sz) for sz in szs[1:]])
        for o in self.lins: kaiming_normal_(o.weight.data)
        self.outp = nn.Linear(szs[-1], out_sz)
        kaiming_normal_(self.outp.weight.data)
        self.emb_drop = nn.Dropout(emb_drop)
        self.drops = nn.ModuleList([nn.Dropout(drop) 
                                        for drop in drops])
        self.bn = nn.BatchNorm1d(n_cont)
        self.use_bn,self.y_range = use_bn,y_range
        self.is_reg = is_reg
        self.is_multi = is_multi
    def forward(self, x_cat, x_cont):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embs)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x2 = self.bn(x_cont)
            # x = torch.cat([x, x2], 1) if self.n_emb != 0 else x2
            x= x2
        for l,d,b in zip(self.lins, self.drops, self.bns):
            x = F.relu(l(x))
            if self.use_bn: x = b(x)
            x = d(x)
        x = self.outp(x)
        if not self.is_reg:
            if self.is_multi:
                x = torch.sigmoid(x)
            else:
                x = F.log_softmax(x, dim=1)
        elif self.y_range:
            x = torch.sigmoid(x)
            x = x*(self.y_range[1] - self.y_range[0])
            x = x+self.y_range[0]
        return x

def emb_init(x):
    x = x.weight.data
    sc = 2/(x.size(1)+1)
    x.uniform_(-sc,sc)



#################

cat_vars=[]
cont_vars=list_param
emb_szs=[]

loc_vars=['LAT','LON']
loc_min_max=[[20,50],[-63,-131,]]
log_vars=['RASTERVALU','Drain_sqkm','2yrFlow','Slope_nhd','Area_corre'] # conus lattitude longitude min max [[Lat_min,Lat_max],[Lon_min.Lon_max]] # longitude coordinates with negative sign

#log_vars=['RASTERVALU','Drain_sqkm','2yrFlow','Slope_nhd']
########################

#df, y, nas, mapper = proc_df(train, cat_vars, cont_vars, dep_var,do_scale=True)
# df_val, y_val, nas, mapper = proc_df(validate, cat_vars, cont_vars, dep_var, do_scale=True, mapper=mapper, na_dict=nas)
# df_test, y_test, nas, mapper = proc_df(test, cat_vars, cont_vars, dep_var, do_scale=True)

df, y, nas, mapper = custom_preproc(train, dep_var, cont_vars,loc_vars=loc_vars, log_vars=log_vars, loc_min_max=loc_min_max ,do_scale=True)
df_val, y_val, nas, mapper = custom_preproc(validate, dep_var, cont_vars, loc_vars=loc_vars, log_vars=log_vars, loc_min_max=loc_min_max ,do_scale=True,mapper=mapper, na_dict=nas)
df_test, y_test, nas, mapper = custom_preproc(test, dep_var, cont_vars,loc_vars=loc_vars, log_vars=log_vars, loc_min_max=loc_min_max ,do_scale=True)


trn_ds = ColumnarDataset(df, cat_vars, y,is_reg=True,is_multi=False)
val_ds = ColumnarDataset(df_val, cat_vars, y_val,is_reg=True,is_multi=False)
test_ds = ColumnarDataset(df_test, cat_vars, y_test,is_reg=True,is_multi=False)

bs = 24
train_dl = DataLoader(trn_ds, bs, shuffle=True)
val_dl = DataLoader(val_ds, bs, shuffle=False)
test_dl = DataLoader(test_ds, len(df_test), shuffle=False)


# def train_model(model, train_dl, val_dl, n_epochs=10, lr=5e-2):
#         "Run training loops."
#         train_loss=[]
#         val_loss=[]
#         numb_epochs=[]
#         correct=0
#         total=0
#         epochs = n_epochs
#         opt = optim.SGD(model.parameters(), lr=lr)
#         loss_func = nn.MSELoss()
#         try:
#             for epoch in range(epochs):
#                 model.train()
#                 for xb1, xb2, yb in train_dl:
#                     preds = model(xb1, xb2)
#                     labels=yb.float()
#                     preds=preds[:,0]
#                     loss = loss_func(preds, labels)
                    
#                     loss.backward()
#                     opt.step()
#                     opt.zero_grad()
#                     predicted=torch.round(preds).float()
#                     # for label, prediction in zip(labels,predicted):
#                     #     print(label,prediction)
#                     #     print(label.long(),prediction.long())
#                     #     set_trace()
#                     #     confusion_matrix[label.long()[0]][prediction.long()[0]] +=1
#                     #set_trace()
#                     total += labels.size(0)
#                     correct += (predicted==labels).sum().item()
                    
#                 model.eval()
#                 with torch.no_grad():
#                     loss_train = sum(loss_func(model(xb1, xb2), 
#                                              yb.float()) 
#                                    for xb1, xb2, yb in train_dl)
#                     loss_val = sum(loss_func(model(xv1, xv2), 
#                                              yv.float()) 
#                                    for xv1, xv2, yv in val_dl)
#                 print('Validation loss:' + str(epoch), loss_val / len(val_dl))
#                 val_loss.append(loss_val/len(val_dl))
#                 numb_epochs.append(epoch)
#                 print('Training loss:' + str(epoch), loss_train /len(train_dl))
#                 train_loss.append(loss_train/len(train_dl))
#                 print('Training Accuracy : ' + str(correct*100/total))
#             plt.figure()
#             plt.plot(numb_epochs,val_loss,'b-',numb_epochs,train_loss,'r-')
#             plt.xlabel('Number of Epochs')
#             plt.ylabel('Loss')
#             plt.title('Validaiton vs Training Loss')
#         except Exception as e:
#             exception = e
#             raise
        
model = MixedInputModel(emb_szs,n_cont=len(df.columns)-len(cat_vars), 
                        emb_drop = 0.04, out_sz = 3, 
                        szs = [4000,2000], drops = [0.001,0.01], 
                        y_range=(0,np.max(y)), use_bn=False, 
                        is_reg=False, is_multi=True)

# train_model(model, train_dl, val_dl, n_epochs=50, lr=5e-4)


# def test_classifier(net):
#     net.eval()
#     loss_func = nn.MSELoss()
#     correct=0
#     total=0
#     confusion_matrix = torch.zeros(10,10)
#     with torch.no_grad():
#         for xt1,xt2,yt in test_dl:
#             predicted=net(xt1,xt2)
#             labels=yt.float()
#             predicted=torch.round(predicted).float()
#             for label, prediction in zip(labels,predicted):
#                 print(label,prediction)
#                 print(label.long(),prediction.long())
#                 #set_trace()
#                 confusion_matrix[label.long()[0]][prediction.long()[0]] +=1
#             total += labels.size(0)
#             correct += (predicted==labels).sum().item()
#         print(confusion_matrix)
#         print('accuracy : ' + str((correct/total)*100) )
#         #loss_test = sum(loss_func(model(xt1, xt2),yt.float()) for xt1, xt2, yt in test_dl)
    



def train_model_class(model, train_dl, val_dl, n_epochs=10, lr=5e-2, is_multi=False, n_classes=0, l_momentum=0.9):
        "Run training loops."
        train_loss=[]
        val_loss=[]
        numb_epochs=[]
        epochs = n_epochs
        opt = optim.SGD(model.parameters(), lr=lr, momentum=l_momentum)
        criterion = nn.CrossEntropyLoss()
        n_class=n_classes
        try:
            for epoch in range(epochs):
                model.train()
                
                running_loss=0.0
                correct=0
                total=0
                for xb1, xb2, yb in train_dl:
                    opt.zero_grad()
                    preds = model(xb1, xb2)
                    labels=yb.long()
                    labels=labels[:,0]    #look if your data needs this step. not necessary                      
                    loss = criterion(preds, labels)                    
                    loss.backward()
                    opt.step()
                    _, predicted = torch.max(preds.data,1)
                    # for label, prediction in zip(labels,predicted):
                    #     confusion_matrix[label][prediction] +=1
                    total += labels.size(0)
                    correct += (predicted==labels).sum().item()
                                    
                model.eval()
                with torch.no_grad():
                    loss_train = sum(criterion(model(xb1, xb2), 
                                       yb.long()[:,0]) for xb1, xb2, yb in train_dl)
                    loss_val = sum(criterion(model(xv1, xv2), 
                                             yv.long()[:,0]) 
                                   for xv1, xv2, yv in val_dl)
                print('Validation loss:' + str(epoch), loss_val / len(val_dl))
                val_loss.append(loss_val/len(val_dl))
                numb_epochs.append(epoch)
                print('Training loss:' + str(epoch), loss_train /len(train_dl))
                train_loss.append(loss_train/len(train_dl))
                print('Training Accuracy : ' + str(correct*100/total))

            plt.figure()
            plt.plot(numb_epochs,val_loss,'b-',numb_epochs,train_loss,'r-')
            plt.xlabel('Number of Epochs')
            plt.ylabel('Loss')
            plt.title('Validaiton vs Training Loss')
        except Exception as e:
            exception = e
            raise


def test_classifier_cross(net):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    correct=0
    total=0
    confusion_matrix = torch.zeros(3,3)
    predicted_labels=[]
    with torch.no_grad():
        running_loss=0.0
        for xt1,xt2,yt in test_dl:
            predicted=net(xt1,xt2)
            labels=yt.long()
            labels=labels[:,0]
            loss = criterion(predicted, labels)
            running_loss += loss.item()
            _, predicted = torch.max(predicted.data,1)
            for label, prediction in zip(labels,predicted):
                confusion_matrix[label][prediction]+=1
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
            predicted_labels.append(predicted)
        print(confusion_matrix)
        print('Testing Loss: ' + str(running_loss/len(test_dl)))
        print('accuracy : ' + str((correct/total)*100) )
    return predicted_labels
        
        #loss_test = sum(loss_func(model(xt1, xt2),yt.float()) for xt1, xt2, yt in test_dl)




train_model_class(model, train_dl, val_dl, n_epochs=300, lr=1e-4, n_classes=3)
print('Testing Loss : ' + str(loss_test/len(test_dl)))



