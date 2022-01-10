# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:33:11 2020

To download NED 10m DEM from USGS

Sayan's code modified and used


@author: aghangha
"""
import os, ftplib, zipfile
import numpy as np


# USER DEFINED FUNCTION FOR DOWNLOADING TILES
def DownloadNED(lat,lon):
    lat = str(lat)
    if lon<100:
        lon = "0"+str(lon)
    else:
        lon = str(lon)    
    name = "n"+lat+"w"+lon
    print("Connecting to server: rockyftp.cr.usgs.gov")
    with ftplib.FTP('rockyftp.cr.usgs.gov') as ftp:
        try:
            ftp.login()
            ftp.cwd('vdelivery/Datasets/Staged/Elevation/13/ArcGrid/')
            contents = ftp.nlst()
            filtered_contents = [f for f in contents if ((name in f) & (".zip" in f))]
            if len(filtered_contents) == 0:
                name = "n"+lat+"w"+lon
                name = "USGS_NED_13_"+name+"_ArcGrid"
                filtered_contents = [f for f in contents if ((name in f) & (".zip" in f))]
                if len(filtered_contents) == 0:
                    print("No file found for: " + name)
                    
                
                
                else:
                    final_file = filtered_contents[0]
                    print("1 file found for current tile")
                    print("Downloading " + final_file)
                    fo = open(os.path.join(work_folder_name, final_file), 'wb')
                    ftp.retrbinary("RETR " + final_file , fo.write)
                    fo.close()
                    print("Download successful")
                    return(final_file)
                
            elif len(filtered_contents) == 1:
                final_file = filtered_contents[0]
                print("1 file found for current tile")
                print("Downloading " + final_file)
                fo = open(os.path.join(work_folder_name, final_file), 'wb')
                ftp.retrbinary("RETR " + final_file , fo.write)
                fo.close()
                print("Download successful")
                return(final_file)                
            elif len(filtered_contents) > 1:
                print("More than 1 file found: Dowloading largest zip file")
                file_list = []
                ftp.sendcmd("TYPE i")
                for f in filtered_contents:
                    file_list.append((f,ftp.size(f)))
                file_list.sort(key=lambda s: s[1])
                final_file = file_list[-1]#return the largest file
                if len(final_file)>1:
                    final_file=final_file[0]
                print("Downloading..." + final_file)
                fo = open(os.path.join(work_folder_name, final_file), 'wb')
                ftp.retrbinary("RETR " + final_file , fo.write)
                fo.close()
                print("Download successful")
                return(final_file)
            else:
                print("Unknown error with file download for:" + name)          
           
        except ftplib.all_errors as e:
            print('FTP error:', e)
   
# USER DEFINED FUNCTION FOR UNZIPPING DOWNLOADED TILES
def UnzipNED(f):
    try:
        zfile1 = zipfile.ZipFile(work_folder_name + "/" + f, 'r')
        name=f
        if len(f)>12: # this will unzip all files in common format so that I have all files named like n46w112
            name=f[12:23]            
        path=os.path.join(work_folder_name,name[:-4])
        
        zfile1.extractall(path)
        zfile1.close()
        print("NED unzipped successfully: " + f)
    except:
        print("Error in unzipping: " + f)

work_folder_name = r'D:\ratingcurve\DEM\tiles\northmidwest'

#specify lat and long list

# # here i have specified for only Indiana
# lat_list=[38,39,40,41,42]
# lon_list=[85,86,87,88,89]


# list of lat long for Texas
#lat_list=[28,29,30,31,32,33,34,35,36,37]
#lon_list=[93,94,95,96,97,98,99,100,101,102,103,104,105,106,107]

#list of lat long for california

# ALTERED AS STARTING FROM 37 OTHER WISE STARTS AT 33
# lat_list=[i for i in np.arange(33,44)]
# lon_list = [i for i in np.arange(115,126)]

# northeast lat long

# lat_list=[i for i in np.arange(37,49)]
# lon_list= [i for i in np.arange(67,86)]


#southeast lat long

# lat_list=[i for i in np.arange(26,37)]
# lon_list= [i for i in np.arange(76,93)]

# # north mid east
# lat_list=[i for i in np.arange(40,51)]
# lon_list= [i for i in np.arange(86,97)]


# # washington state
# lat_list=[i for i in np.arange(48,50)]
# lon_list= [i for i in np.arange(115,126)]

# nothmidwest lat long

lat_list=[i for i in np.arange(37,51)]
lon_list=[i for i in np.arange(98,115)]


# # southmid lat long
# lat_list=[i for i in np.arange(31,37)]
# lon_list=[i for i in np.arange(108,115)]


for i in lat_list:
    for j in lon_list:
      a=DownloadNED(i, j)
      if a != None :
          UnzipNED(a)    

          
  


