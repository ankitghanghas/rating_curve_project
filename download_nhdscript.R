library(nhdplusTools)
huclist<-c("01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18")
nhddir<-"D:/Masters/RC/data/nhd"
download_nhdplushr(nhddir,huclist, download_files = True)