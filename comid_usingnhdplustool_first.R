# Returns and file with USGS gage id, huc02 unit, lat and long and comid associated with the USGSID using a tool nhdplusTools
#


library(readr)
gage <- read_csv("C:/Users/aghangha/Documents/ratingcurve/gage_info_table.csv");
library(nhdplusTools)
STAID<-gage[,2] # gets station id number
a<-prod(dim(STAID))
b<-".csv"
df<- data.frame(USGSID=character(), HUC02=character(), LAT_GAGE=numeric(), LNG_GAGE=numeric(), COMID=character(),stringsAsFactors=FALSE)
for (idx in 1:a) 
{
  id<-paste("USGS","-",STAID$STAID[idx], sep="")
  nldi_nwis <- list(featureSource = "nwissite", featureID = id)
  res <- try((discover_nhdplus_id(nldi_feature = nldi_nwis)), silent = T) # checks for error in the retrieval or what everinput we give
  if(inherits(res, "try-error")){next}# if there is an error then it skips that iteration and moves on
  comid<-toString(discover_nhdplus_id(nldi_feature = nldi_nwis)) # extracts comid and converts it to string
  df[idx,1]<-gage[idx,2]
  df[idx,2]<-gage[idx,4]
  df[idx,3]<-gage[idx,5]
  df[idx,4]<-gage[idx,6]
  df[idx,5]<-comid
}
df<-na.omit(df)
write.csv(df,"C:/Users/aghangha/Documents/ratingcurve/gage_comid_nhdtools.csv")