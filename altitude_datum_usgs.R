library(readr)
gage <- read_csv("C:/Users/aghangha/Documents/ratingcurve/gage_info_table.csv");
library(dataRetrieval)
STAID<-gage[,2]
#a<-prod(dim(STAID))
#df<-data.frame("site_no"=numeric(),"altitude"=numeric(), "vert_dat"=character(), "hor_dat"=character(),stringsAsFactors=FALSE)
i=1
for (idx in 7358:a) 
{
  
  alt_height <- whatNWISdata(siteNumber=STAID$STAID[idx])$alt_va[1]
  ver_datum <- whatNWISdata(siteNumber=STAID$STAID[idx])$alt_datum_cd[1]
  hor_datum <- whatNWISdata(siteNumber=STAID$STAID[idx])$dec_coord_datum_cd[1]
  if (prod(dim(alt_height))==0) {
    next
  }
  df[idx,]=c(STAID$STAID[idx],alt_height,ver_datum,hor_datum)
}

write.csv(df,"C:/Users/aghangha/Documents/ratingcurve/datum_usgs.csv")