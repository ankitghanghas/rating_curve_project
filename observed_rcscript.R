library(readr)
gage <- read_csv("C:/Users/aghangha/Documents/ratingcurve/gage_info_table.csv");
library(dataRetrieval)
STAID<-gage[,2]
a<-prod(dim(STAID))
b<-".csv"
for (idx in 1:a) 
{
  data <- readNWISrating(STAID[idx,1],"exsa");
  if (prod(dim(data))==0) {
    next
  }
  n<- paste(STAID[idx,1],b,sep="")
  fname <- paste("~/Downloads/rating/",n,sep="")
}
