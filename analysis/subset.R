library(sqldf)
library(data.table)
library(dplyr)

year <- 2013
line_separater <- function(year){
  path_surfix <- "../../TextData/reviews_"
  tmp <- fread(paste0(path_surfix, year, '.tsv'), sep = '\t',na.strings = '')
  tmp <- tmp %>% select(-text,-likes, -cool, -useful, -funny)
  fwrite(tmp, file = paste0(path_surfix, year,'m.tsv'), sep = '\t', eol = '\r\n')
}

for(year in 2013:2017){
  line_separater(year)
}