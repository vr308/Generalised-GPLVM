
library(ggplot2)
library(lubridate)
library(data.table)

yellow_cabs = fread('~/Downloads/yellow_tripdata_2020-01.csv')
yellow_cabs[, pickup_time := as.POSIXct(tpep_pickup_datetime)]
minute(yellow_cabs$pickup_time) = 0
second(yellow_cabs$pickup_time) = 0

green_cabs = fread('~/Downloads/green_tripdata_2020-01.csv')
green_cabs[, pickup_time := as.POSIXct(lpep_pickup_datetime)]
minute(green_cabs$pickup_time) = 0
second(green_cabs$pickup_time) = 0

fhv = fread('~/Downloads/fhv_tripdata_2020-01.csv', fill=T)
fhv[, pickup_time := as.POSIXct(pickup_datetime)]
minute(fhv$pickup_time) = 0
second(fhv$pickup_time) = 0

data = yellow_cabs[, .(yellow = .N), by=pickup_time]
data = merge(data, green_cabs[, .(green = .N), by=pickup_time])
data = merge(data, fhv[!is.na(pickup_time), .(fhv = .N), by=pickup_time])

ggplot(data) +
	geom_line(aes(x=pickup_time, y=yellow, color='yellow')) +
	geom_line(aes(x=pickup_time, y=green, color='green')) +
	geom_line(aes(x=pickup_time, y=fhv, color='fhv'))

data[, day_time := paste0(day(pickup_time), '_', hour(pickup_time))]

write.csv(data[, .(day_time, yellow, green, fhv)], '', row.names=F)
