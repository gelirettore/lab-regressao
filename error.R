library(ggplot2)
ds <- read_csv("test.csv")
c<-ggplot(data=ds, aes(error))
c+geom_density(kernel="gaussian")

