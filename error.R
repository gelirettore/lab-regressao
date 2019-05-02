library(ggplot2)
ds <- read_csv("test.csv")
c<-ggplot(data=ds, aes(error))
c+geom_density(kernel="gaussian", alpha=0.2) + 
  geom_vline(aes(xintercept=quantile(ds$error)[2]), size=1)+
  geom_vline(aes(xintercept=quantile(ds$error)[4]), size=1)
