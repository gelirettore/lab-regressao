library(ggplot2)
lr <- read_csv("lr.csv")
c<-ggplot(data=lr, aes(error))
c+geom_density(kernel="gaussian", alpha=0.2) + 
  geom_vline(aes(xintercept=quantile(lr$error)[2]), size=1)+
  geom_vline(aes(xintercept=quantile(lr$error)[4]), size=1)
#ver sumaraze/sumarize