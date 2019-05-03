library(ggplot2)
library(plyr)
lr <- read_csv("lr.csv")
c=ggplot(lr) +
  geom_histogram(aes(x=error, y=..density..), color="black", fill=NA,binwidth = 0.1)+
  geom_density(aes(x=error), fill="cyan", alpha=.15)+
  geom_vline(aes(xintercept=mean(error)), linetype="dashed")+
  geom_vline(aes(xintercept=mean(error)-abs(sd(lr$error))), linetype="dashed", color="red")+
  geom_vline(aes(xintercept=mean(error)+abs(sd(lr$error))), linetype="dashed", color="blue")
c
#ver sumaraze/sumarize