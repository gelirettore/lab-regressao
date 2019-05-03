library(ggplot2)
library(plyr)
library(readr)
library(ISwR)
attach(usina72)

regressao <- read_csv("csv/regressao.csv")
geral = ggplot(regressao)
geral + geom_col(aes(x=regressor, y=f3_mse)) + labs(title="Erro Quadrado Médio (F3)") + xlab("Regressor")+ylab("MSE")
geral + geom_col(aes(x=regressor, y=f5_mse)) + labs(title="Erro Quadrado Médio (F5)") + xlab("Regressor")+ylab("MSE")
geral + geom_point(aes(y=f3_mse, x = f5_mse, color = regressor))

gb1 <- read_csv("csv/gb1.csv")
f3=ggplot(gb1) +
  geom_histogram(aes(x=error, y=..density..), color="black", fill=NA,binwidth = 0.1)+
  geom_density(aes(x=error), fill="cyan", alpha=.15)+
  geom_vline(aes(xintercept=mean(error)), linetype="dashed")+
  geom_vline(aes(xintercept=mean(error)-abs(sd(gb1$error))), linetype="dashed", color="red")+
  geom_vline(aes(xintercept=mean(error)+abs(sd(gb1$error))), linetype="dashed", color="blue")
f3

gb2 <- read_csv("csv/gb2.csv")
f5=ggplot(gb2) +
  geom_histogram(aes(x=error, y=..density..), color="black", fill=NA,binwidth = 0.1)+
  geom_density(aes(x=error), fill="cyan", alpha=.15)+
  geom_vline(aes(xintercept=mean(error)), linetype="dashed")+
  geom_vline(aes(xintercept=mean(error)-abs(sd(gb2$error))), linetype="dashed", color="red")+
  geom_vline(aes(xintercept=mean(error)+abs(sd(gb2$error))), linetype="dashed", color="blue")
f5

gb1t <- read_csv("csv/gb-test1.csv")
f3_test = ggplot(gb1t) +
  #geom_histogram(aes(x=error, y=..density..), color="black", fill=NA,binwidth = 0.1)+
  geom_density(aes(x=error), fill="cyan", alpha=.15)+
  geom_vline(aes(xintercept=mean(error)), linetype="dashed")+
  geom_vline(aes(xintercept=mean(error)-abs(sd(gb1t$error))), linetype="dashed", color="red")+
  geom_vline(aes(xintercept=mean(error)+abs(sd(gb1t$error))), linetype="dashed", color="blue")
f3_test

gb2t <- read_csv("csv/gb-test2.csv")
f5_test = ggplot(gb2t) +
  #geom_histogram(aes(x=error, y=..density..), color="black", fill=NA,binwidth = 0.1)+
  geom_density(aes(x=error), fill="cyan", alpha=.15)+
  geom_vline(aes(xintercept=mean(error)), linetype="dashed")+
  geom_vline(aes(xintercept=mean(error)-abs(sd(gb2t$error))), linetype="dashed", color="red")+
  geom_vline(aes(xintercept=mean(error)+abs(sd(gb2t$error))), linetype="dashed", color="blue")
f5_test

