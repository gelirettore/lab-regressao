library(ggplot2)
library(plyr)
library(readr)
library(ISwR)
attach(usina72)
cor(usina72)
regressao <- read_csv("csv/regressao.csv")
geral = ggplot(regressao)
geral + geom_col(aes(x=regressor, y=f3_mse), fill="darkblue") + 
  labs(title="Erro Quadrado Médio (F3)") + xlab("Regressor")+ylab("MSE")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("mse_f3.png", width = 5, height = 5)
geral + geom_col(aes(x=regressor, y=f5_mse), fill="darkblue") + 
  labs(title="Erro Quadrado Médio (F4)") + xlab("Regressor")+ylab("MSE")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("mse_f5.png", width = 5, height = 5)
geral + geom_point(aes(y=f3_mse, x = f5_mse, color = regressor)) + 
  labs(title="Desempenho dos classificadores")+xlab("MSE (F4)")+ylab("MSE (F3)")
ggsave("regressores.png", width = 5, height = 5)


gb1 <- read_csv("csv/rf1.csv")
f3=ggplot(gb1) +
#  geom_histogram(aes(x=error, y=..density..), color="black", fill=NA,binwidth = 0.1)+
  geom_density(aes(x=error), fill="cyan", alpha=.15)+
  geom_vline(aes(xintercept=mean(error), color="Média"), linetype="dashed")+
  geom_vline(aes(xintercept=mean(error)-abs(sd(gb1$error)), color="Desvio Padrão"), linetype="dashed")+
  geom_vline(aes(xintercept=mean(error)+abs(sd(gb1$error)), color="Desvio Padrão"), linetype="dashed") +
  labs(title="MSE na Base de Validação para a Variável F3") + xlab("Erro") + ylab("Densidade")+
  scale_x_continuous(expand = c(0, 0), limits = c(-0.5, 0.5)) +
  scale_color_manual(name = "Estatística", values = c("Média" = "blue", "Desvio Padrão" = "red"))
f3
ggsave("mse_valf3.png", width = 5, height = 5)

gb2 <- read_csv("csv/rf2.csv")
f5=ggplot(gb2) +
#  geom_histogram(aes(x=error, y=..density..), color="black", fill=NA,binwidth = 0.1)+
  geom_density(aes(x=error), fill="cyan", alpha=.15)+
  geom_vline(aes(xintercept=mean(error), color="Média"), linetype="dashed")+
  geom_vline(aes(xintercept=mean(error)-abs(sd(gb2$error)), color="Desvio Padrão"), linetype="dashed")+
  geom_vline(aes(xintercept=mean(error)+abs(sd(gb2$error)), color="Desvio Padrão"), linetype="dashed")+
  labs(title="MSE na Base de Validação para a Variável F4") + xlab("Erro") + ylab("Densidade")+
  scale_x_continuous(expand = c(0, 0), limits = c(-0.05, 0.05)) +
  scale_color_manual(name = "Estatística", values = c("Média" = "blue", "Desvio Padrão" = "red"))
f5
ggsave("mse_valf5.png", width = 5, height = 5)

gb1t <- read_csv("csv/rf-test1.csv")
f3_test = ggplot(gb1t) +
  #geom_histogram(aes(x=error, y=..density..), color="black", fill=NA,binwidth = 0.1)+
  geom_density(aes(x=error), fill="cyan", alpha=.15)+
  geom_vline(aes(xintercept=mean(error), color="Média"), linetype="dashed")+
  geom_vline(aes(xintercept=mean(error)-abs(sd(gb1t$error)), color="Desvio Padrão"), linetype="dashed")+
  geom_vline(aes(xintercept=mean(error)+abs(sd(gb1t$error)), color="Desvio Padrão"), linetype="dashed") + 
  labs(title="MSE na Base de Teste para a Variável F3") + xlab("Erro") + ylab("Densidade") +
  scale_x_continuous(expand = c(0, 0), limits = c(-1.5, 0.5)) +
  scale_color_manual(name = "Estatística", values = c("Média" = "blue", "Desvio Padrão" = "red"))
f3_test
ggsave("mse_testef3.png", width = 5, height = 5)

gb2t <- read_csv("csv/rf-test2.csv")
f5_test = ggplot(gb2t) +
  #geom_histogram(aes(x=error, y=..density..), color="black", fill=NA,binwidth = 0.1)+
  geom_density(aes(x=error), fill="cyan", alpha=.15)+
  geom_vline(aes(xintercept=mean(error), color="Média"), linetype="dashed")+
  geom_vline(aes(xintercept=mean(error)-abs(sd(gb2t$error)), color="Desvio Padrão"), linetype="dashed")+
  geom_vline(aes(xintercept=mean(error)+abs(sd(gb2t$error)), color="Desvio Padrão"), linetype="dashed")+
  labs(title="MSE na Base de Teste para a Variável F4") + xlab("Erro") + ylab("Densidade")+
  scale_x_continuous(expand = c(0, 0), limits = c(-0.05, 0.05)) +
  scale_color_manual(name = "Estatística", values = c("Média" = "blue", "Desvio Padrão" = "red"))
f5_test
ggsave("mse_testef5.png", width = 5, height = 5)

