# Libraries that need to be installed before running the code
install.packages("xlsx")
install.packages("MASS")
install.packages('agricolae')
install.packages("relaimpo")
install.packages("heplots")
install.packages("rpart.plot")
install.packages("rattle")



#install.packages("xlsx")
library(xlsx)
election <- read.xlsx("./Workbook1.xlsx", 1)
str(election)
names(election)
sum(is.na(election))
election<-election[ ,c(1:10)]
sum(is.na(election))

# Stepwise Regression
library(MASS)
fit <- lm(Hillary.percentage~Income.+GDP.+Unemployment.Rate+Education.Level+White.Ratio+M.to.F.Index+LGBT.Ratio,data=election)
step <- stepAIC(fit, direction="both")
step$anova # display results

fit1<- lm(Trump.percentage~Income.+GDP.+Unemployment.Rate+Education.Level+White.Ratio+M.to.F.Index+LGBT.Ratio,data=election)
step1 <- stepAIC(fit1, direction="both")
step1$anova # display results

#Correlation  Analysis 
#install.packages('agricolae')
library(agricolae)
cor(election[,c(4:10)],method="pearson")

#relimp: Relative Contribution of Effects in a Regression Model
#install.packages("relaimpo")
library(relaimpo)
calc.relimp(fit,type=c("lmg","last","first","pratt"),rela=TRUE)
# Bootstrap Measures of Relative Importance (51 samples) 
boot <- boot.relimp(fit, b = 51, type = c("lmg","last", "first", "pratt"), rank = TRUE, diff = TRUE, rela = TRUE)
booteval.relimp(boot) # print result
plot(booteval.relimp(boot,sort=TRUE)) # plot result

calc.relimp(fit1,type=c("lmg","last","first","pratt"),rela=TRUE)
# Bootstrap Measures of Relative Importance (51 samples) 
boot <- boot.relimp(fit1, b = 51, type = c("lmg","last", "first", "pratt"), rank = TRUE, diff = TRUE, rela = TRUE)
booteval.relimp(boot) # print result
plot(booteval.relimp(boot,sort=TRUE)) # plot result

#Linear Regression
#regression for Trump 
rm<-lm(Trump.percentage~Education.Level+White.Ratio+LGBT.Ratio,data=election)
summary(rm)
#plot(rm)

#regression for Hillary
rm2<-lm(Hillary.percentage~Education.Level+White.Ratio+M.to.F.Index+LGBT.Ratio,data=election)
summary(rm2)
#plot(rm2)

#etasq
#install.packages("heplots")
require(heplots)
etasq(rm, anova=TRUE,partial= FALSE)
etasq(rm2, anova=TRUE,partial= FALSE)
#Decision Tree
library(rpart)
#install.packages("rpart.plot")
library(rpart.plot)
#Trump Decision Tree
dt<-rpart(Trump.percentage~Income.+GDP.+Unemployment.Rate+Education.Level+White.Ratio+M.to.F.Index+LGBT.Ratio,data=election)
summary(dt)

#install.packages("rattle")
library(rattle)	
fancyRpartPlot(dt)

#Hillary Decision Tree 
dt2<-rpart(Hillary.percentage~Income.+GDP.+Unemployment.Rate+Education.Level+White.Ratio+M.to.F.Index+LGBT.Ratio,data=election)
summary(dt2)
library(rattle)	
fancyRpartPlot(dt2)

