##########################################################################################################
## A semi-automated approach to classify and map ecological zones across the dune-beach interface	    	##
## date:  04 Nov. 2017 																					                                        ##
## By: Pablo D. B. Guilherme¹²*; Carlos A. Borzone¹²; André A. Padial¹; Linda R. Harris³ 			        	##
##                                                                                                      ##
## ¹ Programa de Pós-Graduação em Ecologia e Conservação, Universidade Federal do Paraná.               ##
##   Centro Politécnico, Curitiba - PR, CEP 81531-990, Caixa Postal 19031, Brasil.                      ##
## ² Laboratório de Ecologia de Praias Arenosas, Universidade Federal do Paraná. Centro de Estudos do   ##
##   Mar, Pontal do Paraná - PR, CEP 83255-976, Caixa Postal 61, Brasil.                                ##
## ³ Institute for Coastal and Marine Research, and Department of Zoology, PO Box 77000, Nelson Mandela ##
##   University, Port Elizabeth 6031, South Africa.                                                     ##
##########################################################################################################


##############################################################
## data_sub-environments.csv contains/ Table 2:             ##
##															                            ##
## 1.  sec: Sector (Categorical)						              	##
## 2.  trans: Transect (Categorical)					            	##
## 3.  sub: Sub-environments classification 			        	##
## 4.  SLO: Local slope (dimensionless)				          		##
## 5.  SVC: Segmented Vegetation Cover (%)        					##
## 6.  DCR: Drift Cover Rate (%)		              					##
## 7.  ENT: Entropy from texture (dimensionless)		      	##
## 8.  AVH:	Average Vegetation height (cm)	        				##
## 9.  NVC:	Number of points with vegetation	        			##
## 10. NSC:	Number of points with sand or marine litter	  	##
## 11. ipe:	Number of points with Ipomoea pes-caprae	    	##
## 12. ipo1: Number of points with Ipomoea imperati			    ##
## 13. gaam: Number of points with Gamothea americana		    ##
## 14. blpo: Number of points with Blutaparon portulacoides	##
## 15. pol:	Number of points with Polygala cyparissias		  ##
## 16. pan: Number of points with Panicum racemosum		    	##
## 17. spor: Number of points with Sporobolus virginicus	  ##
## 18. pas: Number of points with Paspalum vaginatum		    ##
## 19. api:	Number of points with Apium leptophyllum		    ##
## 20. dal:	Number of points with Dalbergia ecastophyllum	  ##
## 21. hydro: Number of points with Hydrocotyle bonariensis	##
## 22. steno: Number of points with Stenotaphrum secundatum	##
## 23. cord: Number of points with Cordia verbenacea		    ##
## 23. herb: Number of points with herbaceous plants		    ##
## 23. grass: Number of points with grass plants			      ##
## 23. shrub: Number of points with shrub plants		      	##
## 23. S: Species Richness									                ##
##                                                          ##
##############################################################


##############################################################
#packages required			                    								##			
##############################################################

library(dplyr)
library(ggplot2) #plot
library(gridExtra) #plot
library(cowplot)
library(vegan) 
library(corrgram)
library(RColorBrewer)

#machine learning library
library(caret);#machinelearning
library(e1071) #rpart
library(randomForest) #RF
library(gbm) #gbm
library(kernlab) #svmRadial
library(C50) #C5.0
library(vbmp) #vbmpRadial

#https://stackoverflow.com/questions/19012529/correlation-corrplot-configuration
source("panel.shadeNtext.R")
source("panel.signif.R")

set.seed(100)
data_sub <- read.csv2("data_sub-environments.csv")

##########################################################################################################
## Pre-processing of numeric predictor variables to select the final model							              	##
##########################################################################################################

#Figure 3.A
table(data_sub$sub)

#Figure 3.B
str(data_sub)

#Figure 3.C
#searching for and removing variables that were near-zero-variance predictors
nzv <- nearZeroVar(data_sub)
nzv.results<- nearZeroVar(data_sub, saveMetrics= TRUE)
nzv.results$variable <- rownames(nzv.results)
nzv.results

#remove data from the original worksheet
data_sub2 <- data_sub[, -nzv]

##########################################################################################################
## Searching for and removing variables that were Multi-collinearity predictors	/ Figure 3.D						##
##########################################################################################################

#Figure S1 
a<-ggplot(nzv.results, aes(x = factor(variable), y = freqRatio, fill=factor(nzv))) +
  geom_bar(stat = "identity", color="black") +
  scale_fill_manual(values=c("black", "gray"))+
  scale_color_manual(values=c("black", "gray"))+
  theme_minimal()

b<-ggplot(nzv.results, aes(x = factor(variable), y = percentUnique, fill=factor(nzv))) +
  geom_bar(stat = "identity", color="black") +
  scale_fill_manual(values=c("black", "gray"))+
  scale_color_manual(values=c("black", "gray"))+
  theme_minimal()

grid.arrange(a, b, ncol = 1, nrow = 2)

descrCor <- cor(data_sub2[, 4:16])
highlyCorDescr <- findCorrelation(descrCor, cutoff = .9)
data_sub3<-data.frame(data_sub2[1:3],data.frame(data_sub2[, 4:16])[,-highlyCorDescr])

#Figure S2
corrgram(data_sub2, type="data", lower.panel=panel.shadeNtext, 
         upper.panel=panel.signif, cor.method="pearson")

# Inspect our data set after pre-prossesing
dim(data_sub3)
names(data_sub3);
head(data_sub3);

##########################################################################################################
## Testing and selecting a machine-learning algorithm to classify sub-environments	          					##
## Split our data into a training (70%) and test (30%) set. / Figure 3.F									          		##
##########################################################################################################

training_split = createDataPartition(y = data_sub3$sub, p = 0.70, list = FALSE);
training_set = data_sub3[training_split,];
testing_set = data_sub3[-training_split,];

#Formula model
formula.p <- sub ~ sec+SLO+SVC+DCR+ENT+AVH+NSC+ipe+blpo+pan+herb+grass+S

#seed selection to guarantee the reproducible
set.seed(123)
seeds <- vector(mode = "list", length = 101)
for(i in 1:100) seeds[[i]] <- sample.int(1000, 22)
seeds[[101]] <- sample.int(1000, 1)

# generates parameters that further control how models are created
trControl = trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = T, seeds = seeds)

set.seed(100)

# Train our model. / Figure 3.E	and Table 1
model_fit1 = train(formula.p, method = "rpart", data = training_set, trControl = trControl, preProc = c("center","scale"))
model_fit2 = train(formula.p, method = "rf", data = training_set, trControl = trControl, preProc = c("center","scale"))
model_fit3 = train(formula.p, method = "gbm", data = training_set, trControl = trControl, preProc = c("center","scale"))
model_fit4 = train(formula.p, method = "svmRadial", data = training_set, trControl = trControl, preProc = c("center","scale"))
model_fit5 = train(formula.p, method = "C5.0", data = training_set, trControl = trControl, preProc = c("center","scale"))
model_fit6 = train(formula.p, method = "multinom", data = training_set, trControl = trControl, preProc = c("center","scale"))
model_fit7 = train(formula.p, method = "vbmpRadial", data = training_set, trControl = trControl, preProc = c("center","scale"))

#comparing algorithms
resampls = resamples(list(rpart = model_fit1, 
                          RF = model_fit2, 
                          GBM = model_fit3, 
                          SVM = model_fit4, 
                          C50 = model_fit5, 
                          MREG = model_fit6, 
                          VBMP = model_fit7))
summary(resampls)

#Figure 4
dotplot(resampls, metric = c("Kappa", "Accuracy"))


# pair-wise differences between algorithms (Bonferroni correction)
difValues = diff(resampls)
summary(difValues)
dotplot(difValues, metric = "Accuracy")
dotplot(difValues, metric = "Kappa")

#confusion matrix from the training-testing data 
#classify from our reserved test set.
testing_set_1 = predict(model_fit1, newdata = testing_set);
testing_set_2 = predict(model_fit2, newdata = testing_set);
testing_set_3 = predict(model_fit3, newdata = testing_set);
testing_set_4 = predict(model_fit4, newdata = testing_set);
testing_set_5 = predict(model_fit5, newdata = testing_set);
testing_set_6 = predict(model_fit6, newdata = testing_set);
testing_set_7 = predict(model_fit7, newdata = testing_set);

# Verifying our model from the classifications.
table(testing_set_1, testing_set$sub);
table(testing_set_2, testing_set$sub);
table(testing_set_3, testing_set$sub);
table(testing_set_4, testing_set$sub);
table(testing_set_5, testing_set$sub);
table(testing_set_6, testing_set$sub);
table(testing_set_7, testing_set$sub);

#confusion matrix / Figure 3.G and H
cmat_1 <- confusionMatrix(testing_set_1, testing_set$sub)
cmat_2 <- confusionMatrix(testing_set_2, testing_set$sub)
cmat_3 <- confusionMatrix(testing_set_3, testing_set$sub)
cmat_4 <- confusionMatrix(testing_set_4, testing_set$sub)
cmat_5 <- confusionMatrix(testing_set_5, testing_set$sub)
cmat_6 <- confusionMatrix(testing_set_6, testing_set$sub)
cmat_7 <- confusionMatrix(testing_set_7, testing_set$sub)
cmat_1
cmat_2
cmat_3
cmat_4
cmat_5
cmat_6
cmat_7


##########################################################################################################
## Including the classifications in the original worksheet											                      	##
##########################################################################################################

data_sub4<-data_sub3
data_sub4$RPART <- predict(model_fit1, newdata = data_sub4)
data_sub4$RF <- predict(model_fit2, newdata = data_sub4)
data_sub4$GBM <- predict(model_fit3, newdata = data_sub4)
data_sub4$SVM <- predict(model_fit4, newdata = data_sub4)
data_sub4$C50 <- predict(model_fit5, newdata = data_sub4)
data_sub4$MREG <- predict(model_fit6, newdata = data_sub4)
data_sub4$VBMP <- predict(model_fit7, newdata = data_sub4)

#confusion matrix two best models
#RF
table(data_sub4$RF, data_sub4$sub)
cmalldata.rf <- confusionMatrix(data_sub4$RF, data_sub4$sub)
cmalldata.rf

#C50
table(data_sub4$GBM, data_sub4$sub);
cmalldata.gbm <- confusionMatrix(data_sub4$GBM, data_sub4$sub)
cmalldata.gbm

##########################################################################################################
## Running training, via the randomForest function, to access the										                    ##
## predictive variables' relative influence and Partial dependence plots						                		##
##########################################################################################################

sub_rf <- randomForest(sub ~ ., training_set[,-2],
                         importance=TRUE, mtry=7, 
                         ntree=500, proximity=T)


#predictive variables' relative influence / Figure 3.I
imp.rf <- as.data.frame(importance(sub_rf))
row.names(imp.rf) -> imp.rf$var
imp.rf <- imp.rf[order(imp.rf$IF), ]
imp.rf$var <- factor(imp.rf$var, levels = imp.rf$var[order(imp.rf$MeanDecreaseAccuracy)])
imp.rf

#Figure 5

co0 <- ggplot(imp.rf, aes(x = var, y = MeanDecreaseAccuracy)) +
  theme_bw() +
  geom_bar(stat = "identity", fill="black") +
  coord_flip()+
  scale_y_continuous(breaks=seq(0, 70, by = 10), limits = c(0, 75))+
  theme(axis.text = element_text(size = 11, colour = "black"),
        panel.grid = element_blank(),
        panel.border = element_blank(), 
        axis.line = element_line())+
  xlab("Predictor variables") +
  ylab(NULL)

co1 <- ggplot(imp.rf, aes(x = var, y = ML)) +
  theme_bw() +
  geom_bar(stat = "identity", fill="#8c510a", col="black") +
  coord_flip() +
  scale_y_continuous(breaks=seq(0, 70, by = 10), limits = c(0, 75))+
  theme(axis.text = element_text(size = 11, colour = "black"),
        panel.grid = element_blank(),
        panel.border = element_blank(), 
        axis.line = element_line(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank())+
  xlab("Predictor variables") +
  ylab(NULL)

co2 = ggplot(imp.rf, aes(x = var, y = DZ)) +
  theme_bw() +
  geom_bar(stat = "identity", fill="#d8b365", col="black") +
  coord_flip() +
  scale_y_continuous(breaks=seq(0, 70, by = 10), limits = c(0, 75))+
  theme(axis.text = element_text(size = 11, colour = "black"),
        panel.grid = element_blank(),
        panel.border = element_blank(), 
        axis.line = element_line(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank())+
  xlab("Predictor variables") +
  ylab(NULL)


co3 = ggplot(imp.rf, aes(x = var, y = SL)) +
  theme_bw() +
  geom_bar(stat = "identity", fill="#f6e8c3", col="black") +
  coord_flip() +
  scale_y_continuous(breaks=seq(-5, 35, by = 10), limits = c(-5, 35))+
  theme(axis.text = element_text(size = 11, colour = "black"),
        panel.grid = element_blank(),
        panel.border = element_blank(), 
        axis.line = element_line())+
  xlab("Predictor variables") +
  ylab(NULL)

co4 = ggplot(imp.rf, aes(x = var, y = IF)) + theme_bw() +
  geom_bar(stat = "identity", fill="#c7eae5", col="black") +
  coord_flip()+
  scale_y_continuous(breaks=seq(-5, 35, by = 10), limits = c(-5, 35))+
  theme(axis.text = element_text(size = 11, colour = "black"),
        panel.grid = element_blank(),
        panel.border = element_blank(), 
        axis.line = element_line(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank())+
  xlab("Predictor variables") +
  ylab(NULL)

co5 = ggplot(imp.rf, aes(x = var, y = EF)) + theme_bw() +
  geom_bar(stat = "identity", fill="#5ab4ac", col="black") +
  coord_flip()+
  scale_y_continuous(breaks=seq(-5, 35, by = 10), limits = c(-5, 35))+
  theme(axis.text = element_text(size = 11, colour = "black"),
        panel.grid = element_blank(),
        panel.border = element_blank(), 
        axis.line = element_line(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank())+
  xlab("Predictor variables") +
  ylab(NULL)

co6 = ggplot(imp.rf, aes(x = var, y = RE)) +
  theme_bw() +
  geom_bar(stat = "identity", fill="#01665e", col="black") +
  coord_flip() +
  scale_y_continuous(breaks=seq(-5, 35, by = 10), limits = c(-5, 35))+
  theme(axis.text = element_text(size = 11, colour = "black"),
        panel.grid = element_blank(),
        panel.border = element_blank(), 
        axis.line = element_line(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank())+
  xlab("Predictor variables") +
  ylab(NULL)


first_row = plot_grid(co0, co1, co2, labels=c("A (All)", "B (ML)", "C (DZ)"), nrow = 1, rel_widths = c(2.5, 1, 1),label_x = 0.4, label_y = 0.25)
second_row = plot_grid(co3, co4, co5, co6, labels=c("D (SL)", "E (IF)", "F (EF)", "G (RE)"), nrow = 1, rel_widths = c(1.4, 1, 1,1),label_x = 0.4, label_y = 0.25)
gg_all = plot_grid(first_row, second_row, labels=c('', ''), ncol=1)
gg_all

p2 <- add_sub(gg_all, "Variable Importance")
ggdraw(p2)

#Figure 6
#Partial dependence plots		
par(mfrow=c(2, 3))

partialPlot(sub_rf, training_set[,-2], ENT, "ML", lwd=3,col="#8c510a", main=NULL, xlab="ENT", ylab="f(N) partial dependency", rug=F, ylim=c(-20, 20), xlim=c(5.5, 11))
partialPlot(sub_rf, training_set[,-2], ENT, "DZ", lwd=3,col="#d8b365", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], ENT, "SL", lwd=3,col="#f6e8c3", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], ENT, "IF", lwd=3,col="#c7eae5", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], ENT, "EF", lwd=3,col="#5ab4ac", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], ENT, "RE", lwd=3,col="#01665e", add=T, rug=F)

partialPlot(sub_rf, training_set[,-2], DCR, "ML", lwd=3,col="#8c510a", main=NULL, xlab="DCR", ylab="", rug=F, ylim=c(-20, 12), xlim=c(0, 42))
partialPlot(sub_rf, training_set[,-2], DCR, "DZ", lwd=3,col="#d8b365", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], DCR, "SL", lwd=3,col="#f6e8c3", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], DCR, "IF", lwd=3,col="#c7eae5", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], DCR, "EF", lwd=3,col="#5ab4ac", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], DCR, "RE", lwd=3,col="#01665e", add=T, rug=F)

partialPlot(sub_rf, training_set[,-2], SLO, "ML", lwd=3,col="#8c510a", main=NULL, xlab="SLO", ylab="", rug=F, ylim=c(-20, 15), xlim=c(0, 2))
partialPlot(sub_rf, training_set[,-2], SLO, "DZ", lwd=3,col="#d8b365", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], SLO, "SL", lwd=3,col="#f6e8c3", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], SLO, "IF", lwd=3,col="#c7eae5", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], SLO, "EF", lwd=3,col="#5ab4ac", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], SLO, "RE", lwd=3,col="#01665e", add=T, rug=F)

partialPlot(sub_rf, training_set[,-2], NSC, "ML", lwd=3,col="#8c510a", main=NULL, xlab="NSC", ylab="f(N) partial dependency", rug=F, ylim=c(-15, 12), xlim=c(18, 100))
partialPlot(sub_rf, training_set[,-2], NSC, "DZ", lwd=3,col="#d8b365", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], NSC, "SL", lwd=3,col="#f6e8c3", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], NSC, "IF", lwd=3,col="#c7eae5", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], NSC, "EF", lwd=3,col="#5ab4ac", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], NSC, "RE", lwd=3,col="#01665e", add=T, rug=F)

partialPlot(sub_rf, training_set[,-2], SVC, "ML", lwd=3,col="#8c510a", main=NULL, xlab="NSC", ylab="", rug=F, ylim=c(-15, 15), xlim=c(0, 22))
partialPlot(sub_rf, training_set[,-2], SVC, "DZ", lwd=3,col="#d8b365", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], SVC, "SL", lwd=3,col="#f6e8c3", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], SVC, "IF", lwd=3,col="#c7eae5", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], SVC, "EF", lwd=3,col="#5ab4ac", add=T, rug=F)
partialPlot(sub_rf, training_set[,-2], SVC, "RE", lwd=3,col="#01665e", add=T, rug=F)


##########################################################################################################
## Metric multidimensional scaling diagram based on the dissimilarity of proximity matrix 	      			##
##########################################################################################################

#accessing
mds<-cmdscale(1 - sub_rf$proximity, k=5, eig=TRUE)
data.frame.mds <-  as.data.frame(cmdscale(1 - sub_rf$proximity, k=5))

# variance explained by the constrained ordination
round((mds$eig/sum(mds$eig))[1:5], 4)*100

data.frame.mds$predi <- factor(sub_rf$predicted, levels = c("RE","EF", "IF", "SL", "DZ", "ML"))
data.frame.mds$orig <- sub_rf$y
data.frame.mds$sec <- training_set$sec
data.frame.mds$Correct1 = data.frame.mds$predi == data.frame.mds$orig

#Figure 7
cols <- c("RE" = "#01665e", "EF" = "#5ab4ac", "IF" = "#c7eae5", 
          "SL" = "#f6e8c3", "DZ"="#d8b365", "ML"="#8c510a")

d1d2<-ggplot (data=data.frame.mds, aes(V1, V2)) + 
  geom_point(shape = 21, colour = "black", aes(fill = predi), size = 4, stroke = 1)+
  geom_segment(x = 0.15, y = -0.3, xend = 0.4, yend = -0.3,
               arrow = arrow(length = unit(0.03, "npc"),end="both"))+
  geom_segment(x = -0.515, y = 0.2, xend = -0.515, yend = 0.45,
               arrow = arrow(length = unit(0.03, "npc")))+
  scale_fill_manual(values = cols, guide = guide_legend(title = NULL, nrow=1, label.position = "bottom",label.hjust = 0.5,label.vjust = 1))+
  labs(x="Dimension 1 (32.4%)",y="Dimension 2 (16.5%)")+
  theme_bw()+
  theme(axis.text = element_text(size = 11, colour = "black"),
        panel.grid = element_blank(),
        legend.position = c(0.3, 0.08),
        legend.background = element_rect(size=0.1, linetype="solid",colour ="black"))+
  annotate("text", x = -0.515, y = 0.525, label = "ENT, DCR", angle = 90)+
  annotate("text", x = 0.1, y = -0.3, label = "NSC")+
  annotate("text", x = 0.485, y = -0.3, label = "ENT, SVC")

d1d5<-ggplot (data=data.frame.mds, aes(V1, V5)) + 
  geom_point(shape = 21, colour = "black", aes(fill = predi), size = 3, stroke = 1)+
  scale_fill_manual(values = cols, guide=FALSE)+
  labs(x="Dimension 1 (32.4%)",y="Dimension 5 (5.5%)")+
  theme_bw()+
  theme(axis.text = element_text(size = 11, colour = "black"),
        axis.text.x = element_text(hjust=0.85),
        panel.grid = element_blank(),
        plot.background = element_rect(fill = NA, colour = NA))+
  geom_segment(x = -0.5, y = -0.15, xend =-0.5, yend = -0.4,
               arrow = arrow(length = unit(0.03, "npc")))+
  annotate("text", x = -0.5, y = -0.45, label = "SLO", angle=90)

ggdraw() +
  draw_plot(d1d2, 0, 0, 1, 1) +
  draw_plot(d1d5, 0.6, 0.5, 0.4, 0.5) +
  draw_plot_label(c("A", "B"), c(0.1, 0.9), c(0.5, 0.7), size = 20)
