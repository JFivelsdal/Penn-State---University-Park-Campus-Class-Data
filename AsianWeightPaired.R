
#############################

# AsianWeightPaired.R 

# Creates a bar plot for the mean weight and mean ideal weight for Asian female students
# offered at the University Park campus at Penn State
# Also a plot is created for the difference between the mean weight and mean ideal weight
# of Asian female students in the class and the 98.5% confidence interval that 
# estimates the difference between the corresponding population means.

library(RColorBrewer)
library(ggplot2)

ourdataDF <- read.csv("ourdata.csv")


# display.brewer.all() #shows the palettes available in color brewer

#Get the weight and ideal weight values for female Asian students from the data

genWeightandIdealW <- subset(ourdataDF,subset = Gender == "F" & Race == "Asian",select = c(Gender,Race,Weight,Ideal.weight))

asianFemaleW <- genWeightandIdealW[,"Weight"]

asianFemaleIW <- genWeightandIdealW[,"Ideal.weight"]

weightFrame <- data.frame(cbind(mean(asianFemaleW),mean(asianFemaleIW)))

colnames(weightFrame) <- c("actual","ideal")


#A t-test is performed to get the confidence interval for the difference in means
asianWeightResult <- t.test(asianFemaleW,asianFemaleIW,paired = TRUE,conf.level = 0.985)

weightLow <- asianWeightResult$conf.int[1] #Lower endpoint of the confidence interval 

weightHigh <- asianWeightResult$conf.int[2] #Upper endpoint of the confidence interval 





#The difference in mean weight and ideal weight in the sample

weightDiff <-  weightFrame[,"actual"] - weightFrame[,"ideal"]


#Function to remove both grid lines and tick marks on plots

tickGridRemove <- function() {theme(panel.grid.major = element_blank(), 
                                    panel.grid.minor = element_blank(),axis.ticks = element_blank())
                               
                             }


#Plot of the difference in means (Actual Weight - Ideal Weight for Asian Female Students)

ggplot(data = NULL,aes(x = c("Actual Weight - Ideal Weight"), 
       y = weightDiff)) +   
  
  geom_bar(width = .3, fill = brewer.pal(8,"Spectral")[7],stat = "identity") +
  
  
  scale_y_continuous(limits = c(0,12.5),breaks = seq(0,12.5,by = 1.5)) +
  
  scale_x_discrete(labels = "") +
  
  ggtitle("Difference Between the Mean Actual Weight and \n Mean Ideal Weight  of Asian Female Students") +
  
  theme(plot.title = element_text(face = "bold", size = 20)) +
  
  xlab("Actual Weight - Ideal Weight") +
  
  #Changes the appearance of the values on the y-axis
  
  theme(axis.text.y = element_text(size = 18,color = "black")) +
  
  theme(axis.title.x = element_text(face = "bold",size = 18)) +
  
  ylab("Difference in Means") +
  
  theme(axis.title.y = element_text(face = "bold",size = 18)) +

  tickGridRemove() +
  
  #Displays the confidence interval for the difference in means between the ideal weight and
  # actual weight groups
  
  geom_errorbar(width = .1,aes(ymin = weightLow, ymax = weightHigh)) 
  


#This is a plot of the sample means for both the mean actual weight and mean ideal weight

ggplot(weightFrame,aes(x = c("Actual Weight","Ideal Weight"),y = as.numeric(weightFrame[1,]))) + 
  
  geom_bar(width = .3, fill = c(brewer.pal(9,"RdBu")[9],brewer.pal(9,"PRGn")[9]) 
           ,colour = "black",stat = "identity") +
  
 # scale_y_continuous(limits = c(0,120),breaks = seq(0,120,by = 10)) +

  xlab("") +
  
  theme(axis.text.x = element_text(face = "bold",size = 16,color = "black")) +
  
  ylab("Mean Values (lbs.)") +
  
  #Changes the appearance of the values on the y-axis
  theme(axis.text.y = element_text(size = 18,color = "black")) +
  
  theme(axis.title.y = element_text(face = "bold",size = 18)) +
  
  ggtitle("Comparing the Mean Actual Weight and \n Mean Ideal Weight  of Asian Female Students") +
  
  theme(plot.title = element_text(face = "bold", size = 20)) +

  
  scale_y_continuous(breaks = seq(0,120,by = 5)) +
  
  coord_cartesian(ylim = c(100,120)) +
  
  tickGridRemove()
  
