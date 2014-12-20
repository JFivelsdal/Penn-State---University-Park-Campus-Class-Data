
__author__ = 'Jonathan'


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import scipy.stats as scipy

import prettyplotlib as ppl

import statsmodels.api as sm

from scipy.stats import levene as lev


studentData = pd.read_csv('C:/Users/Jonathan/Desktop/ourdata.csv')  # Student survey data

noErrorGPA = studentData.loc[((studentData.GPA < 4) | (studentData.GPA == 4)) & (studentData.GPA.notnull()), ['GPA']]

plt.subplot(121)

xValGPA = np.arange(0,64,4)

plt.figure(1)  # Creates the figure for the raw and cleaned data for the GPA - figure 1

studentData['GPA'].hist(grid = False,color = 'red',bins = 15)

plt.title("Student GPA (Raw Data)",fontsize = 18)
plt.xlabel('GPA',fontsize = 16)
plt.ylabel('Count',fontsize = 16)

plt.xticks(xValGPA)


plt.subplot(122)

plt.hist(x=noErrorGPA['GPA'])

plt.title("Student GPA (Cleaned Data) ", fontsize = 18)
plt.xlabel('GPA',fontsize = 16)
plt.ylabel('Count', fontsize = 16)


########################################

cleanHeight = studentData.loc[(studentData.Height > 40) & (studentData.Height.notnull()), ['Height']]


plt.figure(2)  # Creates the figure for the raw and cleaned data for the height data - figure 2

plt.subplot(121)

studentData['Height'].hist(grid = False,color = 'purple',)
plt.title("Student Height (Raw Data) ", fontsize = 16)
plt.xlabel('Height (inches)',fontsize = 16)
plt.ylabel('Count',fontsize = 16)


plt.subplot(122)

plt.hist(x=cleanHeight['Height'],color = 'yellow')
plt.title("Student Height (Cleaned Data) ", fontsize = 18)
plt.xlabel('Height (inches)',fontsize = 16)
plt.ylabel('Count', fontsize = 16)




plt.figure(3)  # Figure 3 are two bar graphs that show how students rated the cafeteria food


#Create a groupby object based on Hub Food

plt.subplot(121)  # The left subplot provides a bar graph of the raw version of the opinions on the quality of
                  # cafeteria food
byHubFood = studentData.groupby('Hub food')


#foodOpinionCounts stores the number of students that ranked HUB food as being excellent, good, or poor along
# with missing values and the invalid entry of unknown

foodOpinionCounts =  byHubFood['Hub food'].count()


numFoodChoices = len(studentData['Hub food'].unique())


xLocations = np.arange(numFoodChoices)


print foodOpinionCounts


widthHubFood = 0.5

plt.bar(xLocations,foodOpinionCounts,widthHubFood,color = 'red')


#The code in this section creates a bar graph for opinions on HUB food using the raw data

####### Create HUB food opinion graph based on raw data ######

hubFoodAxis = plt.subplot(121)

hubFoodAxis.set_title("Opinions on the Quality of HUB Food (Raw)", fontsize = 20)

hubFoodAxis.set_ylabel('Counts', fontsize = 16)

hubFoodAxis.set_xticks(xLocations + widthHubFood/2)

hubFoodAxis.set_xticklabels(['Missing','Excellent','Good','Poor','Unknown'])

plt.xlabel('Student Ratings',fontsize = 16)

plt.setp(hubFoodAxis.get_xticklabels(),fontsize = 16) #Set the font size of the x tick labels

####### End of code section that creates HUB food opinion graph based on RAW data ######


###### Beginning of Hub food Clean data Section ######

hubFoodClean =  studentData.loc[ (studentData['Hub food'] != 'unknown') & (studentData['Hub food'] != '*'),['Hub food']]

byHubFoodClean = hubFoodClean.groupby('Hub food')


#foodOpinionCountsClean stores the number of students that ranked HUB food as being Excellent, Good, or Poor

foodOpinionCountsClean = byHubFoodClean['Hub food'].count()

numFoodChoicesClean = len(hubFoodClean['Hub food'].unique())

xLocationsClean = np.arange(numFoodChoicesClean)



####### Create HUB food opinion graph based on CLEAN data ######


plt.subplot(122) # The right subplot provides a bar graph of the clean version
                 # (without the missing data and the unknown entry) of the opinions on the quality of
                 # cafeteria food


plt.bar(xLocationsClean,foodOpinionCountsClean,widthHubFood )

hubFoodAxis = plt.subplot(122)

hubFoodAxis.set_title("Opinions on the Quality of HUB Food (Clean)", fontsize = 20)

hubFoodAxis.set_ylabel('Counts', fontsize = 16)

hubFoodAxis.set_xticks(xLocationsClean + widthHubFood/2)

hubFoodAxis.set_xticklabels(['Excellent','Good','Poor'])

plt.xlabel('Student Ratings',fontsize = 16)

plt.setp(hubFoodAxis.get_xticklabels(),fontsize = 16) #Set the font size of the x tick labels

####### End of code section that creates HUB food opinion graph based on CLEAN data ######


# Creates a Box plot of the Clean Height Data and the Raw Height Data
plt.figure(4)

labels = ['Height (Clean)','Height (Raw)']

dataHeight = pd.DataFrame({'Height (Clean)': cleanHeight['Height'],'Height (Raw)': studentData['Height']})

dataHeightArray = np.array(dataHeight)

#print dataHeightArray

ppl.boxplot(dataHeightArray, xticklabels= labels,fontsize= 16)



newAx = np.arange(0,80,5)

plt.yticks(newAx)

plt.ylabel('Height (inches)',fontsize = 16)
plt.title(s = "Heights of Students",fontsize = 20)



plt.show()

#Subset of Data that contains just the Gender Columns and Hair Dyed Columns
studentHairDye = studentData.loc[:,['Gender','Hair dyed?']]




#Gets rid of the Missing values in the Hair Dyed Subset (the subset object called studentHairDye) )

studentHairDyeNoMiss = studentHairDye[studentHairDye.Gender != '*']


print "Contingency Table for Hair Dyed and Gender Variables" + "\n"

hairDyePivot = pd.pivot_table(data = studentHairDyeNoMiss,rows = ['Gender'],columns = ['Hair dyed?'],aggfunc = len)

print hairDyePivot

print "\n"




#Female counts for whether the student dyed their hair or not
femaleHairData = hairDyePivot[hairDyePivot.index == 'F']

#Male counts for whether the student dyed their hair or not
maleHairData = hairDyePivot[hairDyePivot.index == 'M']




print "Results of Chi-Square Test on the Relationship Between a Student Dyeing Their Hair and Gender: " + "\n"


chiStatHairGen = scipy.chi2_contingency([femaleHairData.values,maleHairData.values], correction = False)[0]

pValHairGen = scipy.chi2_contingency([femaleHairData.values,maleHairData.values], correction = False)[1]

degreeFreeHairGen = scipy.chi2_contingency([femaleHairData.values,maleHairData.values], correction = False)[2]



print "Chi-Square statisitc: " + str(chiStatHairGen)

print "P-value: " + str(pValHairGen)

print "Degrees of Freedom: " + str(degreeFreeHairGen)


#Chi-Square statistic: 4.52

#p-value: 0.033

#Degrees of Freedom: 1

# E[1,1] = 16.28  - 16.28 expected number of females to chose not to dye their hair

# E[1,2] = 8.72   - 8.72 expected number of females to chose to dye their hair

# E[2,1] = 11.72  - 11.72 expected number of males to chose not to dye their hair

# E[2,2] = 6.28   - 6.28 expected number of males to chose to dye their hair


################

#### Begin Chi-Square Graph With One Degree Of Freedom Code

#xChi = np.linspace(0,10,200)

xChi = np.linspace(0,10,200)

yChi = scipy.chi2.pdf(xChi,1)



print 1 - scipy.chi2.cdf(4.52,1)

pValChi = 1 - scipy.chi2.cdf(4.52,1)


testStatChi = scipy.chi2.ppf(1-pValChi,1)

print scipy.chi2.ppf(0.95,1)

print len(yChi)

print len(xChi)

criticalChi = scipy.chi2.ppf(0.95,1)



plt.figure(5)

plt.plot(xChi,yChi,color = 'black') # Plots the Curve for the Chi-Squared Distribution with one degree of freedom


plt.title('Chi-Squared Distribution With One Degree of Freedom',size = 17)

# Critical value at the 95% Confidence level (0.05 alpha level)

confLevel95 = scipy.chi2.ppf(0.95,1)


# Critical value at the 99% Confidence level (0.05 alpha level)

confLevel99 = scipy.chi2.ppf(0.99,1)


#Creates a vertical line where the Chi-Squared test statistic is located
plt.vlines(testStatChi,ymin = 0, ymax = 1.8,color = 'blue',linestyles = 'dashed',label = "observed statistic")

#Creates a vertical line where the 95% critical value occurs
plt.vlines(confLevel95,ymin = 0, ymax = 1.8,color = 'red',label = r'$\alpha = 0.05$' + ' level')

#Creates a vertical line where the 99% critical value occurs
plt.vlines(confLevel99,ymin = 0, ymax = 1.8,color = 'purple',label =  r'$\alpha = 0.01$' + ' level')

plt.fill_betweenx(yChi,xChi,criticalChi, where= xChi > criticalChi,color = 'orange')

plt.legend()


plt.show()

#Summary Statistics for Raw Height Data

#The statistics table has the following measures:

# 1. Mean

# 2. Variance

# 3. Skewness

# 4. Excess Kurtosis

# 5. Kurtosis

meanRaw = scipy.describe(studentData['Height'])[2].round(decimals = 2)

varianceRaw = scipy.describe(studentData['Height'])[3].round(decimals = 2)

skewRaw = round(scipy.describe(studentData['Height'])[4],2)

exKurtRaw = round(scipy.describe(studentData['Height'])[5],2)

kurtRaw = round(scipy.kurtosis(studentData['Height'],fisher = False), 2)



print("\n")
print("Raw Height Data Summary Statistics")
print("mean: " + str(meanRaw) + " " + "variance: " + str(varianceRaw) + " " + "skew: " + str(skewRaw)
      + " " + "excess kurtosis: " + str(exKurtRaw) + " " + "kurtosis: " + str(kurtRaw))


#Summary Statistics for Clean Height Data

meanCH = scipy.describe(cleanHeight['Height'])[2].round(decimals = 2)

varianceCH = scipy.describe(cleanHeight['Height'])[3].round(decimals = 2)

skewCH = round(scipy.describe(cleanHeight['Height'])[4], 2)

exKurtCH = round(scipy.describe(cleanHeight['Height'])[5], 2)

kurtCH = round(scipy.kurtosis(cleanHeight['Height'],fisher = False), 2)


#Excess Kurtosis = Kurtosis - 3

#The normal distribution has a kurtosis of 3, since excess kurtosis is kurtosis minus 3, the excess kurtosis of
# the normal distribution is zero.


print("\n")
print("Clean Height Data Summary Statistics")
print("mean: " + str(meanCH) + " " + "variance: " + str(varianceCH) + " " + "skew: " + str(skewCH)
      + " " + "excess kurtosis: " + str(exKurtCH) + " " + "kurtosis: " + str(kurtCH))



#ttestresults is a wrapper function that presents the results of a t-test by means of the
# ttest_ind in the statsmodels package

def ttestresultsind(data1,data2,alt,var):

   gradeRaceTStat = sm.stats.ttest_ind(x1=data1,x2=data2, alternative= alt,usevar= var)[0]

   gradeRacePVal = sm.stats.ttest_ind(x1=data1,x2=data2, alternative= alt,usevar=var )[1]

   gradeDegreesFree = sm.stats.ttest_ind(x1=data1,x2=data2, alternative=alt,usevar=var )[2]



   print "t-test result: " + "t-statistic: " + " " + str(round(gradeRaceTStat, 3)) + " " + "p-value: " + " " \
   + str(round(gradeRacePVal, 3)) + " " + "Degrees of Freedom: " + str(round(gradeDegreesFree, 3))






#genStudyRace is a subset of the data that contains only grades for male Asian students and male Caucasian students
genStudyRaceM = studentData.loc[((studentData['Race'] == 'Asian') | (studentData['Race'] == 'Caucasian'))
                               & (studentData['Gender'] == 'M') &
                               (studentData['GPA'] < 4) | (studentData['GPA'] == 4),['Race','Gender','GPA',]]


gradeAsianMale = genStudyRaceM.loc[genStudyRaceM['Race'] == 'Asian',['GPA']]

gradeCaucasianMale = genStudyRaceM.loc[genStudyRaceM['Race'] == 'Caucasian',['GPA']]

print gradeAsianMale

print gradeCaucasianMale

print  "\n" + "T-test Results between the grades of Asian Males and Caucasian Males" + "\n"

ttestresultsind(gradeAsianMale,gradeCaucasianMale,'larger','pooled')


##################################################################################

genStudyRaceF = studentData.loc[((studentData['Race'] == 'Asian') | (studentData['Race'] == 'Caucasian'))
                               & (studentData['Gender'] == 'F') &
                               (studentData['GPA'] < 4) | (studentData['GPA'] == 4),['Race','Gender','GPA',]]


gradeAsianFemale = genStudyRaceF.loc[genStudyRaceF['Race'] == 'Asian',['GPA']]

gradeCaucasianFemale = genStudyRaceF.loc[genStudyRaceF['Race'] == 'Caucasian',['GPA']]

print gradeAsianFemale

print gradeCaucasianFemale

print "T-test Results between the grades of Asian Females and Caucasian Females" + "\n"

ttestresultsind(gradeCaucasianFemale,gradeAsianFemale,'larger','pooled')


##############################################################################


#Use this for a 2-sample t-test

# Ho: Mean height for caucasian females is equal to the mean height for asian females

#Ha: Mean height for caucasian females is greater than the mean height for asian females

genHeightF = studentData.loc[((studentData['Race'] == 'Asian') | (studentData['Race'] == 'Caucasian'))
                               & (studentData['Gender'] == 'F') &
                               (studentData['Height'] > 40),['Race','Gender','Height',]]


heightAsianFemale = genHeightF.loc[genHeightF['Race'] == 'Asian',['Height']]

heightCaucasianFemale = genHeightF.loc[genHeightF['Race'] == 'Caucasian',['Height']]


print "Variance for Height of Asian Females" + str(scipy.describe(heightAsianFemale)[3].round(decimals = 4))

print "Variance for Height of Caucasian Females" + str(scipy.describe(heightCaucasianFemale)[3].round(decimals = 4))

print heightAsianFemale

print heightCaucasianFemale


print lev(heightAsianFemale,heightCaucasianFemale)



#Results of the Levene test

# test statistic = 2.05 and p-value = 0.168



# Calculated F-test in R using the following command

#var.test(c(63,62.5,62,65,64,63,60,63.5,64,63,61,68.5),c(63,70,62,73,65,67,68,65,66)
# ,alternative = "two.sided",conf.level = 0.95)

#Results of F-test

#F = 0.3838, num df = 11, denom df = 8, p-value = 0.1434



print "\n" + "T-test Results between the heights of Asian Females and Caucasian Females" + "\n"

ttestresultsind(heightCaucasianFemale,heightAsianFemale,'larger','unequal') #p-value = 0.013 for unequal variances

ttestresultsind(heightCaucasianFemale,heightAsianFemale,'larger','pooled')


#####################################################################

heightAsianF = studentData.loc[(studentData['Race'] == 'Asian')
                              & (studentData['Gender'] == 'F') &
                              (studentData['Height'] > 40) & (studentData['Ideal-Hight'] > 40) ,['Race','Height','Ideal-Hight']]


idealHAsianF = heightAsianF['Ideal-Hight']

print idealHAsianF


print "Variance for Height of Asian Females " + str(scipy.describe(heightAsianFemale)[3].round(decimals = 4))

print "Variance for Ideal Height of Asian Females " + str(scipy.describe(idealHAsianF)[3].round(decimals = 4))

######################################################################################

#print studentData

weightAsianF = studentData.loc[(studentData['Race'] == 'Asian')
                              & (studentData['Gender'] == 'F') &
                              (studentData['Weight']) & (studentData['Ideal-weight']) ,['Race','Weight','Ideal-weight']]



actualWAsianF = weightAsianF['Weight']

idealWAsianF = weightAsianF['Ideal-weight']

print lev(actualWAsianF,idealWAsianF)

#The Levene test statistic is 0.0469 and the p-value is 0.8304



print weightAsianF

print "\n" + "T-test Results between the actual weight of Asian Female Students and the ideal weight of Asian Female " \
             "students" + "\n"

#ttestresultsind(actualWAsianF,idealWAsianF,'two-sided','pooled') #p-value = 0.115 for pooled variances

sm.stats.ttest_rel(actualWAsianF,idealWAsianF)
print "Variance for Weight of Asian Females " + str(scipy.describe(actualWAsianF)[3].round(decimals = 4))

print "Variance for Ideal Weight of Asian Females " + str(scipy.describe(idealWAsianF )[3].round(decimals = 4))






