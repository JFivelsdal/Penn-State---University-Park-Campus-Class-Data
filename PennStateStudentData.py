__author__ = 'Jonathan'


import pandas as pd #pd alias for pandas


import numpy as np

import matplotlib.pyplot as plt

import scipy.stats as scipy

import prettyplotlib as ppl

import statsmodels.api as sm

from collections import Counter

#import PyLaTex as pytex

studentData = pd.read_csv('C:/Users/Jonathan/Desktop/ourdata.csv')

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
plt.title("Student Height (Raw Data) ", fontsize = 18)
plt.xlabel('Height',fontsize = 16)
plt.ylabel('Count',fontsize = 16)


plt.subplot(122)

plt.hist(x=cleanHeight['Height'],color = 'yellow')
plt.title("Student Height (Cleaned Data) ", fontsize = 18)
plt.xlabel('Height',fontsize = 16)
plt.ylabel('Count', fontsize = 16)

#print studentData['Hub food']



plt.figure(3)  # Figure 3 are two bar graphs that show how students rated the cafeteria food

#dataHubFood =

#print studentData['Hub food'] == 'excellent'

#hubFoodData = studentData.loc[ (studentData['Hub food'] == 'excellent') | (studentData['Hub food'] == 'good')
                     # | (studentData['Hub food'] == 'poor')
                       #| (studentData['Hub food'] == 'unknown') | (studentData['Hub food'] == '*') ,['Hub food']]


#Create a groupby object based on Hub Food

plt.subplot(121)  # The left subplot provides a bar graph of the raw version of the opinions on the quality of
                  # cafeteria food
byHubFood = studentData.groupby('Hub food')


foodOpinionCounts =  byHubFood['Hub food'].count()


numFoodChoices = len(studentData['Hub food'].unique())


xLocations = np.arange(numFoodChoices)


#print numFoodChoices

print foodOpinionCounts



#print hubFoodData.sort()
widthHubFood = 0.5

plt.bar(xLocations,foodOpinionCounts,widthHubFood,color = 'red')

hubFoodAxis = plt.subplot(121)

hubFoodAxis.set_title("Opinions on the Quality of HUB Food (Raw)", fontsize = 20)

hubFoodAxis.set_ylabel('Counts', fontsize = 16)

hubFoodAxis.set_xticks(xLocations + widthHubFood/2)

hubFoodAxis.set_xticklabels(['Missing','Excellent','Good','Poor','Unknown'])

plt.setp(hubFoodAxis.get_xticklabels(),fontsize = 16) #Set the font size of the x tick labels

#########################


hubFoodClean =  studentData.loc[ (studentData['Hub food'] != 'unknown') & (studentData['Hub food'] != '*'),['Hub food']]

byHubFoodClean = hubFoodClean.groupby('Hub food')

foodOpinionCountsClean = byHubFoodClean['Hub food'].count()

numFoodChoicesClean = len(hubFoodClean['Hub food'].unique())

xLocationsClean = np.arange(numFoodChoicesClean)

#print numFoodChoicesClean

foodSportFrame = studentData.groupby(['Gender','Hair dyed?'])

foodSportFrame = foodSportFrame['Gender','Hair dyed?'].count('Gender')

#print foodSportFrame


#Subset of Data that contains just the Gender Columns and Hair Dyed Columns
studentHairDye = studentData.loc[:,['Gender','Hair dyed?']]




#Gets rid of the Missing values in the Hair Dyed Subset (the subset object called studentHairDye) )

studentHairDyeNoMiss = studentHairDye[studentHairDye.Gender != '*']


print "Contingency Table for Hair Dyed and Gender Variables" + "\n"

hairDyePivot = pd.pivot_table(data = studentHairDyeNoMiss,rows = ['Gender'],columns = ['Hair dyed?'],aggfunc = len)

print hairDyePivot

print "\n"

#print scipy.chi2_contingency([hairDyePivot['no'],hairDyePivot['yes']], correction = False)


#Female counts for whether the student dyed their hair or not
femaleHairData = hairDyePivot[hairDyePivot.index == 'F']

#Male counts for whether the student dyed their hair or not
maleHairData = hairDyePivot[hairDyePivot.index == 'M']


#print maleHairData.values

#print hairDyePivot.index

#print hairDyePivot.loc[hairDyePivot.index == 'F',['Hair dyed?']]

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

# E[1,1] = 16.28

# E[1,2] = 8.72

# E[2,1] = 11.72

# E[2,2] = 6.28



#Contigency Table of Student Hair Dye Choice by Gender

#print "groupHairDye"

#groupHairDye = studentHairDye.groupby(['Gender','Hair dyed?'],as_index = False).size()

#print groupHairDye




#groupHairDyeCounts = Counter(groupHairDye).keys()

#print groupHairDyeCounts


# Chi-Square Test for Student Hair Dye Choice by Gender

#print scipy.chi2_contingency([[12,13],[15,3]], correction = False)

#Results of the Chi-Square Test

#Chi-Square Statistic = 5.59

#p-value = 0.018

#degree of freedom = 1

# Expected values are
# E[1,1] = 15.70
# E[1,2] = 9.30
# E[2,1] = 11.30
# E[2,2] = 6.70


#groupGenDye = studentHairDye.groupby('Gender').aggregate(np.sum)

#np.

#print groupGenDye
#hairPivot = pd.pivot_table(studentHairDye,index = ['Gender'],cols = ['Hair dyed?'],aggfunc= np.sum(), fill_value =0)

#print hairPivot

#print studentHairDye.count()

#print studentHairDye

#foodSportFrame = studentData.groupby(['Gender'])['Hair dyed?']

#foodSportFrame = foodSportFrame.count()

#foodSportFrame = pd.DataFrame(foodSportFrame,index=[1,2])

#print foodSportFrame['Gender']
#foodSportFrame = foodSportFrame.loc[foodSportFrame.Gender != '*',['Gender']]
# Number of people that think Hub food is Excellent

#Basketball 0; Football 0; Other 1

#Number of people that think Hub food is Good

#Basketball 8; Football 8; Other 18

#Number of people that think Hub food is Poor

#Basketball 2; Football 2; Other 3


#print foodSportFrame
#print studentData.pivot_table(['Hub food'],rows = ['Sport'],aggfunc = np.sum)

#foodSportFrame = studentData.loc[ (studentData['Hub food'] != 'unknown') & (studentData['Hub food'] != '*'),['Hub food','Sport']]

#print foodSportFrame

#foodBySport = foodSportFrame.groupby('Sport','Hub food')

#print foodBySport['Hub food'].count()


plt.subplot(122) # The right subplot provides a bar graph of the clean version
                 # (without the missing data and the unknown entry) of the opinions on the quality of
                 # cafeteria food


plt.bar(xLocationsClean,foodOpinionCountsClean,widthHubFood )

hubFoodAxis = plt.subplot(122)

hubFoodAxis.set_title("Opinions on the Quality of HUB Food (Clean)", fontsize = 20)

hubFoodAxis.set_ylabel('Counts', fontsize = 16)

hubFoodAxis.set_xticks(xLocationsClean + widthHubFood/2)

hubFoodAxis.set_xticklabels(['Excellent','Good','Poor'])

plt.setp(hubFoodAxis.get_xticklabels(),fontsize = 16) #Set the font size of the x tick labels



#print hubFoodData.groupby(hubFoodData.unique()).count()


                      # .groupby('Hub food').count()

#print dataHubFood
#print np.array(range(len(dataHubFood))) + 0.5

#xLocations = np.array(range(len(dataHubFood))) + 0.5

#plt.bar(xLocations,dataHubFood)



#print studentData['Hub food'].unique()

#xHubFoodBar = np.arange(numFoodResponses)

#plt.bar(xHubFoodBar,studentData['Hub food'])
#plt.subplot(121)

#plt.boxplot(studentData['Height'])

#plt.xticks([1],["Height"])

#plt.subplot(122)

#plt.axis([0,5,0,cleanHeight['Height'].max() * 1.2])

#plt.boxplot(cleanHeight['Height'])

#plt.xticks([1],["Height"])

#plt.figure(3)



#plt.barh(bottom = cleanHeight['Height'], color = 'yellow')

plt.figure(4)

labels = ['Height (Clean)','Height (Raw)']

dataHeight = pd.DataFrame({'Height (Clean)': cleanHeight['Height'],'Height (Raw)': studentData['Height']})

dataHeightArray = np.array(dataHeight)

print dataHeightArray

ppl.boxplot(dataHeightArray, xticklabels= labels,fontsize= 16)

newAx = np.arange(0,80,5)

plt.yticks(newAx)

plt.title(s = "Student Heights",fontsize = 20)


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

ttestresultsind(gradeAsianMale,gradeCaucasianMale,'larger','pooled')


##################################################################################

genStudyRaceF = studentData.loc[((studentData['Race'] == 'Asian') | (studentData['Race'] == 'Caucasian'))
                               & (studentData['Gender'] == 'F') &
                               (studentData['GPA'] < 4) | (studentData['GPA'] == 4),['Race','Gender','GPA',]]


gradeAsianFemale = genStudyRaceF.loc[genStudyRaceF['Race'] == 'Asian',['GPA']]

gradeCaucasianFemale = genStudyRaceF.loc[genStudyRaceF['Race'] == 'Caucasian',['GPA']]

print gradeAsianFemale

print gradeCaucasianFemale

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

ttestresultsind(heightCaucasianFemale,heightAsianFemale,'larger','unequal') #p-value = 0.013 for unequal variances

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

print weightAsianF

ttestresultsind(actualWAsianF,idealWAsianF,'two-sided','pooled') #p-value = 0.115 for pooled variances


print "Variance for Weight of Asian Females " + str(scipy.describe(actualWAsianF)[3].round(decimals = 4))

print "Variance for Ideal Weight of Asian Females " + str(scipy.describe(idealWAsianF )[3].round(decimals = 4))


print studentData

#print heightAsianF
#print gradeAsianMale
#scipy.ttest_ind()

#gradeRaceTStat = sm.stats.ttest_ind(x1=gradeAsianMale,x2=gradeCaucasianMale, alternative="larger",usevar="pooled")[0]

#gradeRacePVal = sm.stats.ttest_ind(x1=gradeAsianMale,x2=gradeCaucasianMale, alternative="larger",usevar="pooled" )[1]

#gradeDegreesFree = sm.stats.ttest_ind(x1=gradeAsianMale,x2=gradeCaucasianMale, alternative="larger",usevar="pooled" )[2]


#print "t-test result: " + "t-statistic: " + " " + str(round(gradeRaceTStat, 3)) + " " + "p-value: " + " " \
 #     + str(round(gradeRacePVal, 3)) + " " + "Degrees of Freedom: " + str(round(gradeDegreesFree, 3))



dir('stats')


plt.show()


#plt.figure(4)


#dotCH = ggplot(aes(x=cleanHeight['Height']), data=studentData) + geom_dotplot(binwidth = 1)

#ggplot.draw(dotCH)

#Produces LaTex code for a table with mean, standard deviation, and variance

#def latexTable(rowName,colName,summaryStatistics):

# Build table, then print
    #latex = []
    #latex.append('\\begin{tabular}{r|rrrr}')

    #line = ''


    #for i in xrange(len(colName)):
     #   line += '&' + rowName[i]

    #line += ' \\ \hline'

   # latex.append(line)

    #for i in xrange(np.size(summaryStatistics,0)):

     #   line = rowName[i]
      #  for j in xrange(np.size(summaryStatistics,1)):
       #     line += '&' + str(summaryStatistics[i,j])

        #latex.append(line)

    #latex.append('\\end{tabular}')

# Output using write()
    #f = file('latex_table.tex','w')

    #for line in latex:

     #f.write(line + '\n')

    #f.close()

#latexTable(['Height','Height','Height'],['Mean','Variance','Skew'],np.vstack((meanCH,varianceCH,skewCH)))