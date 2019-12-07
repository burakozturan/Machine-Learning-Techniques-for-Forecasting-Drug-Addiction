### Machine Learning Project
### Roberto Daniele Cadili 01/906991
### Burak Özturan 01/944663

import os
import pandas as pd
import numpy as np
import math
from statistics import variance


os.chdir('C:/Users/picch/Desktop/UNI KONSTANZ/Machine Learning/ML project')
dataset = pd.read_csv("drug_consumption.data", header = None)
print(dataset)

column_names = ['Id' , 'Age' , 'Gender' , 'Education' , 'Country' , 'Ethnicity' , 'Nscore' , 
                'Escore' , 'Oscore' , 'Ascore' , 'Cscore' , 'Impulsive', 'SS' , 'Alcohol' , 
                'Amphet' , 'Amyl' , 'Benzos' , 'Caff' , 'Cannabis' , 'Choc' , 'Coke' , 
                'Crack' , 'Ecstasy' , 'Heroin' , 'Ketamine' , 'Legalh' , 'LSD' , 'Meth' , 'Mushroom' , 
                'Nicotine' , 'Semer' , 'VSA']

dataset.columns = column_names

drugs_names = ['Amphet' , 'Amyl' , 'Benzos', 'Coke' , 
                'Crack' , 'Ecstasy' , 'Heroin' , 'Ketamine' , 'Legalh' , 'LSD' , 'Meth' , 'Mushroom' , 
                'Semer' , 'VSA']

X_all = dataset.drop(columns=['Id', 'Amphet' , 'Amyl' , 'Benzos', 'Coke' , "Country",
                'Crack' , 'Ecstasy' , "Ethnicity", 'Heroin' , 'Ketamine' , 'Legalh' , 'LSD' , 'Meth' , 'Mushroom' , 
                'Semer' , 'VSA']) # 10 input features (age, gender, educ, etc) + 5 soft drugs(alcohol, caffeine, nicotine, chocolate, cannabis)

X_all = X_all.replace(['CL2' , 'CL3' , 'CL4' , 'CL5' ,'CL6'], 1) # include soft drugs consumber in X_features
X_all = X_all.replace(['CL0' , 'CL1'], 0) # include soft drugs non-consumber in X_features

Drugs_all =  dataset.drop(columns= ['Id' , 'Age' , 'Gender' , 'Education' , 'Country' , 'Ethnicity' , 'Nscore' , 
                'Escore' , 'Oscore' , 'Ascore' , 'Cscore' , 'Impulsive', 'SS', 'Alcohol', 'Caff' , 'Cannabis' , 'Choc', 'Nicotine']) # 18 drugs + 1 fake (Semer)

Drugs_all_binary = Drugs_all.isin(['CL2' , 'CL3' , 'CL4' , 'CL5' ,'CL6']).astype(int) # For all 14 drugs, drug users coded with 1, non-drug users coded with 0 (CL0 and CL1)


Mushroom_classes_freq = pd.DataFrame(Drugs_all_binary['Mushroom'].value_counts())
n0 = int(Mushroom_classes_freq.iloc[0,:])
n1 = int(Mushroom_classes_freq.iloc[1,:])
n0 +n1

n0 + n1 == dataset.shape[0]

Data_all = pd.concat([X_all , Drugs_all_binary ], axis=1)
Data_all
X_all


##### Descriptive statistics
# Measures of central tendency and dispersion
import matplotlib.pyplot as plt
import scipy.stats as stats

descriptive_stat = pd.DataFrame(X_all, columns = X_all.columns)
describe=descriptive_stat.describe()


#Age ## DOES NOT MAKE MUCH SENSE TO INCLUDE IT
age = X_all['Age']

age_np = np.asarray(X_all['Age'])
age_np = sorted(age_np)
fit = stats.norm.pdf(age_np, np.mean(age_np), np.std(age_np))
 
mean=age.mean()
median=age.median()
mode=age.mode()

print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode[0])

plt.figure(figsize=(10,5))
plt.plot(age_np,fit,'-',linewidth = 2,label="Norm. distrib. with same mean and var")
plt.hist(age,density=True, bins=100,color='gray', label="Actual distribution")
plt.axvline(mean,color='green',label='Mean')
plt.axvline(median,color='yellow',label='Median')
plt.axvline(mode[0],color='red',label='Mode')
plt.xlabel('Age')
plt.ylabel('Density of probability')
plt.legend()
plt.show()

#Gender ## DOES NOT MAKE MUCH SENSE TO INCLUDE IT
gender = X_all['Gender']

gender_np = np.asarray(X_all['Gender'])
gender_np = sorted(gender_np)
fit = stats.norm.pdf(gender_np, np.mean(gender_np), np.std(gender_np))
 
mean=gender.mean()
median=gender.median()
mode=gender.mode()

print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode[0])

plt.figure(figsize=(10,5))
plt.plot(gender_np,fit,'-',linewidth = 2,label="Norm. distrib. with same mean and var")
plt.hist(gender,density=True, bins=100,color='gray', label="Actual distribution")
plt.axvline(mean,color='green',label='Mean')
plt.axvline(median,color='yellow',label='Median')
plt.axvline(mode[0],color='red',label='Mode')
plt.xlabel('Gender')
plt.ylabel('Density of probability')
plt.legend()
plt.show()

#Education ## DOES NOT MAKE MUCH SENSE TO INCLUDE IT
educ = X_all['Education']

educ_np = np.asarray(X_all['Education'])
educ_np = sorted(educ_np)
fit = stats.norm.pdf(educ_np, np.mean(educ_np), np.std(educ_np))
 
mean=educ.mean()
median=educ.median()
mode=educ.mode()

print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode[0])

plt.figure(figsize=(10,5))
plt.plot(educ_np,fit,'-',linewidth = 2,label="Norm. distrib. with same mean and var")
plt.hist(educ,density=True, bins=100,color='gray', label="Actual distribution")
plt.axvline(mean,color='green',label='Mean')
plt.axvline(median,color='yellow',label='Median')
plt.axvline(mode[0],color='red',label='Mode')
plt.xlabel('Education')
plt.ylabel('Density of probability')
plt.legend()
plt.show()

# Nscore
nscore = X_all['Nscore']

nscore_np = np.asarray(X_all['Nscore'])
nscore_np = sorted(nscore_np)
fit = stats.norm.pdf(nscore_np, np.mean(nscore_np), np.std(nscore_np))
 
mean=nscore.mean()
median=nscore.median()
mode=nscore.mode()

print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode[0])

plt.figure(figsize=(10,5))
plt.plot(nscore_np,fit,'-',linewidth = 2,label="Norm. distrib. with same mean and var")
plt.hist(nscore,density=True, bins=100,color='gray', label="Actual distribution")
plt.axvline(mean,color='green',label='Mean')
plt.axvline(median,color='yellow',label='Median')
plt.axvline(mode[0],color='red',label='Mode')
plt.xlabel('Nscore')
plt.ylabel('Density of probability')
plt.legend()
plt.savefig('nscore.jpeg')
plt.show()

# Escore
escore = X_all['Escore']

escore_np = np.asarray(X_all['Escore'])
escore_np = sorted(escore_np)
fit = stats.norm.pdf(escore_np, np.mean(escore_np), np.std(escore_np))
 
mean=escore.mean()
median=escore.median()
mode=escore.mode()

print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode[0])

plt.figure(figsize=(10,5))
plt.plot(escore_np,fit,'-',linewidth = 2,label="Norm. distrib. with same mean and var")
plt.hist(escore,density=True, bins=100,color='gray', label="Actual distribution")
plt.axvline(mean,color='green',label='Mean')
plt.axvline(median,color='yellow',label='Median')
plt.axvline(mode[0],color='red',label='Mode')
plt.xlabel('Escore')
plt.ylabel('Density of probability')
plt.legend()
plt.savefig('escore.jpeg')
plt.show()

# Oscore
oscore = X_all['Oscore']

oscore_np = np.asarray(X_all['Oscore'])
oscore_np = sorted(oscore_np)
fit = stats.norm.pdf(oscore_np, np.mean(oscore_np), np.std(oscore_np))
 
mean=oscore.mean()
median=oscore.median()
mode=oscore.mode()

print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode[0])

plt.figure(figsize=(10,5))
plt.plot(oscore_np,fit,'-',linewidth = 2,label="Norm. distrib. with same mean and var")
plt.hist(oscore,density=True, bins=100,color='gray', label="Actual distribution")
plt.axvline(mean,color='green',label='Mean')
plt.axvline(median,color='yellow',label='Median')
plt.axvline(mode[0],color='red',label='Mode')
plt.xlabel('Oscore')
plt.ylabel('Density of probability')
plt.legend()
plt.savefig('oscore.jpeg')
plt.show()

# Ascore
ascore = X_all['Ascore']

ascore_np = np.asarray(X_all['Ascore'])
ascore_np = sorted(ascore_np)
fit = stats.norm.pdf(ascore_np, np.mean(ascore_np), np.std(ascore_np))
 
mean=ascore.mean()
median=ascore.median()
mode=ascore.mode()

print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode[0])

plt.figure(figsize=(10,5))
plt.plot(ascore_np,fit,'-',linewidth = 2,label="Norm. distrib. with same mean and var")
plt.hist(ascore,density=True, bins=100,color='gray', label="Actual distribution")
plt.axvline(mean,color='green',label='Mean')
plt.axvline(median,color='yellow',label='Median')
plt.axvline(mode[0],color='red',label='Mode')
plt.xlabel('Ascore')
plt.ylabel('Density of probability')
plt.legend()
plt.savefig('ascore.jpeg')
plt.show()

# CScore
cscore = X_all['Cscore']

cscore_np = np.asarray(X_all['Cscore'])
cscore_np = sorted(cscore_np)
fit = stats.norm.pdf(cscore_np, np.mean(cscore_np), np.std(cscore_np))
 
mean=cscore.mean()
median=cscore.median()
mode=cscore.mode()

print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode[0])

plt.figure(figsize=(10,5))
plt.plot(cscore_np,fit,'-',linewidth = 2,label="Norm. distrib. with same mean and var")
plt.hist(cscore,density=True, bins=100,color='gray', label="Actual distribution")
plt.axvline(mean,color='green',label='Mean')
plt.axvline(median,color='yellow',label='Median')
plt.axvline(mode[0],color='red',label='Mode')
plt.xlabel('Cscore')
plt.ylabel('Density of probability')
plt.legend()
plt.savefig('cscore.jpeg')
plt.show()

# Impulsive ## DOES NOT MAKE MUCH SENSE TO INCLUDE IT
impulsive = X_all['Impulsive']

impulsive_np = np.asarray(X_all['Impulsive'])
impulsive_np = sorted(impulsive_np)
fit = stats.norm.pdf(impulsive_np, np.mean(impulsive_np), np.std(impulsive_np))
 
mean=impulsive.mean()
median=impulsive.median()
mode=impulsive.mode()

print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode[0])

plt.figure(figsize=(10,5))
plt.plot(impulsive_np,fit,'-',linewidth = 2,label="Norm. distrib. with same mean and var")
plt.hist(impulsive,density=True, bins=100,color='gray', label="Actual distribution")
plt.axvline(mean,color='green',label='Mean')
plt.axvline(median,color='yellow',label='Median')
plt.axvline(mode[0],color='red',label='Mode')
plt.xlabel('Impulsive')
plt.ylabel('Density of probability')
plt.legend()
plt.savefig('impulsive.jpeg')
plt.show()

# SS ## DOES NOT MAKE MUCH SENSE TO INCLUDE IT
ss = X_all['SS']

ss_np = np.asarray(X_all['SS'])
ss_np = sorted(ss_np)
fit = stats.norm.pdf(ss_np, np.mean(ss_np), np.std(ss_np))
 
mean=ss.mean()
median=ss.median()
mode=ss.mode()

print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode[0])

plt.figure(figsize=(10,5))
plt.plot(ss_np,fit,'-',linewidth = 2,label="Norm. distrib. with same mean and var")
plt.hist(ss,density=True, bins=100,color='gray', label="Actual distribution")
plt.axvline(mean,color='green',label='Mean')
plt.axvline(median,color='yellow',label='Median')
plt.axvline(mode[0],color='red',label='Mode')
plt.xlabel('SS')
plt.ylabel('Density of probability')
plt.legend()
plt.savefig('ss.jpeg')
plt.show()

# Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=[0, 1])
X_all_rescaled = scaler.fit_transform(X_all)

#Correlation of drugs and personality factors
'''
The correlation coefficient ranges from −1 to 1. 
A value of 1 implies that a linear equation describes the relationship between X and Y perfectly, 
with all data points lying on a line for which Y increases as X increases. 
A value of −1 implies that all data points lie on a line for which Y decreases as X increases. 
A value of 0 implies that there is no linear correlation between the variables.
'''
'''
correlation_df = pd.DataFrame(X_all, columns = X_all.columns)
correlation_df = correlation_df.corr(method='pearson')

correlation_5_person_fact = correlation_df.loc['Nscore':'Cscore', 'Nscore':'Cscore']

correlation_drugs = pd.DataFrame(Drugs_all_binary, columns = Drugs_all_binary.columns)
correlation_drugs = correlation_drugs.corr(method='pearson')

#Grouping to 3 drug categories heroin, ecstasy, benzo.
Heroin_pleiad_corr = correlation_drugs.loc[['Crack','Coke', 'Meth' , 'Heroin'] , ['Crack','Coke', 'Meth' , 'Heroin']]
Ecstasy_pleiad_corr =  correlation_drugs.loc[['Amphet', 'Coke' , 'Ketamine' , 'LSD' , 'Legalh' , 'Mushroom' , 'Ecstasy'] , ['Amphet', 'Coke' , 'Ketamine' , 'LSD' , 'Legalh' , 'Mushroom' , 'Ecstasy']]
Benzo_pleiad_corr = correlation_drugs.loc[['Meth','Amphet', 'Coke'],['Meth','Amphet', 'Coke']]


Heroin_pleiad_df = Drugs_all_binary[['Crack','Coke', 'Meth' , 'Heroin']]
Ecstasy_pleiad_df = Drugs_all_binary[['Amphet', 'Coke' , 'Ketamine' , 'LSD' , 'Legalh' , 'Mushroom' , 'Ecstasy']]
Benzo_pleiad_df = Drugs_all_binary[['Meth','Amphet', 'Coke']]


Heroin_pleiad_sum = np.sum(Heroin_pleiad_df,axis=1)
Ecstasy_pleiad_sum = np.sum(Ecstasy_pleiad_df , axis=1)
Benzo_pleiad_sum = np.sum(Benzo_pleiad_df , axis=1)


Heroin_pleiad_binary = []
for i in  range(Heroin_pleiad_df.shape[0]):
    if Heroin_pleiad_sum[i] == 0:
        Heroin_pleiad_binary.append(0)
    if Heroin_pleiad_sum[i] != 0:
        Heroin_pleiad_binary.append(1)

Heroin_pleiad_binary_np = np.array(Heroin_pleiad_binary)
Heroin_pleiad_binary_np = Heroin_pleiad_binary_np.reshape(1885,1)

Ecstasy_pleiad_binary = []
for i in  range(Ecstasy_pleiad_df.shape[0]):
    if Ecstasy_pleiad_sum[i] == 0:
        Ecstasy_pleiad_binary.append(0)
    if Ecstasy_pleiad_sum[i] != 0:
        Ecstasy_pleiad_binary.append(1)

Ecstasy_pleiad_binary_np = np.array(Ecstasy_pleiad_binary)
Ecstasy_pleiad_binary_np = Ecstasy_pleiad_binary_np.reshape(1885,1)

Benzo_pleiad_binary = []
for i in  range(Benzo_pleiad_df.shape[0]):
    if Benzo_pleiad_sum[i] == 0:
        Benzo_pleiad_binary.append(0)
    if Benzo_pleiad_sum[i] != 0:
        Benzo_pleiad_binary.append(1)

Benzo_pleiad_binary_np = np.array(Benzo_pleiad_binary)
Benzo_pleiad_binary_np = Benzo_pleiad_binary_np.reshape(1885,1)


All_pleiad_np = np.concatenate((Heroin_pleiad_binary_np,  Ecstasy_pleiad_binary_np, Benzo_pleiad_binary_np), axis=1)
'''

# PCA
'''
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Fitting the PCA algorithm with our Data
pca_all = PCA(n_components=10).fit(X_all_rescaled)

#Plotting the Cumulative Summation of the Explained Variance
plt.figure(figsize=(8, 8))
plt.plot(np.cumsum(pca_all.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Drug Consumption Dataset Explained Variance')
plt.show()

pca = PCA(n_components=8)
pca.fit_transform(X_all_rescaled)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())  

dataset_reduced = pca.fit_transform(X_all_rescaled) # USE THIS DATASET FOR CLASSIFICATION
df = pd.DataFrame(abs(pca.components_), columns=list(X_all.columns))

n_pcs= pca.components_.shape[0]

most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

initial_feature_names = X_all.columns

most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
dic
'''

# Sampling with pleaids
'''
from sklearn.model_selection import train_test_split

X_all_np = np.array(X_all_rescaled)

X_train, X_test, y_train, y_test = train_test_split(X_all_np, All_pleiad_np, test_size=0.1, random_state=41)

drugs_names = ['HeroinPl' , 'EcstasyPl' , 'BenzoPl']
'''

# Sampling

from sklearn.model_selection import train_test_split

X_all_np = np.array(X_all_rescaled)

Drugs_all_binary_np = np.array(Drugs_all_binary)

X_train, X_test, y_train, y_test = train_test_split(X_all_np, Drugs_all_binary_np, test_size=0.1, random_state=41)

################################
################################

#Linear classifier Sgd

from sklearn import linear_model

linear_classifier = linear_model.SGDClassifier(loss = 'squared_loss', alpha= 0.0001,max_iter=10000, penalty = None, tol=1e-3, random_state= 18)

linear_accuracies = []    
for i in range(14):
    fitted_linear = linear_classifier.fit(X_train, y_train[:,i])
    scores_linear = linear_classifier.score(X_test, y_test[:,i])
    linear_accuracies.append(scores_linear)
    print(fitted_linear.score(X_test, y_test[:,i]))    



linear_accuracies_array = np.array(linear_accuracies)
linear_accuracies_array = linear_accuracies_array.reshape(14,1)
linear_accuracies_array = linear_accuracies_array.T


df_LinearSGD =  pd.DataFrame(linear_accuracies_array, index= ['LinearSGD'] ,columns = drugs_names) 
 
df_all = df_LinearSGD

#Linear classifier Sgd regularized   
linear_classifier_regularized = linear_model.SGDClassifier(loss = 'squared_loss', alpha= 0.001, penalty = 'L2' ,max_iter=10000, tol=1e-3, random_state= 18)

linear_accuracies_reg = []
for i in range(14):
    fitted_regularized = linear_classifier_regularized.fit(X_train, y_train[:,i])
    scores_linear_reg = linear_classifier_regularized.score(X_test, y_test[:,i])
    linear_accuracies_reg.append(scores_linear_reg)
    print(fitted_regularized.score(X_test, y_test[:,i]))
    

linear_accuracies_reg_array = np.array(linear_accuracies_reg)
linear_accuracies_reg_array = linear_accuracies_reg_array.reshape(14,1)
linear_accuracies_reg_array = linear_accuracies_reg_array.T
  
df_LinearSGD_reg =  pd.DataFrame(linear_accuracies_reg_array, index= ['LinearSGD_reg'] ,columns = drugs_names) 
df_all = pd.concat((df_all,df_LinearSGD_reg) , axis=0)   

###Logistic regression 
log_classifier= linear_model.SGDClassifier(loss = 'log', alpha= 0.0001, penalty = None, max_iter=10000, tol=1e-3, random_state= 18)

log_accuracies = []
for k in range(14):
    fitted_log = log_classifier.fit(X_train, y_train[:,k])
    scores_log = log_classifier.score(X_test, y_test[:,k])
    log_accuracies.append(scores_log)
    print(scores_log)
 
log_accuracies_array = np.array(log_accuracies)
log_accuracies_array= log_accuracies_array.reshape(14,1)
log_accuracies_array= log_accuracies_array.T


df_log =  pd.DataFrame(log_accuracies_array, index= ['Logistic Regr'],columns = drugs_names) 

df_all = df_all.append(df_log)

### Logistic regression regularized
log_classifier_reg= linear_model.SGDClassifier(loss = 'log', alpha= 0.001, penalty = 'L2', max_iter=10000, tol=1e-3, random_state= 18)

log_accuracies_reg = []
for k in range(14):
    fitted_log_reg = log_classifier_reg.fit(X_train, y_train[:,k])
    scores_log_reg = log_classifier_reg.score(X_test, y_test[:,k])
    log_accuracies_reg.append(scores_log_reg)
    print(scores_log_reg)
 
log_accuracies_reg_array = np.array(log_accuracies_reg)
log_accuracies_reg_array = log_accuracies_reg_array.reshape(14,1)
log_accuracies_reg_array = log_accuracies_reg_array.T


df_log_reg =  pd.DataFrame(log_accuracies_reg_array, index= ['Logistic Regr reg'],columns = drugs_names) 

df_all = df_all.append(df_log_reg)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  
LDA =  LinearDiscriminantAnalysis()

lda_accuracies = []
for k in range(14):
    fitted_LDA = LDA.fit(X_train, y_train[:,k])
    scores_LDA = LDA.score(X_test, y_test[:,k])
    lda_accuracies.append(scores_LDA)
    print(scores_LDA)
    
lda_accuracies_array = np.array(lda_accuracies)
lda_accuracies_array = lda_accuracies_array.reshape(14,1)
lda_accuracies_array=lda_accuracies_array.T    
    
df_LDA =  pd.DataFrame(lda_accuracies_array, index= ['LDA'],columns = drugs_names) 
   
df_all = df_all.append(df_LDA)


#QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
 
QDA =   QuadraticDiscriminantAnalysis()

qda_accuracies = []
for k in range(14):
    fitted_QDA = QDA.fit(X_train, y_train[:,k])
    scores_QDA = QDA.score(X_test, y_test[:,k])
    qda_accuracies.append(scores_QDA)
    print(scores_QDA)
    
qda_accuracies_array = np.array(qda_accuracies)
qda_accuracies_array = qda_accuracies_array.reshape(14,1)
qda_accuracies_array= qda_accuracies_array.T    
    
df_QDA =  pd.DataFrame(qda_accuracies_array, index= ['QDA'],columns = drugs_names) 
   
df_all = df_all.append(df_QDA)

#Naie Bayes
from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()

NB_accuracies = []
for k in range(14):
    fitted_NB = NB.fit(X_train, y_train[:,k])
    scores_NB = NB.score(X_test, y_test[:,k])
    NB_accuracies.append(scores_NB)
    print(scores_NB)

NB_accuracies_array = np.array(NB_accuracies)
NB_accuracies_array = NB_accuracies_array.reshape(14,1)
NB_accuracies_array=NB_accuracies_array.T    
    

df_NB =  pd.DataFrame(NB_accuracies_array, index= ['NB'],columns = drugs_names) 

df_all =df_all.append(df_NB)


# SVM
from sklearn.svm import SVC

SVM = SVC(kernel='rbf', C=1, gamma=0.067, decision_function_shape= "ovr", random_state= 18)

SVM_accuracies = []
for k in range(14):
    fitted_SVM = SVM.fit(X_train, y_train[:,k])
    scores_SVM = SVM.score(X_test, y_test[:,k])
    SVM_accuracies.append(scores_SVM)
    print(scores_SVM)

SVM_accuracies_array = np.array(SVM_accuracies)
SVM_accuracies_array = SVM_accuracies_array.reshape(14,1)
SVM_accuracies_array= SVM_accuracies_array.T    
    

df_SVM =  pd.DataFrame(SVM_accuracies_array, index= ['SVM'],columns = drugs_names) 

df_all =df_all.append(df_SVM)


# KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics 
from collections import defaultdict

k_range = range(1,11)
scores = defaultdict(list)
scores_list = []
for k in k_range:
    for l in range(14):
        knn = KNeighborsClassifier(n_neighbors=k)
        fitted_knn = knn.fit(X_train, y_train[:,l])
    
    
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores[k].append(metrics.accuracy_score(y_test[:,l], y_pred[:,l]))
        scores_list.append(metrics.accuracy_score(y_test[:,l], y_pred[:,l]))



knn_accuracies = np.empty(shape=(10,14))
k1 = np.array(scores[1]).reshape(1,14)
k2 = np.array(scores[2]).reshape(1,14)
k3 =  np.array(scores[3]).reshape(1,14)
k4 = np.array(scores[4]).reshape(1,14)
k5 = np.array(scores[5]).reshape(1,14)
k6 =  np.array(scores[6]).reshape(1,14)
k7 = np.array(scores[7]).reshape(1,14)
k8 = np.array(scores[8]).reshape(1,14)
k9 =  np.array(scores[9]).reshape(1,14)
k10 =  np.array(scores[10]).reshape(1,14)

k_all = np.concatenate((k1,k2,k3,k4,k5,k6,k7,k8,k9,k10),axis=0)
k_accuracies = np.sum((k_all),axis=1)
df_knn =pd.DataFrame(k_all, columns = drugs_names)

df_knn2= np.amax(df_knn, axis=0)
knn_accuracies_array= np.array(df_knn2)
knn_accuracies_array = knn_accuracies_array.reshape(14,1)
knn_accuracies_array= knn_accuracies_array.T

df_KNN =  pd.DataFrame(knn_accuracies_array, index= ['KNN'],columns = drugs_names) 

df_all = df_all.append(df_KNN)  

#Neural Network
from sklearn.neural_network import MLPClassifier

NN = MLPClassifier(hidden_layer_sizes=(100,), batch_size= 25, learning_rate='adaptive', early_stopping= True, validation_fraction=0.1, random_state=18)

NN_accuracies = []
for k in range(14):
    fitted_NN = NN.fit(X_train, y_train[:,k])
    scores_NN = NN.score(X_test, y_test[:,k])
    NN_accuracies.append(scores_NN)
    print(scores_NN)

NN_accuracies_array = np.array(NN_accuracies)
NN_accuracies_array = NN_accuracies_array.reshape(14,1)
NN_accuracies_array= NN_accuracies_array.T    
    

df_NN =  pd.DataFrame(NN_accuracies_array, index= ['NN'],columns = drugs_names) 

df_all =df_all.append(df_NN)


# Decision Tree
from sklearn.tree import DecisionTreeClassifier

Decision_Tree = DecisionTreeClassifier(criterion='gini', min_impurity_decrease= 0.005,max_features= 'log2', random_state=18)

Decision_Tree_accuracies = []
for k in range(14):
    fitted_Decision_Tree = Decision_Tree.fit(X_train, y_train[:,k])
    scores_Decision_Tree = Decision_Tree.score(X_test, y_test[:,k])
    Decision_Tree_accuracies.append(scores_Decision_Tree)
    print(scores_Decision_Tree)

Decision_Tree_accuracies_array = np.array(Decision_Tree_accuracies)
Decision_Tree_accuracies_array = Decision_Tree_accuracies_array.reshape(14,1)
Decision_Tree_accuracies_array= Decision_Tree_accuracies_array.T    
    

df_Decision_Tree =  pd.DataFrame(Decision_Tree_accuracies_array, index= ['Decision Tree'],columns = drugs_names) 

df_all =df_all.append(df_Decision_Tree)

#####

# Best classifier for each drug
best_class_acc = np.amax(df_all, axis=0) # Highest accuracy per drug given all classifiers
best_class_acc

best_classifier= df_all.idxmax(axis=0, skipna=True) # Translation of those accuracy in classifier name
best_classifier


###### CONFUSION MATRICES AND SENSITIVITY/SPECIFICITY SCORES #####
from sklearn.metrics import confusion_matrix

# AMPHET using Logistic Regr reg
log_classifier_reg= linear_model.SGDClassifier(loss = 'log', alpha= 0.001, penalty = 'L2', max_iter=10000, tol=1e-3, random_state= 18)

log_reg_pred_amphet = []
for k in range(14):
    fitted_log_reg = log_classifier_reg.fit(X_train, y_train[:,k])
    predict_log_reg = log_classifier_reg.predict(X_test)
    log_reg_pred_amphet.append(predict_log_reg)

amphet_pred = log_reg_pred_amphet[0]
amphet_pred = pd.DataFrame(amphet_pred) # predicted values
y_test_amphet = pd.DataFrame(y_test)[0]

conf_matrix_amphet = confusion_matrix(y_test_amphet, amphet_pred)

conf_matrix_amphet = pd.DataFrame(conf_matrix_amphet, columns=["Predicted Negative", "Predicted Positive"])
conf_matrix_amphet.rename(index={0:'Actual Negative', 1:'Actual Positive'}, inplace=True)
conf_matrix_amphet

TP_amphet = conf_matrix_amphet.iloc[1,1] 
TN_amphet = conf_matrix_amphet.iloc[0,0] 
FP_amphet = conf_matrix_amphet.iloc[0,1] 
FN_amphet = conf_matrix_amphet.iloc[1,0] 

sensitivity_amphet = TP_amphet/(TP_amphet+FN_amphet)
sensitivity_amphet

specificity_amphet = TN_amphet/(TN_amphet+FP_amphet)
specificity_amphet

# AMYL using Linear KNN
KNN = KNeighborsClassifier(n_neighbors=5)

KNN_pred_amyl = []  
for i in range(14):
    fitted_KNN = KNN.fit(X_train, y_train[:,i])
    predict_KNN = KNN.predict(X_test)
    KNN_pred_amyl.append(predict_KNN)

amyl_pred = KNN_pred_amyl[1]
amyl_pred = pd.DataFrame(amyl_pred) # predicted values
y_test_amyl = pd.DataFrame(y_test)[1]

conf_matrix_amyl = confusion_matrix(y_test_amyl, amyl_pred)

conf_matrix_amyl= pd.DataFrame(conf_matrix_amyl, columns=["Predicted Negative", "Predicted Positive"])
conf_matrix_amyl.rename(index={0:'Actual Negative', 1:'Actual Positive'}, inplace=True)
conf_matrix_amyl

TP_amyl = conf_matrix_amyl.iloc[1,1] 
TN_amyl = conf_matrix_amyl.iloc[0,0] 
FP_amyl = conf_matrix_amyl.iloc[0,1] 
FN_amyl = conf_matrix_amyl.iloc[1,0] 

sensitivity_amyl = TP_amyl/(TP_amyl+FN_amyl)
sensitivity_amyl

specificity_amyl = TN_amyl/(TN_amyl+FP_amyl)
specificity_amyl

# BENZOS using KNN
KNN = KNeighborsClassifier(n_neighbors=9)

KNN_pred_benzos = []  
for i in range(14):
    fitted_KNN = KNN.fit(X_train, y_train[:,i])
    predict_KNN = KNN.predict(X_test)
    KNN_pred_benzos.append(predict_KNN)

benzos_pred = KNN_pred_benzos[2]
benzos_pred = pd.DataFrame(benzos_pred) # predicted values
y_test_benzos = pd.DataFrame(y_test)[2]

conf_matrix_benzos = confusion_matrix(y_test_benzos, benzos_pred)

conf_matrix_benzos= pd.DataFrame(conf_matrix_benzos, columns=["Predicted Negative", "Predicted Positive"])
conf_matrix_benzos.rename(index={0:'Actual Negative', 1:'Actual Positive'}, inplace=True)
conf_matrix_benzos

TP_benzos = conf_matrix_benzos.iloc[1,1] 
TN_benzos = conf_matrix_benzos.iloc[0,0] 
FP_benzos = conf_matrix_benzos.iloc[0,1] 
FN_benzos = conf_matrix_benzos.iloc[1,0] 

sensitivity_benzos= TP_benzos/(TP_benzos+FN_benzos)
sensitivity_benzos

specificity_benzos= TN_benzos/(TN_benzos+FP_benzos)
specificity_benzos

# COKE using Logistic Regr reg
log_classifier_reg= linear_model.SGDClassifier(loss = 'log', alpha= 0.001, penalty = 'L2', max_iter=10000, tol=1e-3, random_state= 18)

log_reg_pred_coke = []
for k in range(14):
    fitted_log_reg = log_classifier_reg.fit(X_train, y_train[:,k])
    predict_log_reg = log_classifier_reg.predict(X_test)
    log_reg_pred_coke.append(predict_log_reg)

coke_pred = log_reg_pred_coke[3]
coke_pred = pd.DataFrame(coke_pred) # predicted values
y_test_coke = pd.DataFrame(y_test)[3]

conf_matrix_coke = confusion_matrix(y_test_coke, coke_pred)

conf_matrix_coke = pd.DataFrame(conf_matrix_coke, columns=["Predicted Negative", "Predicted Positive"])
conf_matrix_coke.rename(index={0:'Actual Negative', 1:'Actual Positive'}, inplace=True)
conf_matrix_coke

TP_coke = conf_matrix_coke.iloc[1,1] 
TN_coke = conf_matrix_coke.iloc[0,0] 
FP_coke = conf_matrix_coke.iloc[0,1] 
FN_coke = conf_matrix_coke.iloc[1,0] 

sensitivity_coke = TP_coke/(TP_coke+FN_coke)
sensitivity_coke

specificity_coke = TN_coke/(TN_coke+FP_coke)
specificity_coke

# CRACK using KNN
KNN = KNeighborsClassifier(n_neighbors=8)

KNN_pred_crack = []  
for i in range(14):
    fitted_KNN = KNN.fit(X_train, y_train[:,i])
    predict_KNN = KNN.predict(X_test)
    KNN_pred_crack.append(predict_KNN)

crack_pred = KNN_pred_crack[4]
crack_pred = pd.DataFrame(crack_pred) # predicted values
y_test_crack = pd.DataFrame(y_test)[4] # true values

conf_matrix_crack = confusion_matrix(y_test_crack, crack_pred)

conf_matrix_crack = pd.DataFrame(conf_matrix_crack, columns=["Predicted Negative", "Predicted Positive"])
conf_matrix_crack.rename(index={0:'Actual Negative', 1:'Actual Positive'}, inplace=True)
conf_matrix_crack

TP_crack = conf_matrix_crack.iloc[1,1] 
TN_crack = conf_matrix_crack.iloc[0,0] 
FP_crack = conf_matrix_crack.iloc[0,1] 
FN_crack = conf_matrix_crack.iloc[1,0] 

sensitivity_crack= TP_crack/(TP_crack+FN_crack)
sensitivity_crack

specificity_crack= TN_crack/(TN_crack+FP_crack)
specificity_crack

# ECSTASY using QDA
QDA =   QuadraticDiscriminantAnalysis()

qda_pred_ecstasy = []
for k in range(14):
    fitted_QDA = QDA.fit(X_train, y_train[:,k])
    predict_QDA = QDA.predict(X_test)
    qda_pred_ecstasy.append(predict_QDA)

ecstasy_pred = qda_pred_ecstasy[5]
ecstasy_pred = pd.DataFrame(ecstasy_pred) # predicted values
y_test_ecstasy = pd.DataFrame(y_test)[5]

conf_matrix_ecstasy = confusion_matrix(y_test_ecstasy, ecstasy_pred)

conf_matrix_ecstasy= pd.DataFrame(conf_matrix_ecstasy, columns=["Predicted Negative", "Predicted Positive"])
conf_matrix_ecstasy.rename(index={0:'Actual Negative', 1:'Actual Positive'}, inplace=True)
conf_matrix_ecstasy

TP_ecstasy = conf_matrix_ecstasy.iloc[1,1] 
TN_ecstasy = conf_matrix_ecstasy.iloc[0,0] 
FP_ecstasy = conf_matrix_ecstasy.iloc[0,1] 
FN_ecstasy = conf_matrix_ecstasy.iloc[1,0] 

sensitivity_ecstasy= TP_ecstasy/(TP_ecstasy+FN_ecstasy)
sensitivity_ecstasy

specificity_ecstasy= TN_ecstasy/(TN_coke+FP_ecstasy)
specificity_ecstasy

# HEROIN using Logistic Regr
log_classifier = linear_model.SGDClassifier(loss = 'log', alpha= 0.0001,max_iter=10000, penalty = None, tol=1e-3, random_state= 18)

log_pred_heroin = []    
for i in range(14):
    fitted_log = log_classifier.fit(X_train, y_train[:,i])
    predict_log = log_classifier.predict(X_test)
    log_pred_heroin.append(predict_log)

heroin_pred = log_pred_heroin[6]
heroin_pred = pd.DataFrame(heroin_pred) # predicted values
y_test_heroin = pd.DataFrame(y_test)[6]

conf_matrix_heroin = confusion_matrix(y_test_heroin, heroin_pred)

conf_matrix_heroin = pd.DataFrame(conf_matrix_heroin, columns=["Predicted Negative", "Predicted Positive"])
conf_matrix_heroin.rename(index={0:'Actual Negative', 1:'Actual Positive'}, inplace=True)
conf_matrix_heroin

TP_heroin = conf_matrix_heroin.iloc[1,1] 
TN_heroin = conf_matrix_heroin.iloc[0,0] 
FP_heroin = conf_matrix_heroin.iloc[0,1] 
FN_heroin = conf_matrix_heroin.iloc[1,0] 

sensitivity_heroin= TP_heroin/(TP_heroin+FN_heroin)
sensitivity_heroin

specificity_heroin= TN_heroin/(TN_heroin+FP_heroin)
specificity_heroin

# KETAMINE using Linear SGD
linear_classifier= linear_model.SGDClassifier(loss = 'squared_loss', alpha= 0.0001, penalty = None, max_iter=10000, tol=1e-3, random_state= 18)

linear_pred_ketamine = []
for k in range(14):
    fitted_linear = linear_classifier.fit(X_train, y_train[:,k])
    predict_linear = linear_classifier.predict(X_test)
    linear_pred_ketamine.append(predict_linear)

ketamine_pred = linear_pred_ketamine[7]
ketamine_pred = pd.DataFrame(ketamine_pred) # predicted values
y_test_ketamine = pd.DataFrame(y_test)[7]

conf_matrix_ketamine = confusion_matrix(y_test_ketamine, ketamine_pred)

conf_matrix_ketamine = pd.DataFrame(conf_matrix_ketamine, columns=["Predicted Negative", "Predicted Positive"])
conf_matrix_ketamine.rename(index={0:'Actual Negative', 1:'Actual Positive'}, inplace=True)
conf_matrix_ketamine

TP_ketamine = conf_matrix_ketamine.iloc[1,1] 
TN_ketamine = conf_matrix_ketamine.iloc[0,0] 
FP_ketamine = conf_matrix_ketamine.iloc[0,1] 
FN_ketamine = conf_matrix_ketamine.iloc[1,0] 

sensitivity_ketamine= TP_ketamine/(TP_ketamine+FN_ketamine)
sensitivity_ketamine

specificity_ketamine= TN_ketamine/(TN_ketamine+FP_ketamine)
specificity_ketamine

# LEGALH using QDA
QDA = QuadraticDiscriminantAnalysis()

qda_pred_legalh = []
for k in range(14):
    fitted_QDA = QDA.fit(X_train, y_train[:,k])
    predict_QDA = QDA.predict(X_test)
    qda_pred_legalh.append(predict_QDA)
    
legalh_pred = qda_pred_legalh[8]
legalh_pred = pd.DataFrame(legalh_pred) # predicted values
y_test_legalh = pd.DataFrame(y_test)[8]

conf_matrix_legalh = confusion_matrix(y_test_legalh, legalh_pred)

conf_matrix_legalh= pd.DataFrame(conf_matrix_legalh, columns=["Predicted Negative", "Predicted Positive"])
conf_matrix_legalh.rename(index={0:'Actual Negative', 1:'Actual Positive'}, inplace=True)
conf_matrix_legalh

TP_legalh = conf_matrix_legalh.iloc[1,1] 
TN_legalh = conf_matrix_legalh.iloc[0,0] 
FP_legalh = conf_matrix_legalh.iloc[0,1] 
FN_legalh = conf_matrix_legalh.iloc[1,0] 

sensitivity_legalh= TP_legalh/(TP_legalh+FN_legalh)
sensitivity_legalh

specificity_legalh= TN_legalh/(TN_legalh+FP_legalh)
specificity_legalh

# LSD using Logistic Regr reg
log_classifier_reg= linear_model.SGDClassifier(loss = 'log', alpha= 0.001, penalty = 'L2', max_iter=10000, tol=1e-3, random_state= 18)

log_reg_pred_lsd = []
for k in range(14):
    fitted_log_reg = log_classifier_reg.fit(X_train, y_train[:,k])
    predict_log_reg = log_classifier_reg.predict(X_test)
    log_reg_pred_lsd.append(predict_log_reg)

lsd_pred = log_reg_pred_lsd[9]
lsd_pred = pd.DataFrame(lsd_pred) # predicted values
y_test_lsd = pd.DataFrame(y_test)[9]

conf_matrix_lsd = confusion_matrix(y_test_lsd, lsd_pred)

conf_matrix_lsd = pd.DataFrame(conf_matrix_lsd, columns=["Predicted Negative", "Predicted Positive"])
conf_matrix_lsd.rename(index={0:'Actual Negative', 1:'Actual Positive'}, inplace=True)
conf_matrix_lsd

TP_lsd = conf_matrix_lsd.iloc[1,1] 
TN_lsd = conf_matrix_lsd.iloc[0,0] 
FP_lsd = conf_matrix_lsd.iloc[0,1] 
FN_lsd = conf_matrix_lsd.iloc[1,0] 

sensitivity_lsd= TP_lsd/(TP_lsd+FN_lsd)
sensitivity_lsd

specificity_lsd= TN_lsd/(TN_lsd+FP_lsd)
specificity_lsd

# METH using Logistic Regr
log_classifier= linear_model.SGDClassifier(loss = 'log', alpha= 0.0001, penalty = None, max_iter=10000, tol=1e-3, random_state= 18)

log_pred_meth = []
for k in range(14):
    fitted_log = log_classifier.fit(X_train, y_train[:,k])
    predict_log = log_classifier.predict(X_test)
    log_pred_meth.append(predict_log)

meth_pred = log_pred_meth[10]
meth_pred = pd.DataFrame(meth_pred) # predicted values
y_test_meth = pd.DataFrame(y_test)[10]

conf_matrix_meth = confusion_matrix(y_test_meth, meth_pred)

conf_matrix_meth = pd.DataFrame(conf_matrix_meth, columns=["Predicted Negative", "Predicted Positive"])
conf_matrix_meth.rename(index={0:'Actual Negative', 1:'Actual Positive'}, inplace=True)
conf_matrix_meth

TP_meth = conf_matrix_meth.iloc[1,1] 
TN_meth = conf_matrix_meth.iloc[0,0] 
FP_meth = conf_matrix_meth.iloc[0,1] 
FN_meth = conf_matrix_meth.iloc[1,0] 

sensitivity_meth= TP_meth/(TP_meth+FN_meth)
sensitivity_meth

specificity_meth= TN_meth/(TN_meth+FP_meth)
specificity_meth
  
# MUSHROOM using KNN
KNN = KNeighborsClassifier(n_neighbors=10)

KNN_pred_mushroom = []  
for i in range(14):
    fitted_KNN = KNN.fit(X_train, y_train[:,i])
    predict_KNN = KNN.predict(X_test)
    KNN_pred_mushroom.append(predict_KNN)

mushroom_pred = KNN_pred_mushroom[11]
mushroom_pred = pd.DataFrame(mushroom_pred) # predicted values
y_test_mushroom = pd.DataFrame(y_test)[11] # true values

conf_matrix_mushroom = confusion_matrix(y_test_mushroom, mushroom_pred)

conf_matrix_mushroom= pd.DataFrame(conf_matrix_mushroom, columns=["Predicted Negative", "Predicted Positive"])
conf_matrix_mushroom.rename(index={0:'Actual Negative', 1:'Actual Positive'}, inplace=True)
conf_matrix_mushroom

TP_mushroom = conf_matrix_mushroom.iloc[1,1] 
TN_mushroom = conf_matrix_mushroom.iloc[0,0] 
FP_mushroom = conf_matrix_mushroom.iloc[0,1] 
FN_mushroom = conf_matrix_mushroom.iloc[1,0] 

sensitivity_mushroom = TP_mushroom/(TP_mushroom+FN_mushroom)
sensitivity_mushroom

specificity_mushroom= TN_mushroom/(TN_mushroom+FP_mushroom)
specificity_mushroom

# SEMER using Linear SGD
linear_classifier = linear_model.SGDClassifier(loss = 'squared_loss', alpha= 0.0001,max_iter=10000, penalty = None, tol=1e-3, random_state= 18)

linear_pred_semer = []    
for i in range(14):
    fitted_linear = linear_classifier.fit(X_train, y_train[:,i])
    predict_linear = linear_classifier.predict(X_test)
    linear_pred_semer.append(predict_linear)

semer_pred = linear_pred_semer[12]
semer_pred = pd.DataFrame(semer_pred) # predicted values
y_test_semer = pd.DataFrame(y_test)[12]

conf_matrix_semer = confusion_matrix(y_test_semer, semer_pred)

conf_matrix_semer = pd.DataFrame({'Predicted Negative': [189, 0],
           'Predicted Positive': [0, 0]})
    
conf_matrix_semer.rename(index={0:'Actual Negative', 1:'Actual Positive'}, inplace=True)
conf_matrix_semer

TP_semer = conf_matrix_semer.iloc[1,1] 
TN_semer = conf_matrix_semer.iloc[0,0] 
FP_semer = conf_matrix_semer.iloc[0,1] 
FN_semer = conf_matrix_semer.iloc[1,0] 

sensitivity_semer= TP_semer/(TP_semer+FN_semer)
sensitivity_semer

specificity_semer= TN_semer/(TN_semer+FP_semer)
specificity_semer


# VSA using Logistic Regr reg
log_classifier_reg = linear_model.SGDClassifier(loss = 'log', alpha= 0.001,max_iter=10000, penalty = 'L2', tol=1e-3, random_state= 18)

log_reg_pred_vsa = []    
for i in range(14):
    fitted_log_reg = log_classifier_reg.fit(X_train, y_train[:,i])
    predict_log_reg = log_classifier_reg.predict(X_test)
    log_reg_pred_vsa.append(predict_log_reg)

vsa_pred = log_reg_pred_vsa[13]
vsa_pred = pd.DataFrame(vsa_pred) # predicted values
y_test_vsa = pd.DataFrame(y_test)[13]

conf_matrix_vsa = confusion_matrix(y_test_vsa, vsa_pred)

conf_matrix_vsa = pd.DataFrame(conf_matrix_vsa, columns=["Predicted Negative", "Predicted Positive"])
conf_matrix_vsa.rename(index={0:'Actual Negative', 1:'Actual Positive'}, inplace=True)
conf_matrix_vsa

TP_vsa = conf_matrix_vsa.iloc[1,1] 
TN_vsa = conf_matrix_vsa.iloc[0,0] 
FP_vsa = conf_matrix_vsa.iloc[0,1] 
FN_vsa = conf_matrix_vsa.iloc[1,0] 

sensitivity_vsa= TP_vsa/(TP_vsa+FN_vsa)
sensitivity_vsa

specificity_vsa= TN_vsa/(TN_vsa+FP_vsa)
specificity_vsa

sens_specif_all = pd.DataFrame({'Sensitivity': [sensitivity_amphet, sensitivity_amyl,
                                                sensitivity_benzos,
                                                sensitivity_coke, sensitivity_crack,
                                                sensitivity_ecstasy, sensitivity_heroin,
                                                sensitivity_ketamine, sensitivity_legalh,
                                                sensitivity_lsd, sensitivity_meth,
                                                sensitivity_mushroom,
                                                sensitivity_semer, sensitivity_vsa],
                                'Specificity':[specificity_amphet,specificity_amyl,
                                                specificity_benzos,
                                                specificity_coke, specificity_crack,
                                                specificity_ecstasy, specificity_heroin,
                                                specificity_ketamine, specificity_legalh,
                                                specificity_lsd, specificity_meth,
                                                specificity_mushroom,
                                                specificity_semer, specificity_vsa]})


sens_specif_all.rename(index={0:'Amphet', 1:'Amyl', 2:'Benzos', 3:'Coke', 4:'Crack', 5: 'Ecstasy', 6: 'Heroin', 7: 'Ketamine',
                              8: 'Legalh', 9: 'LSD', 10: 'Meth', 11:'Mushroom', 12:'Semer',
                              13: 'VSA'}, inplace=True)
sens_specif_all

# Saving sens specif all as a csv
sens_specif_all.to_csv('sens_specif_all.csv', index=True)

# Saving confusion matrices to csv
tfile = open('confusion matrices.csv', 'w')
tfile.write(conf_matrix_amphet.to_csv())
tfile.close()

tfile = open('confusion matrices.csv', 'a')
tfile.write(conf_matrix_amyl.to_csv())
tfile.close()

tfile = open('confusion matrices.csv', 'a')
tfile.write(conf_matrix_benzos.to_csv())
tfile.close()

tfile = open('confusion matrices.csv', 'a')
tfile.write(conf_matrix_coke.to_csv())
tfile.close()

tfile = open('confusion matrices.csv', 'a')
tfile.write(conf_matrix_crack.to_csv())
tfile.close()

tfile = open('confusion matrices.csv', 'a')
tfile.write(conf_matrix_ecstasy.to_csv())
tfile.close()

tfile = open('confusion matrices.csv', 'a')
tfile.write(conf_matrix_heroin.to_csv())
tfile.close()

tfile = open('confusion matrices.csv', 'a')
tfile.write(conf_matrix_ketamine.to_csv())
tfile.close()

tfile = open('confusion matrices.csv', 'a')
tfile.write(conf_matrix_legalh.to_csv())
tfile.close()

tfile = open('confusion matrices.csv', 'a')
tfile.write(conf_matrix_lsd.to_csv())
tfile.close()

tfile = open('confusion matrices.csv', 'a')
tfile.write(conf_matrix_meth.to_csv())
tfile.close()

tfile = open('confusion matrices.csv', 'a')
tfile.write(conf_matrix_mushroom.to_csv())
tfile.close()

tfile = open('confusion matrices.csv', 'a')
tfile.write(conf_matrix_semer.to_csv())
tfile.close()

tfile = open('confusion matrices.csv', 'a')
tfile.write(conf_matrix_vsa.to_csv())
tfile.close()

# Save csv.file with best accuracy for each drug
best_class_acc.to_csv('best_class_acc.csv', index=True)

# Save csv.file with best classifier for each drug
best_classifier.to_csv('best_classifier.csv', index=True)

# Save cvs.file with all scores for all classifiers and drugs
df_all.to_csv('df_all.csv', index=True)
 
# Saving input features to .csv file
X_all.to_csv('X_all.csv', index = True) 

# Saving cvs.file with descriptive statistics
describe.to_csv('descriptive_stat.csv', index= True)




















