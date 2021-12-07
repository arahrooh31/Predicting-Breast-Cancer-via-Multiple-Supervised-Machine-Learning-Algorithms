#Dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns  
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


###PREPROCESSING###


#importing the dataset 
dataset = pd.read_csv("data.csv", header = 0)
dataset.drop("Unnamed: 32", axis = 1, inplace = True)
dataset.drop("id",axis=1,inplace=True)
dataset['diagnosis']=dataset['diagnosis'].map({'M':1,'B':0})
Y = dataset['diagnosis'].values
X = dataset.drop('diagnosis', axis=1).values

#Analysis
corr = dataset[X].corr() 
plt.figure(figsize=(14,14))
sns.heatmap(dataset[X].corr() , cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15}, xticklabels= X, yticklabels= X, cmap= 'coolwarm')
plt.show()


# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

#Visualization of data
dataset.groupby('diagnosis').hist(figsize=(12, 12))
dataset.isnull().sum()
dataset.isna().sum()
dataframe = pd.DataFrame(Y)

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


####MODELING###


#Logistic Regression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
predictions = [round(value) for value in Y_pred]
Y_test = np.squeeze(Y_test)

cm = confusion_matrix(Y_test, predictions)

def plot_confusion_matrix1(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Logistic Regression')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix1(cm)
plt.show()

log_reg_accuracy = metrics.accuracy_score(predictions, Y_test)
log_reg_scores = cross_val_score(classifier, X_train, Y_train, cv = 5)


#Fitting K-NN Algorithm
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
predictions = [round(value) for value in Y_pred]
Y_test = np.squeeze(Y_test)

cm = confusion_matrix(Y_test, predictions)

def plot_confusion_matrix2(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('K-NN')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix2(cm)
plt.show()

KNN_accuracy_score = metrics.accuracy_score(predictions, Y_test)
KNN_reg_scores = cross_val_score(classifier, X_train, Y_train, cv = 5)


#Fitting SVM
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train) 
Y_pred = classifier.predict(X_test)
predictions = [round(value) for value in Y_pred]
Y_test = np.squeeze(Y_test)

cm = confusion_matrix(Y_test, predictions)

def plot_confusion_matrix3(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('SVM')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix3(cm)
plt.show()

SVM_accuracy_score = metrics.accuracy_score(predictions, Y_test)
SVM_reg_scores = cross_val_score(classifier, X_train, Y_train, cv = 5)


#Fitting K-SVM
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
predictions = [round(value) for value in Y_pred]
Y_test = np.squeeze(Y_test)

cm = confusion_matrix(Y_test, predictions)

def plot_confusion_matrix4(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('K-SVM')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix4(cm)
plt.show()

K_SVM_accuracy_score = metrics.accuracy_score(predictions, Y_test)
K_SVM_reg_scores = cross_val_score(classifier, X_train, Y_train, cv = 5)


#Fitting Naive_Bayes
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
predictions = [round(value) for value in Y_pred]
Y_test = np.squeeze(Y_test)

cm = confusion_matrix(Y_test, predictions)

def plot_confusion_matrix5(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Naive Bayes')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix5(cm)
plt.show()

Naive_Bayes_accuracy_score = metrics.accuracy_score(predictions, Y_test)
NB_reg_scores = cross_val_score(classifier, X_train, Y_train, cv = 5)


#Fitting Decision Tree Algorithm
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
predictions = [round(value) for value in Y_pred]
Y_test = np.squeeze(Y_test)

cm = confusion_matrix(Y_test, predictions)

def plot_confusion_matrix6(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Decision Tree')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix6(cm)
plt.show()

Decision_tree_accuracy_score = metrics.accuracy_score(predictions, Y_test)
DT_reg_scores = cross_val_score(classifier, X_train, Y_train, cv = 5)


#Fitting Random Forest Classification Algorithm
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
predictions = [round(value) for value in Y_pred]
Y_test = np.squeeze(Y_test)

cm = confusion_matrix(Y_test, predictions)

def plot_confusion_matrix7(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Random Forrest')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix7(cm)
plt.show()

Random_Forrest_accuracy_score = metrics.accuracy_score(predictions, Y_test)
RF_reg_scores = cross_val_score(classifier, X_train, Y_train, cv = 5)


####EVALUATION####


#Accuracy Scores
print('Logistic Regression Accuracy Score = ' , log_reg_accuracy)
print('SVM Accuracy Score = ', SVM_accuracy_score)
print('K-NN Accuracy Score = ', KNN_accuracy_score)
print('K-SVM Accuracy Score = ', K_SVM_accuracy_score)
print('Naive Bayes Accuracy Score = ', Naive_Bayes_accuracy_score)
print('Decision Tree Accuracy Score = ', Decision_tree_accuracy_score)
print('Random Forrest Accuracy Score = ', Random_Forrest_accuracy_score)


#Cross Validation
print(log_reg_scores.mean(), "accuracy with a standard devciation of ", log_reg_scores.std())
print(KNN_reg_scores.mean(), "accuracy with a standard devciation of ", KNN_reg_scores.std())
print(SVM_reg_scores.mean(),"accuracy with a standard devciation of ", SVM_reg_scores.std())
print(K_SVM_reg_scores.mean(), "accuracy with a standard devciation of ", K_SVM_reg_scores.std())
print(NB_reg_scores.mean(), "accuracy with a standard devciation of ", NB_reg_scores.std())
print(DT_reg_scores.mean(), "accuracy with a standard devciation of ", DT_reg_scores.std())
print(RF_reg_scores.mean(), "accuracy with a standard devciation of ", RF_reg_scores.std())
