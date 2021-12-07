#Dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns  
from sklearn.model_selection import cross_val_score



#Read in BCSC Risk Estimation data
dataset = pd.read_csv('BCSC Risk Estimation Dataset One Observation per woman.csv', low_memory = False)
dataset.drop('training', axis = 1, inplace = True)
Y = dataset['cancer'].values
X = dataset.drop('cancer', axis = 1).values

print(dataset.columns)

#split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_) 
feat_importances = pd.Series(model.feature_importances_, index=dataset.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()




corr = dataset[X].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15}, xticklabels = X, yticklabels = X, cmap= 'coolwarm')
plt.show()

#Using Pearson Correlation
df = pd.DataFrame(X)
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#Visualization of data
dataset.groupby('cancer').hist(figsize=(12, 12))
dataset.isnull().sum()
dataset.isna().sum()
dataframe = pd.DataFrame(Y)

#Logistic Regression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
predictions = [round(value) for value in Y_pred]
Y_test = np.squeeze(Y_test)

cm = confusion_matrix(Y_test, predictions)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Logistic Regression')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)
plt.show()

log_reg_accuracy = metrics.accuracy_score(predictions, Y_test)

log_reg_scores = cross_val_score(classifier, X_train, Y_train, cv = 5)

print(log_reg_scores.mean(), log_reg_scores.std())

#Fitting K-NN Algorithm
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
predictions = [round(value) for value in Y_pred]
Y_test = np.squeeze(Y_test)

cm = confusion_matrix(Y_test, predictions)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('K-NN')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)
plt.show()

KNN_accuracy_score = metrics.accuracy_score(predictions, Y_test)


#Fitting SVM
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train) 
Y_pred = classifier.predict(X_test)
predictions = [round(value) for value in Y_pred]
Y_test = np.squeeze(Y_test)

cm = confusion_matrix(Y_test, predictions)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('SVM')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)
plt.show()

SVM_accuracy_score = metrics.accuracy_score(predictions, Y_test)


#Fitting K-SVM
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
predictions = [round(value) for value in Y_pred]
Y_test = np.squeeze(Y_test)

cm = confusion_matrix(Y_test, predictions)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('K-SVM')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)
plt.show()

K_SVM_accuracy_score = metrics.accuracy_score(predictions, Y_test)

#Fitting Naive_Bayes
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
predictions = [round(value) for value in Y_pred]
Y_test = np.squeeze(Y_test)

cm = confusion_matrix(Y_test, predictions)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Naive Bayes')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)
plt.show()

Naive_Bayes_accuracy_score = metrics.accuracy_score(predictions, Y_test)

#Fitting Decision Tree Algorithm
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
predictions = [round(value) for value in Y_pred]
Y_test = np.squeeze(Y_test)

cm = confusion_matrix(Y_test, predictions)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Decision Tree')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)
plt.show()

Decision_tree_accuracy_score = metrics.accuracy_score(predictions, Y_test)


#Fitting Random Forest Classification Algorithm
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
predictions = [round(value) for value in Y_pred]
Y_test = np.squeeze(Y_test)

cm = confusion_matrix(Y_test, predictions)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Random Forrest')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)
plt.show()

Random_Forrest_accuracy_score = metrics.accuracy_score(predictions, Y_test)


#Accuracy Scores
print('Logistic Regression Accuracy Score = ' , log_reg_accuracy)
print('SVM Accuracy Score = ', SVM_accuracy_score)
print('K-NN Accuracy Score = ', KNN_accuracy_score)
print('K-SVM Accuracy Score = ', K_SVM_accuracy_score)
print('Naive Bayes Accuracy Score = ', Naive_Bayes_accuracy_score)
print('Decision Tree Accuracy Score = ', Decision_tree_accuracy_score)
print('Random Forrest Accuracy Score = ', Random_Forrest_accuracy_score)
