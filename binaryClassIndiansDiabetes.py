import warnings
warnings.filterwarnings("ignore")

# Load CSV using Pandas
from pandas import read_csv
filename = 'diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# UNDERSTANDING YOUR DATAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

peek = data.head(20) #Peek at Your Data
print(peek)
print('------------------------------------------------')

print(data.shape) #Dimensions of Your Data
print('------------------------------------------------')

types = data.dtypes #Data Type For Each Attribute
print(types)
print('------------------------------------------------')

from pandas import set_option
set_option('display.width', 100) #Descriptive Statistics
set_option('precision', 3)
description = data.describe()
print(description)
print('------------------------------------------------')

# Highly imbalanced problems (a lot more observations for one class than another)
# are common and may need special
# handling in the data preparation stage of your project.
class_counts = data.groupby('class').size() #Class Distribution
print(class_counts)
print('------------------------------------------------')

correlations = data.corr(method='pearson') #Correlations Between Attributes
print(correlations)
print('------------------------------------------------')

skew = data.skew() #Review the skew of the distributions of each attribute.
print(skew)
print('------------------------------------------------')


# Univariate Histograms
from matplotlib import pyplot
data.hist()
pyplot.show()

# Density Plots
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.show()

#Box and Whisker Plots
# Boxplots summarize the distribution of each attribute, drawing a
# line for the median (middle value) and a box around the 25th and 75th
#  percentiles (the middle 50% of the data).
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()


# plot correlation matrix
import numpy
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()



#PREPARING YOUR DATAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

# Rescale data (between 0 and 1)
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
dataframe = read_csv(filename, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])
print('------------------------------------------------')

# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])
print('------------------------------------------------')

# Normalize data (length of 1)
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(normalizedX[0:5,:])
print('------------------------------------------------')

# binarization
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(binaryX[0:5,:])

print('------------------------------------------------')
# FEATURE SELECTIONNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN

#Univariate Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])

print('------------------------------------------------')

# Feature Extraction with RFE: Recursive Feature Elimination
# The Recursive Feature Elimination (or RFE) works by recursively removing attributes and
# building a model on those attributes that remain
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print(fit.n_features_)
print(fit.support_)
print(fit.ranking_)

print('------------------------------------------------')
# Feature Importance
# given an importance score for each attribute where the larger the
# score, the more important the attribute
# Feature Importance with Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
print('------------------------------------------------')

# EVALUATE THE PERFORMANCE OF MACHINE LEARNING ON UNSEEN DATAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

# Evaluate using a train and a test set
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
print(f'Accuracy: {result*100.0}')
print('------------------------------------------------')

# Evaluate using K-Fold Cross Validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print(f'Accuracy: {results.mean()*100.0} , {results.std()*100.0}')
print('------------------------------------------------')


# Evaluate using Shuffle Split Cross Validation
from sklearn.model_selection import ShuffleSplit
n_splits = 10
test_size = 0.33
seed = 7
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print(f'Accuracy: {results.mean()*100.0} , {results.std()*100.0}')
print('------------------------------------------------')

# Algorithm Performance Metricsssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss

# Classification Accuracy
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
scoring = 'accuracy'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(f'Accuracy: {results.mean()} , {results.std()}')
print('------------------------------------------------')

# Classification LogLoss
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
scoring = 'neg_log_loss'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(f'Logloss: {results.mean()} , {results.std()}')
print('------------------------------------------------')

# Classification ROC AUC
#  for binary classification problems
scoring = 'roc_auc'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(f'AUC: {results.mean()} , {results.std()}')
print('------------------------------------------------')

#  Classification Report
from sklearn.metrics import classification_report
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)
print('------------------------------------------------')

# Classification ALGORITHMSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS

# Linear Machine Learning Algorithms : - logistic Regression, - linear discriminant analysis

# Logistic Regression
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print(f'Logistic Regression: {results.mean()}')
print('------------------------------------------------')


# LDA Classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()
results = cross_val_score(model, X, Y, cv=kfold)
print(f'LDA Classification: {results.mean()}')
print('------------------------------------------------')

#Non-Linear Machine Learning Algorithms:-k-Nearest neighbors, -naive bayes, -trees, -support vector machine

# k-Nearest neighbors: KNN Classification
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print(f'k-Nearest neighbors: {results.mean()}')
print('------------------------------------------------')

# Gaussian Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
results = cross_val_score(model, X, Y, cv=kfold)
print(f'Gaussian Naive Bayes: {results.mean()}')
print('------------------------------------------------')

# CART Classification: Classification And Regression Trees (decision trees)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print(f'Classification And Regression Trees: {results.mean()}')
print('------------------------------------------------')

# SupportVectorMachines SVC Classification
from sklearn.svm import SVC
model = SVC()
results = cross_val_score(model, X, Y, cv=kfold)
print(f'SupportVectorMachines: {results.mean()}')
print('------------------------------------------------')
