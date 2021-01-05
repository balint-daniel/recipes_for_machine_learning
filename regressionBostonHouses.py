import warnings
warnings.filterwarnings("ignore")

# Cross Validation Regression MAE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT', 'MEDV']
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = LinearRegression()
scoring = 'neg_mean_absolute_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(f'MAE: {results.mean()} , {results.std()}')
print('------------------------------------------------')

# MSE
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(f'MSE: {results.mean()} , {results.std()}')
print('------------------------------------------------')

#R^2
scoring = 'r2'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(f'R^2: {results.mean()} , {results.std()}')
print('------------------------------------------------')

# Regression ALGORITHMSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS

# Linear ML regression algorithms: - linear regression, -ridge, -lasso, -elastic net regression

# Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(f'Linear Regression: {results.mean()}')
print('------------------------------------------------')

# Ridge Regression
from sklearn.linear_model import Ridge
model = Ridge()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(f'Ridge: {results.mean()}')
print('------------------------------------------------')

# Lasso Regression
from sklearn.linear_model import Lasso
model = Lasso()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(f'Lasso: {results.mean()}')
print('------------------------------------------------')

# ElasticNet Regression
from sklearn.linear_model import ElasticNet
model = ElasticNet()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(f'Elastic Net (L1+L2): {results.mean()}')
print('------------------------------------------------')

#NON-LINEAR REGRESSION ALGORITHMS:

# KNN Regression
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(f'KNN Regression: {results.mean()}')
print('------------------------------------------------')

# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(f'Decision Tree Regression: {results.mean()}')
print('------------------------------------------------')

# SVM Regression
from sklearn.svm import SVR
model = SVR()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(f'SVR: {results.mean()}')
print('------------------------------------------------')