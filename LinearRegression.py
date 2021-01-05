from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error

boston = datasets.load_boston()

lr = LinearRegression(normalize=True) #object can automatically normalize (or scale) inputs

#. Pass the independent and dependent variables to the fit method of LinearRegression:
lr.fit(boston.data, boston.target)

#Now, get the 10-fold cross-validated predictions
predictions_cv = cross_val_predict(lr, boston.data, boston.target,cv=10)

# get the errors
print('MAE: ', mean_absolute_error(boston.target, predictions_cv))
print('MSE: ', mean_squared_error(boston.target, predictions_cv))
