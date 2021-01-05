from sklearn import datasets
from sklearn.linear_model import LinearRegression

boston = datasets.load_boston()

lr = LinearRegression()
#. Pass the independent and dependent variables to the fit method of LinearRegression:
lr.fit(boston.data, boston.target)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1,normalize=False)
#Now, get the 10-fold cross-validated predictions

lr2 = LinearRegression(normalize=True) #object can automatically normalize (or scale) inputs
lr2.fit(boston.data, boston.target)
LinearRegression(copy_X=True, fit_intercept=True, normalize=True)
predictions2 = lr2.predict(boston.data)



