from sklearn import datasets
from sklearn.linear_model import LinearRegression

boston = datasets.load_boston()

lr = LinearRegression()
lr.fit(boston.data, boston.target)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1,normalize=False)
predictions = lr.predict(boston.data)

lr2 = LinearRegression(normalize=True) #object can automatically normalize (or scale) inputs
lr2.fit(boston.data, boston.target)
LinearRegression(copy_X=True, fit_intercept=True, normalize=True)
predictions2 = lr2.predict(boston.data)

