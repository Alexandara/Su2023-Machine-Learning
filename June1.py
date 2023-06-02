import numpy as np
import pandas as pd
import sklearn
import bs4
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
#from sklearn.preprocessing import

x = np.array([1,2,3])
#print(x.mean())

california = fetch_california_housing()
#print(california.DESCR)
df = pd.DataFrame(california.data, columns=california.feature_names)
df['price'] = california.target
#print(df)

cm = df.corr() # Correlation
#print(cm)

X = df[['MedInc', 'HouseAge', 'AveRooms']]
# print(X)
y = df[['price']]
# print(y)
# print(X.describe())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# create a linear regression model using X_train and Y_train
model = SGDRegressor(random_state=42)
model.fit(X_train, y_train)
# print(model.coef_)
# print(model.intercept_)

# pred = model.intercept_ + model.coef_[0] * MedInc + model.coef_[1] * HouseAge + model.coef_[2] * AveRooms
y_pred = model.predict(X_test)
# print(y_pred)
# print(y_test)

rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test,y_pred)
print(rmse)
print(r2)