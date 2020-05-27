# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification


###############################################################################Random

data = pd.read_csv('../Kuggle/trainData4.csv')
data.describe(include='all')

######clean data if needed
cols = ['elvt','lon', 'city','prov','yr','da','mo','hr','smax','smin','dmax','dmin','hmax','hmin','yhr','wdsp','NEW']
data = data.drop(cols, axis=1)
y = data['temp']
data = data[['lat',    'stp',      'gbrd' , 'dewp',  'hmdy',  'gust' , 'xhr']]
#data.drop(data.columns[data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

data.dropna()
data.info()
print(data.shape)
print(data.info)
list(data.columns)

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3)
X_train.info()
print(y_train)

clf = RandomForestRegressor()
clf.fit(X_train, y_train)

y_pred_random = clf.predict(X_test)

from sklearn import metrics

print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred_random)))

test = pd.read_csv('../Kuggle/testData.csv')
cols = ['elvt','lon', 'city','prov','yr','da','mo','hr','smax','smin','dmax','dmin','hmax','hmin','yhr','wdsp','NEW']
test = test.drop(cols, axis=1)
#test.drop(test.columns[test.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
test = test[['lat',    'stp',      'gbrd' , 'dewp',  'hmdy',  'gust' , 'xhr']]

print(f'test: {test}')
#test  = pd.DataFrame(test)
y_test_pred = clf.predict(test)
print(y_test_pred)
pred = pd.DataFrame(y_test_pred, columns=['temp'])
pred["id"] = pred.index + 1
csv = pred.set_index('id')
csv.to_csv('predictions_random.csv')
