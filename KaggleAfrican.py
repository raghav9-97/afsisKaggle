import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('training.csv')
test= pd.read_csv('sorted_test.csv')

target = train[['Ca','P','pH','SOC','Sand']].values

PIDN = test['PIDN'].values

train.drop(['Ca','P','pH','SOC','Sand'],axis=1,inplace=True)
train.drop(['PIDN'],axis=1,inplace=True)

test.drop(['PIDN'],axis=1,inplace=True)

features = pd.DataFrame()
features['feature'] = train.columns

from sklearn.preprocessing import LabelEncoder
code1 = LabelEncoder()
train['Depth'] = code1.fit_transform(train['Depth'])
test['Depth'] = code1.transform(test['Depth'])

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_X = sc_X.fit(train)
train = sc_X.transform(train)
test = sc_X.transform(test)

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
regression = RandomForestRegressor(n_estimators=50,max_features='sqrt')
regression = regression.fit(train,target)

features['importance'] = regression.feature_importances_
features.sort_values(by=['importance'],ascending=True,inplace=True)
features.set_index('feature',inplace=True)

model = SelectFromModel(regression,prefit=True)
train_reduced = model.transform(train)
test_reduced = model.transform(test)

from sklearn.svm import SVR
reg1 = SVR(kernel='rbf')
reg1 = reg1.fit(train_reduced,target[:,0])
reg2 = SVR(kernel='rbf')
reg2 = reg2.fit(train_reduced,target[:, 1])
reg3 = SVR(kernel = 'rbf')
reg3 = reg3.fit(train_reduced,target[:, 2])
reg4 = SVR(kernel='rbf')
reg4 = reg4.fit(train_reduced,target[:, 3])
reg5 = SVR(kernel='rbf')
reg5 = reg5.fit(train_reduced,target[:, 4])

pred1 = reg1.predict(test_reduced)
pred2 = reg2.predict(test_reduced)
pred3 = reg3.predict(test_reduced)
pred4 = reg4.predict(test_reduced)
pred5 = reg5.predict(test_reduced)

pred = pd.DataFrame({'PIDN':PIDN,'Ca':pred1,'P':pred2,'pH':pred3,'SOC':pred4,'Sand':pred5},columns=['PIDN','Ca','P','pH','SOC','Sand'])
pred.to_csv('D:/Movies/afsis.csv',sep=',',header=True,index=False)

