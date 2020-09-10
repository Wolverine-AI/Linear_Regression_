#importing libaries
import pandas as pd
import numpy as ny
import matplotlib.pyplot as plt 
import seaborn as sns
import os


df = pd.read_csv('USA_Housing.csv')
 
#print(df.columns)
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']

"""X=df['YearsExperience']
y=df['Salary']
"""
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=101)

from sklearn.linear_model import LinearRegression

testCheck = LinearRegression()

testCheck.fit(X_train,y_train)

y_pre = testCheck.predict(X_test)

print(testCheck.intercept_)
print(testCheck.coef_)

cdf=pd.DataFrame(testCheck.coef_,X.columns,columns=["Coef"])
print(cdf)

plt.scatter(y_test,y_pre)
plt.xlabel('Y_test')
plt.ylabel('Y_pre')
plt.legend('G')

sns.distplot(y_test-y_pre)


