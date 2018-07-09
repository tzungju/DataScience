from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

#assigning predictor and target variables
df=pd.read_excel('data.xlsx')
y=df.iloc[:,0]
x=df.iloc[:,1:]

# Fit Model
clf = GaussianNB()
clf.fit(x,y)


dfTest=np.array([6, 130, 8]).reshape(-1,3)
print(clf.predict(dfTest))
print(clf.predict_proba(dfTest))