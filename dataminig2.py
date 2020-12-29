import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing,svm
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris




style.use('fivethirtyeight')

df = pd.read_csv("/home/matin/Programming/ML/dataset.csv",header=None,sep='\n')
df = df[0].str.split(',', expand=True)
X= np.array(df.drop([4],1))
y=np.array(df[4])

X_train,X_test,y_trian,y_test= train_test_split(X,y,test_size=0.2,shuffle=True)
X1, y1 = load_iris(return_X_y=True)

"""clf=LinearRegression()
clf.fit(X_train,y_trian)
accuracy = clf.score(X_test,y_test)""" 

print(y1)