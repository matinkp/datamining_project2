import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing,svm
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier



style.use('fivethirtyeight')
df = pd.read_csv("/home/matin/Programming/ML/dataset.csv",header=None,sep='\n')
df = df[0].str.split(',', expand=True)
X= np.array(df.drop([4],1))
y=np.array(df[4])



"""yz=[]

categories = ['bus', 'suv', 'microbus', 'sedan','truck', 'minivan']
for i in y:
    for z in range(len(categories)) :       
        
        if i == categories[z]:
            yz.append(z)
            
            
yz = np.array(yz)""" 
    


X_train,X_test,y_trian,y_test= train_test_split(X,y,test_size=0.2,shuffle=True)

clf=RandomForestClassifier()
clf.fit(X_train,y_trian)
accuracy = clf.score(X_test,y_test)

print("%",accuracy*100)