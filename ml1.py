from statistics import mean 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6,7,8,9], dtype=np.float64)
ys = np.array([3,4,3,5,6,8,7,9,8], dtype=np.float64)

def best_fit_scope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys) ) /
    ( (mean(xs)*mean(xs)) - mean(xs*xs))  ) 

    b =  mean(ys) - m*(mean(xs))

    return m,b



m ,b= best_fit_scope(xs, ys)
regression_line = [(m*x + b) for x in xs]
plt.scatter(xs,ys)
plt.plot(xs,regression_line)
plt.show()


