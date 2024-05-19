import numpy as np
dt=0.01
dx=0.1
n=10
T=np.zeros((101,11))
for i in range(0,100):
    T[i][0]=1
    for j in range(1,n):
        T[i+1][j]=(dt/dx)*(-T[i][j+1]+T[i][j])+T[i][j]
    T[i][10]=0
print(T)    