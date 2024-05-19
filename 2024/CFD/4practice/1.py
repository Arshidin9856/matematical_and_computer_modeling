import numpy as np
dt=0.01
dx=0.5
n=10
l=np.zeros((101,11))
for i in range(0,100):
    l[i][0]=1
    for j in range(1,n):
        l[i+1][j]=(dt/(dx**2))*(l[i][j+1]-2*l[i][j]+l[i][j-1])+l[i][j]
    l[i][10]=0
print(l)    