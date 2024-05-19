import numpy as np
import matplotlib.pyplot as plt
import math
dt=0.01
n=10
dx=2/n
l=[]
for i in range(101):
    l.append(i*dt)

U=np.zeros((101,11))
b=False
for i in range(1,100):
    U[i][0]=1
    for j in range(1,n):
        U[i+1][j]=(dt/(dx**2))*(U[i][j+1]-2*U[i][j]+U[i][j-1])-(dt/dx)*(U[i][j+1]-U[i][j])+U[i][j]
        if not b and abs(U[i+1][j]-U[i][j])<10**-4:
            print(U)
            print("\nHELLOOOOO",U[i+1][j],U[i][j])
            # xpoints = np.array(l)
            # Q=U.copy()
            # Q=Q.transpose()
            # # print(l,Q[i])
            # ypoints = np.array(Q[i])

            # plt.plot(xpoints, ypoints)
            # plt.show()

            b=True           
            break
   
    U[i][10]=0
print(U)    