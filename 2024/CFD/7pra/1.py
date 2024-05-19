import numpy as np
import matplotlib.pyplot as plt
import math
import decimal
t=1000
dt=1/t
n=5
dx=1/n
dy=1/n

l=[]
for i in range(101):
    l.append(i*dt)

U=np.zeros((t+1,n+1,n+1),dtype=float)
b=False
for i in range(1,t):
    
    for j in range(1,n):
        U[i][j][0]=0
        for k in range(1,n):
            U[0][j][k]=0
            U[i][0][k]=1
            U[i+1][j][k]=(dt)*((U[i][j+1][k]-2*U[i][j][k]+U[i][j-1][k])/(dx**2)+(U[i][j][k+1]-2*U[i][j][k]+U[i][j][k-1])/(dy**2))+U[i][j][k]
            if not b and abs(U[i+1][j][k]-U[i][j][k])<10**-4 :
                print(U)
                print("\nHELLOOOOO",U[i+1][j][k],U[i][j][k])
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                
                x = [i*dx for i in range((n+1)**2)  ]
                y = [i*dy for i in range((n+1)**2)  ]
                z = U[i+1]
                # print(x,y)
                c = [i*dt for i in range((n+1)**2)  ]

                img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
                fig.colorbar(img)
                plt.show()
                b=True           
                break

            U[i][n][k]=0
        U[i][j][n]=0
        

if not b:print(U)  

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# it=(n+1)*(n+1)*(t+1)
# x = [i*dx for i in range(it)  ]
# y = [i*dy for i in range(it)  ]
# z = U
# # print(x,y)
# c = [i*dt for i in range(it)  ]

# img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
# fig.colorbar(img)
# plt.show()
