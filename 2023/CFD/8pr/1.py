import numpy as np
import matplotlib.pyplot as plt
import math
import decimal
t=1000
dt=1/t
n=10
dx=4/n
dy=4/n
T_out=273
U=np.zeros((t+1,n+1,n+1),dtype=decimal.Decimal)
b=False
R=(1,3)
for j in range(int(R[0]/dy),int(R[1]/dy)+1):
    U[0][n][j]=353
for i in range(1,t):
    
    for j in range(1,n):
        U[i][j][1]=2*U[i][j][0]-T_out
        for k in range(1,n):
            U[i][j][k]=(U[i][j+1][k]+T_out)/2
            U[i+1][j][k]=(dt)*math.pow(1.5,2)*((U[i][j+1][k]-2*U[i][j][k]+U[i][j-1][k])/(dx**2)+(U[i][j][k+1]-2*U[i][j][k]+U[i][j][k-1])/(dy**2))+U[i][j][k]
            if not b and abs(U[i+1][j][k]-U[i][j][k])<10**-2 :
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
            U[i][n][k]=(T_out-U[i][n-1][k])/2
        U[i][j][n]=(T_out+U[i][j][n-1])/2

if not b:
    print(U)  

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    it=((n+1)**2)*(t+1)
    x = [i*dx for i in range(it)  ]
    y = [i*dy for i in range(it)  ]
    z = U
    # print(x,y)
    c = [i*dt for i in range(it)  ]

    img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
    fig.colorbar(img)
    plt.show()
