import numpy as np
import matplotlib.pyplot as plt
t=100
dt=1/t
n=18
dx=9/n
dy=9/n
l=9
M=0.01
u=1
v=0



U=np.zeros((t+1,n+1,n+1),dtype='float64')
b=False
for i in range(1,t):
    
    for j in range(1,n):
        
        for k in range(1,n):
            if k*dy>(2*l/3):
                U[i][0][k]=1
                # print('Hello world')
            else:
                U[i][0][k]=0
            U[i+1][j][k]=(dt)*((M*((U[i][j+1][k]-2*U[i][j][k]+U[i][j-1][k])/(dx**2)+(U[i][j][k+1]-2*U[i][j][k]+U[i][j][k-1])/(dy**2)))-v*(U[i][j][k+1]-U[i][j][k])/(dy)-u*(U[i][j+1][k]-U[i][j][k])/(dx))+U[i][j][k]
            
            if ( not b) and (abs(U[i+1][j][k]-U[i][j][k])<10**-5) and abs(U[i+1][j][k]-U[i][j][k])!=0  and str(U[i][j][k])[len(str(U[i][j][k]))-4]=='e'and U[i][j][k]!=0 and str(U[i+1][j][k]).split('.')[0]==str(U[i][j][k]).split('.')[0]:
                print(U)
                print("\nHELLOOOOO",i,U[i+1][j][k],U[i][j][k],abs(U[i+1][j][k]-U[i][j][k]))
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
            if k*dy>=l/3 and k*dy<=2*l/3:
                U[i][n][k]=0
            U[i][j+1][k]=U[i][j][k]
        
    U[i][0][n]=0

if not b:
    print(U)  

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D



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
