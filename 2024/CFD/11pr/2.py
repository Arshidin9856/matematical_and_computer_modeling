import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


t=10000
dt=4
n=10
dx=8
v=dt/dx
w=4*(v**2)-v**4

l=np.zeros((t+1,n+1),dtype="float64")
y=[]
for i in range(t+1):
    y.append(i*dt)
x=[]
for i in range(n+1):
    x.append(i*dx)
b=1
i=0
# def subs(L_i,L_j,iter):
    
#     for i in range(n):
#         if abs(L_i[i]-L_j[i])>10**-5:
             
#           return True 
#     print('Steady')
#     return False
l[t][0]=1
while b>10**-5:
        
    l[i][0]=1

    temp=np.full(n+1,1,dtype="float64")
    temp1=np.full(n+1,1,dtype="float64")

    for j in range(1,n-1):
        if i ==0:
            l[0][j]=0
        temp1[j]=(1/2)*(l[i][j+1]+l[i][j])-(v/3)*(l[i][j+1]-l[i][j])
        temp[j]=l[i][j]-(2*v/3)*(temp1[j+1]-temp1[j-1])
        l[i+1][j]=l[i][j]-(v/24)*((-2)*l[i][j+2]+7*l[i][j+1]-7*l[i][j-1]+2*l[i][j-2])-(3*v/8)*(temp[j+1]-temp[j-1])-(w/24)*(l[i][j+2]-4*l[i][j+1]+6*l[i][j]-4*l[i][j-1]+l[i][j-2])

    l[i][n]=0
    i+=1
    b=max(abs(l[i]-l[i-1]))


print(l,b)   
x_mesh, t_mesh = np.meshgrid(x,y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x_mesh, t_mesh, l, cmap='viridis')

ax.set_xlabel('X-axis')
ax.set_ylabel('Time (t)')
ax.set_zlabel('f(x, t)')

plt.show()